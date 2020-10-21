from typing import Tuple
import networkx as nx
import numpy as np
import torch
from torch import LongTensor, Tensor
import torch_sparse as tsparse
import torch_scatter as tscatter
import torch_geometric
from torch_geometric.data.data import Data
import tqdm

from . import eigenpairs

def check_data(pos:torch.Tensor=None, edges:torch.Tensor=None, faces:torch.Tensor=None, float_type:type=None):
    # check input consistency 

    if pos is not None:
      if not torch.is_floating_point(pos): 
        raise ValueError("The vertices matrix must have floating point type!")
    
      if float_type is None: float_type = pos.dtype

      if (len(pos.shape)!= 2 or pos.shape[1] != 3) and pos.dtype != float_type:
        raise ValueError("The vertices matrix must have shape [n,3] and type {}!".format(float_type))

    if edges is not None and (len(edges.shape) != 2 or edges.shape[1] != 2 or edges.dtype != torch.long):
      raise ValueError("The edge index matrix must have shape [m,2] and type long!")
    
    if faces is not None and (len(faces.shape) != 2 or faces.shape[1] != 3 or faces.dtype != torch.long):
      raise ValueError("The edge index matrix must have shape [#faces,3] and type long!")

def prediction(classifier:torch.nn.Module, x:torch.Tensor):
  Z = classifier(x)
  prediction = Z.argmax()
  return prediction

def kNN(
  pos:torch.Tensor, 
  edges:torch.LongTensor,
  neighbors_num:int=256,
  cutoff:int=3):
  device = pos.device

  if len(pos.shape)!= 2 or pos.shape[1] != 3:
      raise ValueError("The vertices matrix must have shape [n,3] and type float!")
  if len(edges.shape) != 2 or edges.shape[1] != 2 or edges.dtype != torch.long:
      raise ValueError("The edge index matrix must have shape [m,2] and type long!")

  n = pos.shape[0]
  m = edges.shape[0]
  k = neighbors_num
  edge_index = edges.cpu().clone().detach().numpy() # they are all necessary unfortunately

  graph = nx.Graph()
  graph.add_nodes_from(range(n))
  graph.add_edges_from(edge_index)

  N = np.zeros([n,k], dtype=float)
  spiral = nx.all_pairs_shortest_path(graph, cutoff=cutoff)
  for node_idx, neighborhood in spiral:

    if len(neighborhood) < k:
      raise RuntimeError("Node {} has only {} neighbours, increase cutoff value!".format(node_idx, len(neighborhood)))

    for i, neighbour_idx in enumerate(neighborhood.keys()):
      if i >= k: break
      else: N[node_idx, i] = neighbour_idx
    
  node_neighbours_matrix = torch.tensor(N, device=device, dtype=torch.long)
  return node_neighbours_matrix


#-------------------------------------------------------------------------------------------------
def heat_kernel(eigvals:torch.Tensor, eigvecs:torch.Tensor, t:float) -> torch.Tensor:
    #hk = eigvecs.matmul(torch.diag(torch.exp(-t*eigvals)).matmul(eigvecs.t()))
    tmp = torch.exp(-t*eigvals).view(1,-1)
    hk = (tmp*eigvecs).matmul(eigvecs.t())
    return hk

def diffusion_distance(eigvals:torch.Tensor, eigvecs:torch.Tensor, t:float):
    n, k = eigvecs.shape
    device = eigvals.device
    dtype = eigvals.dtype
    
    hk = heat_kernel(eigvals, eigvecs,2*t)
    tmp = torch.diag(hk).repeat(n, 1)
    return tmp + tmp.t() -2*hk

def compute_dd_mse(pos, perturbed_pos, faces, K, t):
    eigvals1, eigvecs1 = eigenpairs(pos, faces, K)
    eigvals2, eigvecs2 = eigenpairs(perturbed_pos, faces, K)
    d1 = diffusion_distance(eigvals1,eigvecs1,t)
    d2 = diffusion_distance(eigvals2,eigvecs2,t)
    return torch.nn.functional.mse_loss(d1, d2)

#----------------------------------------------------------------------------------
def tri_areas(pos, faces):
    check_data(pos=pos, faces=faces)
    v1 = pos[faces[:, 0], :]
    v2 = pos[faces[:, 1], :]
    v3 = pos[faces[:, 2], :]
    v1 = v1 - v3
    v2 = v2 - v3
    return torch.norm(torch.cross(v1, v2, dim=1), dim=1) * .5

def pos_areas(pos, faces):
  check_data(pos=pos, faces=faces)
  n = pos.shape[0]
  m = faces.shape[0]
  triareas = tri_areas(pos, faces)/3
  posareas = torch.zeros(size=[n], device=triareas.device, dtype=triareas.dtype)
  for i in range(3):
    tmp = tscatter.scatter_add(triareas, faces[:,i], dim_size=n)
    posareas += tmp
  return posareas

#------------------------------------------------------------------------------
def tri_normals(pos, faces):
    check_data(pos=pos, faces=faces)
    v1 = pos[faces[:, 0], :]
    v2 = pos[faces[:, 1], :]
    v3 = pos[faces[:, 2], :]
    v1 = v1 - v3
    v2 = v2 - v3
    normals =  torch.cross(v1, v2, dim=1)
    return normals/normals.norm(p=2,dim=1,keepdim=True)

def pos_normals(pos, faces):
  check_data(pos=pos, faces=faces)
  n, m = pos.shape[0], faces.shape[0] 
  trinormals = tri_normals(pos, faces)
  posnormals = torch.zeros(size=[n, 3], device=trinormals.device, dtype=trinormals.dtype)
  for i in range(3):
    for j in range(3):
      posnormals[:,j] +=  tscatter.scatter_add(trinormals[:,j], faces[:,i], dim_size=n)
  return posnormals/posnormals.norm(p=2,dim=1,keepdim=True)


#-----------------------------------------------------------------------------
def l2_distance(pos, ppos, faces, normalize=False):
    check_data(pos=pos, faces=faces)
    check_data(pos=ppos)
    diff = pos - ppos
    areas = pos_areas(pos,faces)
    weight_diff = diff*torch.sqrt(areas.view(-1,1))
    L2 = weight_diff.norm(p="fro")
    if normalize: L2 = L2/areas.sum().sqrt()
    return L2

#------------------------------------------------------------------------------
def least_square_meshes(pos:Tensor, edges:LongTensor) -> Tensor:
    check_data(pos=pos, edges=edges)
    laplacian = torch_geometric.utils.get_laplacian(edges.t(), normalization="rw")
    n = pos.shape[2]
    tmp = tsparse.spmm(*laplacian, n, n, pos) #Least square Meshes problem 
    return (tmp**2).sum()

#--------------------------------------------------------------------------------
def write_obj(pos:Tensor,faces:Tensor, file:str):
    check_data(pos=pos, faces=faces)
    file = file if file.split(".")[-1] == "obj" else file + ".obj" # add suffix if necessary
    pos = pos.detach().cpu().clone().numpy();
    faces = faces.detach().cpu().clone().numpy();

    with open(file, 'w') as f:
        f.write("# OBJ file\n")
        for v in pos:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            
        for face in faces:
            f.write("f")
            for i in face:
                f.write(" %d" % (i + 1))
            f.write("\n")

def write_off(pos:Tensor,faces:Tensor, file:str):
    check_data(pos=pos, faces=faces)
    n, m = pos.shape[0], faces.shape[0]
    pos = pos.detach().cpu().clone().numpy();
    faces = faces.detach().cpu().clone().numpy();

    file = file if file.split(".")[-1] == "off" else file + ".off" # add suffix if necessary
    with open(file, 'w') as f:
        f.write("OFF\n")
        f.write("{} {} 0\n".format(n, m))
        for v in pos:
            f.write("{} {} {}\n".format(v[0], v[1], v[2]))
            
        for face in faces:
            f.write("3 {} {} {}\n".format(face[0],face[1],face[2]))

#---------------------------------------

try:
  from knn_cuda import KNN

  def knn_grad(ref, query, n, k) -> torch.Tensor: #NOTE output tensor shape [n,k,3]
      ref = ref.view(1,n,3)
      query = query.view(1,n,3)
      d, I = KNN(ref=ref, query=query)
      diff = query.view(n,1,3) - ref[0, I.view(-1),:].view(n,k,3) #shape [n,k,3]
      return diff.view(n,k,3), I

  def knn(ref, query, n, k) -> torch.Tensor:
        ref = ref.view(1,n,3)
        query = query.view(1,n,3)
        d, I = KNN(k, transpose_mode=True)(ref=ref, query=query)
        return d.view(n,k), I.view(n*k)

  def chamfer(ref, query):
    check_data(pos=ref)
    check_data(pos=query)

    n = ref.shape[0]
    nn_d1, nn_idx1 = knn_grad(ref=ref,query=query,n=n,k=1)
    nn_d2, nn_idx2 = knn_grad(ref=query,query=ref,n=n,k=1)

    chamfer1 = torch.bmm(nn_d1.view(n,1,3), nn_d1.view(n,3,1)).mean()
    chamfer2 = torch.bmm(nn_d2.view(n,1,3), nn_d2.view(n,3,1)).mean()
    return chamfer1+chamfer1

except ImportError as e:
    pass