import torch
import torch_sparse as tsparse
from .laplacian import laplacebeltrami_FEM_v2

def meancurvature(pos, faces):
  if pos.shape[-1] != 3:
    raise ValueError("Vertices positions must have shape [n,3]")

  if faces.shape[-1] != 3:
    raise ValueError("Face indices must have shape [m,3]") 

  n = pos.shape[0]
  stiff, mass = laplacebeltrami_FEM_v2(pos, faces)
  ai, av = mass
  mcf = tsparse.spmm(ai, torch.reciprocal(av), n, n, tsparse.spmm(*stiff, n, n, pos))
  return mcf.norm(dim=-1, p=2), stiff, mass

def meancurvature_diff_l2(perturbed_pos, pos, faces):
  ppos = perturbed_pos
  mcp, _, _  = meancurvature(ppos, faces)
  mc, _, (_, a) = meancurvature(pos, faces)
  diff_curvature = mc-mcp
  a = a/a.sum()
  curvature_dist = (a*diff_curvature**2).sum().sqrt().item()
  return curvature_dist

def meancurvature_diff_abs(perturbed_pos, pos, faces):
  ppos = perturbed_pos
  mcp, _, _  = meancurvature(ppos, faces)
  mc, _, (_, a) = meancurvature(pos, faces)
  diff_curvature = mc-mcp
  a = a/a.sum()
  curvature_dist = (a*diff_curvature.abs()).sum().item()
  return curvature_dist
