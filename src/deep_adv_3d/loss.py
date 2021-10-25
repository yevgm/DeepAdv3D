# repository modules
import utils
import torch
from utils.misc import kNN
from deep_adv_3d.laplacian import Laplacian, LaplacianFunc
# from deep_adv_3d.ido_laplacian import laplacian_batch
from pytorch3d.loss.mesh_laplacian_smoothing import mesh_laplacian_smoothing
from pytorch3d.loss.mesh_edge_loss import mesh_edge_loss
from pytorch3d.structures.meshes import Meshes
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Loss Functions
# ----------------------------------------------------------------------------------------------------------------------#


class LossFunction(object):  # TODO: make this more efficient after the bug's fixed
    def __init__(self, original_pos=0, perturbed_pos=0):    # 0 is default for adversarial example loss
        self.original_pos = original_pos
        self.perturbed_pos = perturbed_pos

    def __call__(self) -> torch.Tensor:
        raise NotImplementedError


class AdversarialLoss_single_batch(LossFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, perturbed_logits, target) -> torch.Tensor:
        Z = perturbed_logits
        values, index = torch.sort(Z)
        # index = index[-1]
        argmax = index[-1] if index[-1] != target else index[-2]  # max{Z(i): i != target}
        # Z = Z[-1]
        Ztarget, Zmax = Z[target], Z[argmax]
        return torch.nn.functional.relu(Zmax - Ztarget)

class AdversarialLoss(LossFunction):
    def __init__(self):
        '''
        t = target
        i = index of maximum value (that is not the target)
        Z(x) - logits of a classifier

        Loss function: f(x) = max{0, max{Z(i)-Z(t): i!=t}}

        The function models the difference between the maximum value of Z to the value of Z at the target
        When the diff is negative - that means the target has been reached
        '''
        super().__init__()


    def __call__(self, perturbed_logits, target) -> torch.Tensor:

        Z = perturbed_logits
        _, Zsorted_idx = torch.sort(Z, dim=1)
        Z_max_index = Zsorted_idx[:, -1]

        # if target equals the max of logits, take the second max. else do continue
        max_logit_equals_target_idx = Z_max_index == target
        second_max_value = Zsorted_idx[max_logit_equals_target_idx, -2]
        Z_max_index[max_logit_equals_target_idx] = second_max_value

        Ztarget = Z.gather(1, target.unsqueeze(1)).squeeze()
        Zmax = Z.gather(1, Z_max_index.unsqueeze(1)).squeeze()
        out = (Zmax - Ztarget)
        out = torch.nn.functional.relu(out)
        out = out.sum()

        # try from ido's blog
        # ce = torch.nn.CrossEntropyLoss()(perturbed_logits, target)
        # out = -1 * ce

        return out

class AdversarialLoss2(torch.nn.Module):
    def __init__(self):
        '''
        t = target
        F(x) - probabilities of a classifier

        Loss function: f(x) = max{0, 0.5 - F(t)}

        '''
        super().__init__()


    def forward(self, perturbed_logits, target) -> torch.Tensor:
        batch_size = target.shape[0]
        softmax = torch.nn.Softmax(dim=1)
        F = softmax(perturbed_logits)
        F_target = F.gather(1, target.unsqueeze(1)).squeeze()
        out = torch.nn.functional.relu(0.5 - F_target)
        out = out.sum() / batch_size  # batch average
        return out

# class AdversarialLoss(LossFunction):
#     def __init__(self):
#         '''
#         t = target
#         i = index of maximum value (that is not the target)
#         Z(x) - logits of a classifier
#
#         Loss function: f(x) = max{0, max{Z(i)-Z(t): i!=t}}
#
#         The function models the difference between the maximum value of Z to the value of Z at the target
#         When the diff is negative - that means the target has been reached
#         '''
#         super().__init__()
#
#
#     def __call__(self, perturbed_logits, target) -> torch.Tensor:
#         batch_size = target.shape[0]
#         Z = perturbed_logits
#         _, Zsorted_idx = torch.sort(Z, dim=1)
#         Z_max_index = Zsorted_idx[:, -1]
#
#         # if target equals the max of logits, take the second max. else do continue
#         max_logit_equals_target_idx = Z_max_index == target
#         second_max_value = Zsorted_idx[max_logit_equals_target_idx, -2]
#         Z_max_index[max_logit_equals_target_idx] = second_max_value
#
#         # Ztarget = Z[:, target].diag()
#         # Zmax = Z[:, Z_max_index].diag()
#         Ztarget = Z.gather(1, target.unsqueeze(1)).squeeze()
#         Zmax = Z.gather(1, Z_max_index.unsqueeze(1)).squeeze()
#         out = (Zmax - Ztarget)
#         # out[out <= 0] = 0
#         out = torch.nn.functional.relu(out)
#         out = out.sum() / batch_size  # batch average
#         return out

class L2Similarity(LossFunction):
    def __init__(self, original_pos: torch.Tensor,
                       perturbed_pos: torch.Tensor,
                       vertex_area: torch.Tensor=None):
        super().__init__(original_pos, perturbed_pos)
        self.vertex_area = vertex_area
        # check input validity
        if original_pos.shape[-2] != 3:
            raise ValueError("Vertices positions must have shape [b,3,n]")
        if perturbed_pos.shape[-2] != 3:
            raise ValueError("Vertices positions must have shape [b,3,n]")

    def __call__(self) -> torch.Tensor:

        diff = self.perturbed_pos - self.original_pos
        N = self.perturbed_pos.shape[0]  # batch size
        # (sqrt(ai)*(xi-perturbed(xi)) )^2  = ai*(x-perturbed(xi))^2
        weight_diff = diff[:, None, :] #   * torch.sqrt(self.vertex_area)
        # this reformulation uses the sub-gradient (hence ensuring a valid behaviour at zero)
        L2 = weight_diff.norm(p="fro")

        return L2 / N
#
class EdgeLoss(LossFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, e, pos, ppos):
        out = 0
        for b in range(e.shape[0]):
            e_single = e[b, :, :]
            ppos_single = ppos[b, :, :]
            pos_single = pos[b, :, :]
            reconstructed_norm = torch.norm(ppos_single[e_single[:, 0]] - ppos_single[e_single[:, 1]])
            original_norm = torch.norm(pos_single[e_single[:, 0]] - pos_single[e_single[:, 1]])
            out = out + (( (reconstructed_norm / original_norm) - 1).abs()).mean()
        return out


class LaplacianLoss(LossFunction):
    """
        Encourages minimal mean curvature shapes.
        """
    def __init__(self, faces, vert, toref=True):
        # Input:
        #  faces: B x F x 3
        self.toref = toref
        self.curve_gt = 0
        self.faces =  faces #torch.autograd.Variable(faces)
        # V x V
        self.laplacian = Laplacian()
        # self.Lx = None
        # # verts = torch.autograd.Variable(vert, requires_grad=True)
        # tmp = self.laplacian.apply(vert, self.faces)
        # self.curve_gt = torch.norm(tmp.view(-1, tmp.size(2)), p=2, dim=1).float()
        # if not self.toref:
        #     self.curve_gt = self.curve_gt * 0



    def __call__(self, verts):
        # vert = torch.autograd.Variable(verts, requires_grad=True)
        self.Lx = self.laplacian.apply(verts, self.faces)
        # Reshape to BV x 3
        Lx = self.Lx.view(-1, self.Lx.size(2))
        loss = (torch.norm(Lx, p=2, dim=1).float() - self.curve_gt).mean()
        return loss

# class LaplacianLossNoBackward(object):
#     """
#     Encourages minimal mean curvature shapes.
#     """
#
#     def __init__(self, faces, vert, toref=True):
#         # Input:
#         #  faces: B x F x 3
#         self.toref = toref
#         # V x V
#         self.curve_gt = 0
#         # self.Lx = None
#         # # tmp = LaplacianFunc(vert, faces)
#         # L, _ = laplacian_batch(vert, faces)
#         # tmp = torch.bmm(L, vert)
#         # self.curve_gt = torch.norm(tmp.view(-1, tmp.size(2)), p=2, dim=1).float()
#         # if not self.toref:
#         #     self.curve_gt = self.curve_gt * 0
#
#     def __call__(self, verts, faces):
#         L, _ = laplacian_batch(verts, faces)
#         Lx = torch.bmm(L, verts)
#         # Reshape to BV x 3
#         Lx = Lx.view(-1, Lx.size(2))
#         loss = (torch.norm(Lx, p=2, dim=1).float() - self.curve_gt).mean()
#         return loss

class MeshLaplacianSmoothing(LossFunction):
    """
    Encourages minimal mean curvature shapes.
    """
    def __call__(self, verts, faces):
        verts_list = []
        faces_list = []
        for b in range(verts.shape[0]):
            verts_list.append(verts[b,:,:])
            faces_list.append(faces[b, :, :])

        meshes = Meshes(verts_list, faces_list)
        loss = mesh_laplacian_smoothing(meshes=meshes, method='uniform')
        return loss


class MeshEdgeLoss(LossFunction):
    """
    Encourages minimal mean curvature shapes.
    """
    def __call__(self, verts, faces):
        verts_list = []
        faces_list = []
        for b in range(verts.shape[0]):
            verts_list.append(verts[b,:,:].T)
            faces_list.append(faces[b, :, :].T)

        meshes = Meshes(verts_list, faces_list)
        loss = mesh_edge_loss(meshes=meshes)
        return loss

# class LocalEuclideanSimilarity(LossFunction):
#     def __init__(self, original_pos: torch.Tensor,
#                        perturbed_pos: torch.Tensor,
#                        edges: torch.Tensor):
#         super().__init__(original_pos, perturbed_pos)
#         # check input validity
#         if original_pos.shape[-1] != 3:
#             raise ValueError("Vertices positions must have shape [b,3,n]")
#         if perturbed_pos.shape[-1] != 3:
#             raise ValueError("Vertices positions must have shape [b,3,n]")
#         if edges.shape[-1] != 2:
#             raise ValueError("Edges must have shape [b,2,n]")
#
#         self.neighborhood = NEIGHBORS
#         self.batch_size = self.original_pos.shape[0]
#         out = [kNN(pos=self.original_pos[batch, :, :],  #TODO try to find a way to automatically compute cut-off
#                         edges=edges[batch, :],
#                         neighbors_num=self.neighborhood,
#                         cutoff=CUTOFF) for batch in range(0, original_pos.shape[0])]
#         # convert to batch tensor
#         out = [out[batch].unsqueeze(0) for batch in range(0, self.batch_size)]
#         self.kNN = torch.cat(out)
#
#     def __call__(self) -> torch.Tensor:
#         n = self.original_pos.shape[1]  # vertex count
#         pos = self.original_pos
#         ppos = self.perturbed_pos
#
#         flat_kNN = self.kNN.view(self.batch_size, -1)
#         X = torch.cat([pos[batch, flat_kNN[batch, :]].view(-1, self.neighborhood, 3).unsqueeze(0)
#                        for batch in range(0, self.batch_size)])
#         Xr = torch.cat([ppos[batch, flat_kNN[batch, :]].view(-1, self.neighborhood, 3).unsqueeze(0)
#                        for batch in range(0, self.batch_size)])
#         dist = torch.norm(X - pos.view(self.batch_size, n, 1, 3), p=2, dim=-1)
#         dist_r = torch.norm(Xr - ppos.view(self.batch_size, n, 1, 3), p=2, dim=-1)
#         dist_loss = torch.nn.functional.mse_loss(dist, dist_r, reduction="mean")
#         return dist_loss


def chamfer_distance(original_pos: torch.Tensor, perturbed_pos: torch.Tensor):

    dist1, dist2, idx1, idx2 = dist_chamfer_aux(original_pos, perturbed_pos)  # mean over points
    out = torch.mean(dist1) + torch.mean(dist2)
    return out

def dist_chamfer_aux(original_pos: torch.Tensor,
                   perturbed_pos: torch.Tensor):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b:  Pointclouds Batch x nul_points x dim
    :return:
    -closest point on b of points from a
    -closest point on a of points from b
    -idx of closest point on b of points from a
    -idx of closest point on a of points from b
    Works for pointcloud of any dimension
    """
    a = original_pos
    b = perturbed_pos
    x, y = a.double(), b.double()
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()

    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x)  # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y)  # Diagonal elements yy
    P = rx.transpose(2, 1) + ry - 2 * zz
    return torch.min(P, 2)[0].float(), torch.min(P, 1)[0].float(), torch.min(P, 2)[1].int(), torch.min(P, 1)[1].int()

def CenterOfMassLoss(original_pos: torch.Tensor, perturbed_pos: torch.Tensor, run_config):
    orig_center = torch.sum(original_pos, dim=2, keepdim=True)
    adex_center = torch.sum(perturbed_pos, dim=2, keepdim=True)
    diff = orig_center - adex_center
    loss = diff.norm(p="fro")
    return loss


def LocalEuclideanBatch(original_pos: torch.Tensor, perturbed_pos: torch.Tensor, run_config):
        # check input validity
        # if original_pos.shape[-1] != 3:
        #     raise ValueError("Vertices positions must have shape [b,3,n]")
        # if perturbed_pos.shape[-1] != 3:
        #     raise ValueError("Vertices positions must have shape [b,3,n]")

        neighborhood = run_config['NEIGHBORS']
        knn_orig_dist, knn_orig_idx = batch_knn(x=original_pos, k=neighborhood)  # (batch_size, num_points, k)
        knn_adex_dist = dist_from_given_indices(x=perturbed_pos, indices=knn_orig_idx)

        diff = knn_orig_dist - knn_adex_dist
        l2 = diff.norm(p="fro")

        return l2 / neighborhood

def batch_knn(x, k):
    inner = torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = xx - 2 * inner + xx.transpose(2, 1)

    value, idx = pairwise_distance.topk(k=k, dim=-1, largest=False)  # (batch_size, num_points, k), [0] for values, [1] for indices
    return value, idx

def dist_from_given_indices(x, indices):
    inner = torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = xx - 2 * inner + xx.transpose(2, 1)

    value = torch.gather(pairwise_distance, dim=2, index=indices)
    return value
