# repository modules
import utils
import torch
from pytorch3d.loss.mesh_laplacian_smoothing import mesh_laplacian_smoothing
from pytorch3d.loss.mesh_edge_loss import mesh_edge_loss
from pytorch3d.structures.meshes import Meshes
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Loss Functions
# ----------------------------------------------------------------------------------------------------------------------#


class LossFunction(object):
    def __init__(self, original_pos=0, perturbed_pos=0):    # 0 is default for adversarial example loss
        self.original_pos = original_pos
        self.perturbed_pos = perturbed_pos

    def __call__(self) -> torch.Tensor:
        raise NotImplementedError


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
