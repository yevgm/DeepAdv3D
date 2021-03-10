
# variable definitions
from config import *

# repository modules
import utils
from utils.misc import kNN

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
    def __init__(self, perturbed_logits, target, k: float = 0):
        super().__init__()
        # check input validity
        if perturbed_logits.shape[-1] != DATASET_CLASSES:
            raise ValueError("must have a shape [b,DATASET_CLASSES]")

        self.k = torch.tensor([k], device=DEVICE, dtype=torch.float32)
        self.perturbed_logits = perturbed_logits
        self.target = target

    def __call__(self) -> torch.Tensor:
        batch_size = self.target.shape[0]
        Z = self.perturbed_logits
        values, index = torch.sort(Z, dim=1)
        argmax = index[:, -1].clone()

        # if target equals the max of logits, take the second max. else do continue
        argmax[index[:, -1] == self.target] = index[index[:, -1] == self.target, -2]

        Ztarget, Zmax = Z[:, self.target].diag(), Z[:, argmax].diag()
        out = (Zmax - Ztarget).clone()
        out[out <= -self.k] = 0
        return out.sum() / batch_size  # batch average


class L2Similarity(LossFunction):
    def __init__(self, original_pos: torch.Tensor,
                       perturbed_pos: torch.Tensor,
                       vertex_area: torch.Tensor):
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
        weight_diff = diff * torch.sqrt(self.vertex_area)[:, None, :]
        # this reformulation uses the sub-gradient (hance ensuring a valid behaviour at zero)
        L2 = weight_diff.norm(p="fro")
        return L2 / N

    #TODO: fix this loss
class LocalEuclideanSimilarity(LossFunction):
    def __init__(self, original_pos: torch.Tensor,
                       perturbed_pos: torch.Tensor,
                       edges: torch.Tensor):
        super().__init__(original_pos, perturbed_pos)
        self.neighborhood = NEIGHBORS
        self.kNN = kNN(pos=self.original_pos,
                        edges=edges,
                        neighbors_num=self.neighborhood,
                        cutoff=CUTOFF)  #TODO try to find a way to automatically compute cut-off

    def __call__(self) -> torch.Tensor:
        n = self.original_pos.shape[1] # vertex count # TODO: check if this works
        pos = self.original_pos
        ppos = self.perturbed_pos

        flat_kNN = self.kNN.view(-1)
        X = pos[flat_kNN].view(-1, self.neighborhood, 3)  # shape [n*K*3]
        Xr = ppos[flat_kNN].view(-1, self.neighborhood, 3)
        dist = torch.norm(X - pos.view(n, 1, 3), p=2, dim=-1)
        dist_r = torch.norm(Xr - ppos.view(n, 1, 3), p=2, dim=-1)
        dist_loss = torch.nn.functional.mse_loss(dist, dist_r, reduction="sum")
        return dist_loss
