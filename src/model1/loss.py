# variable definitions
from config import *

# repository modules
import utils
from adversarial.base import AdversarialExample, LossFunction


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Classes Definition
# ----------------------------------------------------------------------------------------------------------------------#

class AdversarialLoss(LossFunction):
    def __init__(self, adv_example: AdversarialExample, k: float = 0):
        super().__init__(adv_example)
        self.k = torch.tensor([k], device=adv_example.device, dtype=adv_example.dtype_float)

    def __call__(self) -> torch.Tensor:
        Z = self.adv_example.perturbed_logits
        values, index = torch.sort(Z, dim=1)
        index = index[-1]
        argmax = index[-1] if index[-1] != self.adv_example.target else index[-2]  # max{Z(i): i != target}
        Z = Z[-1]
        Ztarget, Zmax = Z[self.adv_example.target], Z[argmax]
        return torch.max(Zmax - Ztarget, -self.k)


class LocalEuclideanSimilarity(LossFunction):
    def __init__(self, adv_example: AdversarialExample, K: int = 140,
                 cutoff: int = 40):
        super().__init__(adv_example)
        self.neighborhood = K
        self.kNN = utils.misc.kNN(
            pos=self.adv_example.pos,
            edges=self.adv_example.edges,
            neighbors_num=self.neighborhood,
            cutoff=cutoff)

    def __call__(self) -> torch.Tensor:
        n = self.adv_example.vertex_count
        pos = self.adv_example.pos
        ppos = self.adv_example.perturbed_pos

        flat_kNN = self.kNN.view(-1)
        X = pos[flat_kNN].view(-1, self.neighborhood, 3)  # shape [n*K*3]
        Xr = ppos[flat_kNN].view(-1, self.neighborhood, 3)
        dist = torch.norm(X - pos.view(n, 1, 3), p=2, dim=-1)
        dist_r = torch.norm(Xr - ppos.view(n, 1, 3), p=2, dim=-1)
        dist_loss = torch.nn.functional.mse_loss(dist, dist_r, reduction="sum")
        return dist_loss








# class LossFunction:
#
#     def __init__(self):
#         pass
#
#     def local_euclidean_similarity_loss(self):
#         pass
#
#     def mis_classification_loss(self):
#         pass
