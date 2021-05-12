
# variable definitions
from config import *

# repository modules
import utils
from utils.misc import kNN

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
        batch_size = target.shape[0]
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
        out = out.sum() / batch_size  # batch average
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
        # this reformulation uses the sub-gradient (hence ensuring a valid behaviour at zero)
        L2 = weight_diff.norm(p="fro")

        return L2 / N


class LocalEuclideanSimilarity(LossFunction):
    def __init__(self, original_pos: torch.Tensor,
                       perturbed_pos: torch.Tensor,
                       edges: torch.Tensor):
        super().__init__(original_pos, perturbed_pos)
        # check input validity
        if original_pos.shape[-1] != 3:
            raise ValueError("Vertices positions must have shape [b,3,n]")
        if perturbed_pos.shape[-1] != 3:
            raise ValueError("Vertices positions must have shape [b,3,n]")
        if edges.shape[-1] != 2:
            raise ValueError("Edges must have shape [b,2,n]")

        self.neighborhood = NEIGHBORS
        self.batch_size = self.original_pos.shape[0]
        out = [kNN(pos=self.original_pos[batch, :, :],  #TODO try to find a way to automatically compute cut-off
                        edges=edges[batch, :],
                        neighbors_num=self.neighborhood,
                        cutoff=CUTOFF) for batch in range(0, original_pos.shape[0])]
        # convert to batch tensor
        out = [out[batch].unsqueeze(0) for batch in range(0, self.batch_size)]
        self.kNN = torch.cat(out)

    def __call__(self) -> torch.Tensor:
        n = self.original_pos.shape[1]  # vertex count
        pos = self.original_pos
        ppos = self.perturbed_pos

        flat_kNN = self.kNN.view(self.batch_size, -1)
        X = torch.cat([pos[batch, flat_kNN[batch, :]].view(-1, self.neighborhood, 3).unsqueeze(0)
                       for batch in range(0, self.batch_size)])
        Xr = torch.cat([ppos[batch, flat_kNN[batch, :]].view(-1, self.neighborhood, 3).unsqueeze(0)
                       for batch in range(0, self.batch_size)])
        dist = torch.norm(X - pos.view(self.batch_size, n, 1, 3), p=2, dim=-1)
        dist_r = torch.norm(Xr - ppos.view(self.batch_size, n, 1, 3), p=2, dim=-1)
        dist_loss = torch.nn.functional.mse_loss(dist, dist_r, reduction="mean")
        return dist_loss
