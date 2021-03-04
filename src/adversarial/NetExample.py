from torch_geometric.data.data import Data
import scipy.sparse
import numpy as np
import tqdm
import torch
import utils
from adversarial.base import AdversarialExample, Builder, LossFunction


class NetExample:  # an adversarial example class that is created by a neural net
    def __init__(self,
                 pos: torch.Tensor,
                 edges: torch.LongTensor,
                 faces: torch.LongTensor,
                 target: int,  # NOTE can be None
                 true_y: torch.Tensor):

        super().__init__(
            pos=pos, edges=edges, faces=faces, target=target)

        # class components
        self.perturbation = None
        self.adversarial_loss = None
        self.similarity_loss = None
        self.regularization_loss = lambda: self._zero
        self._zero = torch.tensor(0, dtype=self.dtype_float, device=self.device)
        self.true_y = true_y
        self.animation_vertices = []
        self.animation_faces = []

    @property
    def perturbed_pos(self):
        return self.perturbation.perturb_positions()
