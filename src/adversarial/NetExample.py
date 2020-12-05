from torch_geometric.data.data import Data
import scipy.sparse
import numpy as np
import tqdm
import torch
import utils
from adversarial.base import AdversarialExample, Builder, LossFunction


class NetExample(AdversarialExample):  # an adversarial example class that is created by a neural net
    def __init__(self,
                 pos: torch.Tensor,
                 edges: torch.LongTensor,
                 faces: torch.LongTensor,
                 classifier: torch.nn.Module,
                 target: int,  # NOTE can be None
                 adversarial_coeff: float,
                 regularization_coeff: float,
                 learning_rate: float,
                 additional_model_args: dict,
                 true_y: torch.Tensor):

        super().__init__(
            pos=pos, edges=edges, faces=faces,
            classifier=classifier, target=target,
            classifier_args=additional_model_args)
        # coefficients
        self.adversarial_coeff = torch.tensor([adversarial_coeff], device=self.device, dtype=self.dtype_float)
        self.regularization_coeff = torch.tensor([regularization_coeff], device=self.device, dtype=self.dtype_float)

        # other parameters
        self.learning_rate = learning_rate
        self.model_args = additional_model_args

        # for untargeted-attacks use second most probable class
        if target is None:
            values, index = torch.sort(self.logits, dim=0)
            self._target = index[-2]

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

    def compute(self, usetqdm: str = None, patience=3, animate=False):
        # reset variables
        self.perturbation.reset()
        self.logger.reset()

        # compute gradient w.r.t. the perturbation
        optimizer = torch.optim.Adam([self.perturbation.r], lr=self.learning_rate,
                                     betas=(0.9, 0.999))  # betas=(0.5,0.75))

        if usetqdm is None or usetqdm == False:
            iterations = range(self.minimization_iterations)
        elif usetqdm == "standard" or usetqdm == True:
            iterations = tqdm.trange(self.minimization_iterations)
        elif usetqdm == "notebook":
            iterations = tqdm.tqdm_notebook(range(self.minimization_iterations))
        else:
            raise ValueError("Invalid input for 'usetqdm', valid values are: None, 'standard' and 'notebook'.")

        flag, counter = False, patience
        last_r = self.perturbation.r.data.clone();

        for i in iterations:
            # compute loss
            optimizer.zero_grad()

            # compute total loss
            similarity_loss = self.similarity_loss()
            adversarial_loss = self.adversarial_coeff * self.adversarial_loss()
            regularization_loss = self.regularization_coeff * self.regularization_loss()
            loss = adversarial_loss + similarity_loss + regularization_loss
            self.logger.log()  # log results

            # add the perturbed pos to the animation list
            if animate:
                self.animation_vertices.append(self.perturbed_pos)
                self.animation_faces.append(self.faces)

            # cutoff procedure to improve performance
            is_successful = adversarial_loss <= 0
            if is_successful:
                counter -= 1
                if counter <= 0:
                    last_r.data = self.perturbation.r.data.clone()
                    flag = True
            else:
                counter = patience
            # Debug:
            # flag=True
            # is_successful = False
            if (flag and not is_successful):
                self.perturbation.reset()  # NOTE required to clean cache
                self.perturbation.r.data = last_r
                break  # cutoff policy used to speed-up the tests

            # backpropagate
            loss.backward()
            optimizer.step()
