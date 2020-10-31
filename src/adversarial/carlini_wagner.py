from typing import Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tqdm
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as func
import torch_sparse as tsparse
from torch_geometric.data.data import Data
import scipy.sparse

import utils
from adversarial.base import AdversarialExample, Builder, LossFunction


# ------------------------------------------------------------------------------
class Logger(object):
    def __init__(self, adv_example: AdversarialExample, log_interval: int = 10):
        super().__init__()
        self.adv_example = adv_example
        self.log_interval = log_interval

    def reset(self): raise NotImplementedError()

    def log(self): raise NotImplementedError()


class EmptyLogger(Logger):
    def __init__(self, adv_example: AdversarialExample, log_interval: int = 10):
        super().__init__(adv_example=adv_example, log_interval=log_interval)

    def reset(self): return

    def log(self): return


class ValueLogger(Logger):
    def __init__(self,
                 adv_example: AdversarialExample,
                 value_functions: dict = {
                     "adversarial": lambda x: x.adversarial_loss().item(),
                     "similarity": lambda x: x.similarity_loss().item(),
                     "regularization": lambda x: x.regularization_loss().item()},
                 log_interval: int = 10):

        super().__init__(adv_example=adv_example, log_interval=log_interval)
        self.logged_values = dict()
        self.value_functions = value_functions
        self.iteration = 0

        # initialize logging metrics
        for n, f in self.value_functions.items():
            self.logged_values[n] = []

        for n, f in self.value_functions.items():
            self.value_functions[n] = f

    def reset(self):
        self.iteration = 0
        for func, values in self.logged_values.items():
            values.clear()

    def log(self):
        self.iteration += 1
        if self.log_interval != 0 and self.iteration % self.log_interval == 0:
            for n, f in self.value_functions.items():
                v = f(self.adv_example)
                self.logged_values[n].append(v)

    def show(self):
        plt.figure()
        X = [np.array(v) for v in self.logged_values.values()]
        for i, array in enumerate(X): plt.plot(array / array.max())
        legend = ["{}:{:.4g}".format(k, vs[-1]) for k, vs in self.logged_values.items()]
        plt.legend(legend)
        plt.show()


# ------------------------------------------------------------------------------
class CWAdversarialExample(AdversarialExample):
    def __init__(self,
                 pos: torch.Tensor,
                 edges: torch.LongTensor,
                 faces: torch.LongTensor,
                 classifier: torch.nn.Module,
                 target: int,  # NOTE can be None
                 adversarial_coeff: float,
                 regularization_coeff: float,
                 minimization_iterations: int,
                 learning_rate: float,
                 additional_model_args: dict,
                 true_y: torch.Tensor):  # fixed a bug of unexpected variable

        super().__init__(
            pos=pos, edges=edges, faces=faces,
            classifier=classifier, target=target,
            classifier_args=additional_model_args)
        # coefficients
        self.adversarial_coeff = torch.tensor([adversarial_coeff], device=self.device, dtype=self.dtype_float)
        self.regularization_coeff = torch.tensor([regularization_coeff], device=self.device, dtype=self.dtype_float)

        # other parameters
        self.minimization_iterations = minimization_iterations
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
            if (self.true_y == self.target):
                print('Given mesh class is equal to the target, continuing..')
                break
            # backpropagate
            loss.backward()
            optimizer.step()


class CWBuilder(Builder):
    USETQDM = "usetqdm"
    ADV_COEFF = "adversarial_coeff"
    REG_COEFF = "regularization_coeff"
    MIN_IT = "minimization_iterations"
    LEARN_RATE = "learning_rate"
    MODEL_ARGS = "additional_model_args"
    K_nn = "k_nearest_neighbors"
    NN_CUTOFF = "knn_cutoff_parameter"

    def __init__(self, search_iterations=1, animate=False):
        super().__init__()
        self.search_iterations = search_iterations
        self.animate = animate
        self._perturbation_factory = LowbandPerturbation
        self._adversarial_loss_factory = AdversarialLoss
        self._similarity_loss_factory = L2Similarity
        self._regularizer_factory = EmptyRegularizer
        self._logger_factory = EmptyLogger

    def set_perturbation(self, perturbation_factory):
        self._perturbation_factory = perturbation_factory
        return self

    def set_adversarial_loss(self, adv_loss_factory):
        self._adversarial_loss_factory = adv_loss_factory
        return self

    def set_similarity_loss(self, sim_loss_factory):
        self._similarity_loss_factory = sim_loss_factory
        return self

    def set_regularization_loss(self, regularizer_factory):
        self._regularizer_factory = regularizer_factory
        return self

    def set_logger(self, logger_factory):
        self._logger_factory = logger_factory
        return self

    def build(self, **args: dict) -> AdversarialExample:
        usetqdm = args.get(CWBuilder.USETQDM, False)
        self.adex_data[CWBuilder.MIN_IT] = args.get(CWBuilder.MIN_IT, 500)
        self.adex_data[CWBuilder.ADV_COEFF] = args.get(CWBuilder.ADV_COEFF, 1)
        self.adex_data[CWBuilder.REG_COEFF] = args.get(CWBuilder.REG_COEFF, 1)
        self.adex_data[CWBuilder.LEARN_RATE] = args.get(CWBuilder.LEARN_RATE, 1e-3)
        self.adex_data[CWBuilder.MODEL_ARGS] = args.get(CWBuilder.MODEL_ARGS, dict())
        self.K_nn = args.get(CWBuilder.K_nn, 140)
        self.cutoff = args.get(CWBuilder.NN_CUTOFF, 40)

        # exponential search variables
        start_adv_coeff = self.adex_data[CWBuilder.ADV_COEFF]
        range_min, range_max = 0, start_adv_coeff
        optimal_example = None
        exp_search = True  # flag used to detected whether it is the
        # first exponential search phase, or the binary search phase

        # start search
        for i in range(self.search_iterations):
            midvalue = (range_min + range_max) / 2
            c = range_max if exp_search else midvalue

            print("[{},{}] ; c={}".format(range_min, range_max, c))

            # create adversarial example
            self.adex_data[
                CWBuilder.ADV_COEFF] = c  # NOTE non-consistent state during execution (problematic during concurrent
            # programming)
            adex = CWAdversarialExample(**self.adex_data)

            adex.adversarial_loss = self._adversarial_loss_factory(adex)
            adex.perturbation = self._perturbation_factory(adex)
            adex.similarity_loss = self._similarity_loss_factory(adex, K=self.K_nn, cutoff=self.cutoff)
            adex.regularization_loss = self._regularizer_factory(adex)
            adex.logger = self._logger_factory(adex)
            adex.compute(usetqdm=usetqdm, animate=self.animate)

            # get perturbation
            r = adex.perturbation.r
            adex.adversarial_loss().item()

            # add true classs to adversarial object
            adex.y = self.adex_data['true_y']

            # update best estimation
            if adex.is_successful:
                optimal_example = adex

            # update loop variables
            if exp_search and not adex.is_successful:
                range_min = range_max
                range_max = range_max * 2
            elif exp_search and adex.is_successful:
                exp_search = False
            else:
                range_max = range_max if not adex.is_successful else midvalue
                range_min = midvalue if not adex.is_successful else range_min

        # reset the adversarial example to the original state
        self.adex_data[CWBuilder.ADV_COEFF] = start_adv_coeff

        # if unable to find a good c,r pair, return the best found solution
        is_successful = optimal_example is not None
        if not is_successful: optimal_example = adex
        return optimal_example


# ==============================================================================
# perturbation functions ------------------------------------------------------
class Perturbation(object):
    def __init__(self, adv_example: CWAdversarialExample):
        super().__init__()
        self._r = None
        self._adv_example = adv_example
        self._perturbed_pos_cache = None
        self.reset()

    @property
    def r(self): return self._r

    @property
    def adv_example(self): return self._adv_example

    def _reset(self):
        self._r = torch.zeros(
            [self.adv_example.vertex_count, 3],
            device=self.adv_example.device,
            dtype=self.adv_example.dtype_float,
            requires_grad=True)

    def reset(self):
        self._reset()
        self._perturbed_pos_cache = None

        def hook(grad): self._perturbed_pos_cache = None

        self.r.register_hook(hook)

    def _perturb_positions(self):
        pos, r = self.adv_example.pos, self.r
        return pos + r

    def perturb_positions(self):
        if self._perturbed_pos_cache is None:
            self._perturbed_pos_cache = self._perturb_positions()
        return self._perturbed_pos_cache


class LowbandPerturbation(Perturbation):
    EIGS_NUMBER = "eigs_num"

    def __init__(self, adv_example, eigs_num=50):
        self._eigs_num = eigs_num
        self._eigvals, self._eigvecs = utils.eigenpairs(
            adv_example.pos, adv_example.faces, K=eigs_num)
        super().__init__(adv_example)

    @property
    def eigs_num(self): return self._eigs_num

    @property
    def eigvals(self): return self._eigvals

    @property
    def eigvecs(self): return self._eigvecs

    def _reset(self):
        self._r: torch.Tensor = torch.zeros(
            [self.eigs_num, 3],
            device=self.adv_example.device,
            dtype=self.adv_example.dtype_float,
            requires_grad=True)

    def _perturb_positions(self):
        return self.adv_example.pos + self.eigvecs.matmul(self.r)


# ===============================================================================
# adversarial losses ----------------------------------------------------------
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


class LogSoftmaxAdversarialLoss(LossFunction):
    def __init__(self, adv_example: AdversarialExample):
        super().__init__(adv_example)

    def __call__(self) -> torch.Tensor:
        plogits = self.adv_example.perturbed_logits
        loss = -torch.nn.functional.log_softmax(plogits, dim=-1).view(-1)[self.adv_example.target]
        return loss


# regularizers ----------------------------------------------------------------
class EmptyRegularizer(LossFunction):
    def __init__(self, adv_example: AdversarialExample):
        super().__init__(adv_example)
        self._zero = torch.zeros(1, dtype=adv_example.dtype_float, device=adv_example.device)

    def __call__(self): return self._zero


class CentroidRegularizer(LossFunction):
    def __init__(self, adv_example: AdversarialExample):
        super().__init__(adv_example)

    def __call__(self):
        adv_centroid = torch.mean(self.adv_example.perturbed_pos, dim=0)
        centroid = torch.mean(self.adv_example.pos, dim=0)
        return torch.nn.functional.mse_loss(adv_centroid, centroid)


# similarity functions --------------------------------------------------------
class L2Similarity(LossFunction):
    def __init__(self, adv_example: AdversarialExample):
        super().__init__(adv_example)

    def __call__(self) -> torch.Tensor:
        diff = self.adv_example.perturbed_pos - self.adv_example.pos
        area_indices, area_values = self.adv_example.area
        weight_diff = diff * torch.sqrt(
            area_values.view(-1, 1))  # (sqrt(ai)*(xi-perturbed(xi)) )^2  = ai*(x-perturbed(xi))^2
        L2 = weight_diff.norm(
            p="fro")  # this reformulation uses the sub-gradient (hance ensuring a valid behaviour at zero)
        return L2


class LocalEuclideanSimilarity(LossFunction):
    def __init__(self, adv_example: AdversarialExample, K: int = 140,
                 cutoff: int = 40):  # was 30
        super().__init__(adv_example)
        self.neighborhood = K
        self.kNN = utils.misc.kNN(
            pos=self.adv_example.pos,
            edges=self.adv_example.edges,
            neighbors_num=self.neighborhood,
            cutoff=cutoff)  # was 5 TODO try to find a way to automatically compute cut-off

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


try:
    from knn_cuda import KNN


    def _grad_distances(ref, query, n, k, knn) -> torch.Tensor:  # NOTE output tensor shape [n,k,3]
        ref = ref.view(1, n, 3)
        query = query.view(1, n, 3)
        d, I = knn(ref=ref, query=query)
        diff = query.view(n, 1, 3) - ref[0, I.view(-1), :].view(n, k, 3)  # shape [n,k,3]
        # print((d.view(-1) - diff.norm(p=2,dim=-1).view(-1)).abs().max()) #NOTE check if correct
        return diff.view(n, k, 3), I


    class RAOChamferSimilarity(LossFunction):
        def __init__(self, adv_example: AdversarialExample):
            super().__init__(adv_example)
            self.knn = KNN(1, transpose_mode=True)

        def __call__(self):
            pos, ppos = self.adv_example.pos, self.adv_example.perturbed_pos
            n, k = self.adv_example.vertex_count, 1

            diff, _ = _grad_distances(ref=pos, query=ppos, n=n, k=1, knn=self.knn)
            term = torch.bmm(diff.view(n * k, 1, 3), diff.view(n * k, 3, 1)).mean()
            return term


    class RAOSmoothSimilarity(LossFunction):
        def __init__(self, adv_example: AdversarialExample, K: int, wp_threshold_coeff: float = 1):
            super().__init__(adv_example)
            self.K = K
            self.knn = KNN(K, transpose_mode=True)
            self.wp_threshold_coeff = wp_threshold_coeff

        def __call__(self):
            pos, ppos = self.adv_example.pos, self.adv_example.perturbed_pos
            n, k = self.adv_example.vertex_count, self.K

            diff, _ = _grad_distances(ref=pos, query=ppos, n=n, k=k, knn=self.knn)
            dists = torch.bmm(diff.view(n * k, 1, 3), diff.view(n * k, 3, 1)).view(n, k).mean(dim=1)
            wp_mask = dists >= self.wp_threshold_coeff * dists.std()
            dists_masked = dists * wp_mask
            term = dists_masked.mean()
            return term


    class ChamferSimilarity(LossFunction):
        def __init__(self, adv_example: AdversarialExample):
            super().__init__(adv_example)
            self.knn = KNN(1, transpose_mode=True)

        def __call__(self):
            pos, ppos = self.adv_example.pos, self.adv_example.perturbed_pos
            n, k = self.adv_example.vertex_count, 1

            diff, _ = _grad_distances(ref=pos, query=ppos, n=n, k=1, knn=self.knn)
            term1 = torch.bmm(diff.view(n * k, 1, 3), diff.view(n * k, 3, 1)).mean()

            diff, _ = _grad_distances(ref=ppos, query=pos, n=n, k=k, knn=self.knn)
            term2 = torch.bmm(diff.view(n * k, 1, 3), diff.view(n * k, 3, 1)).mean()
            return term1 + term2


    class HausdorffSimilarity(LossFunction):
        def __init__(self, adv_example: AdversarialExample):
            super().__init__(adv_example)
            self.knn = KNN(1, transpose_mode=True)

        def __call__(self):
            pos, ppos = self.adv_example.pos, self.adv_example.perturbed_pos
            n, k = self.adv_example.vertex_count, 1

            diff, _ = _grad_distances(ref=pos, query=ppos, n=n, k=k, knn=self.knn)
            loss = torch.bmm(diff.view(n * k, 1, 3), diff.view(n * k, 3, 1)).max()
            return loss


    class CurvatureSimilarity(LossFunction):
        def __init__(self, adv_example: AdversarialExample, neighbourhood=30):
            super().__init__(adv_example)
            self.k = neighbourhood
            self.knn = KNN(self.k + 1, transpose_mode=True)
            self.nn = KNN(1, transpose_mode=True)
            self.normals = utils.misc.pos_normals(adv_example.pos, adv_example.faces)
            self.curv = self._curvature(adv_example.pos, self.normals)

        def _curvature(self, pos, normals):
            n, k = self.adv_example.vertex_count, self.k
            diff, knn_idx = _grad_distances(ref=pos, query=pos, n=n, k=k + 1, knn=self.knn)
            normalized_diff = torch.nn.functional.normalize(diff, p=2, dim=-1)  # NOTE shape [N,k+1,3]

            cosine_sim = torch.bmm(
                normalized_diff.view(n, k + 1, 3),
                normals.view(n, 3, 1))

            abs_cosine_sim = cosine_sim.abs().view(n, k + 1)
            curvature = abs_cosine_sim[:, 1:].mean(
                dim=1)  # remove first column (all zeros) and compute the cosine mean for each column
            return curvature

        def __call__(self):
            pos, ppos = self.adv_example.pos, self.adv_example.perturbed_pos
            n = self.adv_example.vertex_count

            _, nn_idx = self.nn(ref=pos.view(1, n, 3), query=ppos.view(1, n, 3))
            perturbed_normals = self.normals[nn_idx.view(-1), :]
            diff = self.curv - self._curvature(ppos, perturbed_normals)
            loss = (diff ** 2).mean()
            return loss


    class GeoA3Similarity(LossFunction):
        def __init__(self, adv_example: AdversarialExample, lambda1: float = 0.1, lambda2: float = 1,
                     neighbourhood: int = 16):
            super().__init__(adv_example)
            self.curvature_loss = CurvatureSimilarity(adv_example=adv_example, neighbourhood=neighbourhood)
            self.hausdorff_loss = HausdorffSimilarity(adv_example=adv_example)
            self.chamfer_loss = ChamferSimilarity(adv_example=adv_example)
            self.lambda1 = torch.tensor(lambda1, device=adv_example.device, dtype=adv_example.dtype_float)
            self.lambda2 = torch.tensor(lambda2, device=adv_example.device, dtype=adv_example.dtype_float)

        def __call__(self):
            loss = self.chamfer_loss() + self.lambda1 * self.hausdorff_loss() + self.lambda2 * self.curvature_loss()
            return loss

except ImportError as e:
    pass


# ==============================================================================
# ------------------------------------------------------------------------------
def generate_adversarial_example(
        mesh: Data, classifier: Module, target: int,
        search_iterations=1,
        lowband_perturbation=True,
        adversarial_loss="carlini_wagner",
        similarity_loss="local_euclidean",
        animate=False,
        regularization="none", **args) -> CWAdversarialExample:
    builder = CWBuilder(search_iterations, animate).set_mesh(mesh.pos, mesh.edge_index.t(), mesh.face.t(), mesh.y)
    builder.set_classifier(classifier).set_target(target)

    # set type of perturbation
    if lowband_perturbation:
        eigs_num = args[LowbandPerturbation.EIGS_NUMBER]
        builder.set_perturbation(perturbation_factory=lambda x: LowbandPerturbation(x, eigs_num=eigs_num))
    else:
        builder.set_perturbation(perturbation_factory=Perturbation)

    # set type of adversarial loss
    if adversarial_loss == "carlini_wagner":
        builder.set_adversarial_loss(adv_loss_factory=AdversarialLoss)
    elif adversarial_loss == "log_softmax":
        builder.set_adversarial_loss(adv_loss_factory=LogSoftmaxAdversarialLoss)
    else:
        raise ValueError("Invalid adversarial loss!")

    # set type of similarity loss
    if similarity_loss == "local_euclidean":
        builder.set_similarity_loss(sim_loss_factory=LocalEuclideanSimilarity)
    elif similarity_loss == "l2":
        builder.set_similarity_loss(sim_loss_factory=L2Similarity)
    elif similarity_loss == "GeoA3":
        builder.set_similarity_loss(sim_loss_factory=GeoA3Similarity)
    else:
        raise ValueError("Invalid similarity loss!")

    # set type of regularizer
    if regularization != "none":
        if regularization == "centroid":
            builder.set_regularization_loss(regularizer_factory=CentroidRegularizer)
        elif regularization == "chamfer":
            builder.set_regularization_loss(regularizer_factory=ChamferSimilarity)
        else:
            raise ValueError("Invalid regularization term!")

    return builder.build(**args)