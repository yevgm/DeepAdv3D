import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.data.data import Data

import utils
from utils import laplacebeltrami_FEM, eigenpairs
from .base import AdversarialExample, Builder
from .carlini_wagner import Perturbation, LowbandPerturbation

class PGDBuilder(Builder):
    IT = "iterations"
    ALPHA = "alpha"
    CLASS_ARGS = "classifier_args"
    GROUND_TRUTH = "ground_truth"

    def __init__(self, ground_truth:int):
        super().__init__()
        self.perturbation_factory = Perturbation
        self.transforms = []
        self.adex_data[PGDBuilder.GROUND_TRUTH] = ground_truth

    def set_perturbation(self, perturbation_factory):
        self.perturbation_factory = perturbation_factory
        return self

    def set_transforms(self, transforms_factories:list):
        self.transforms = transforms_factories
        return self

    def build(self, **args) -> AdversarialExample:
        self.adex_data[PGDBuilder.IT] = args.get(PGDBuilder.IT, 1)
        self.adex_data[PGDBuilder.CLASS_ARGS] = args.get(PGDBuilder.CLASS_ARGS, dict())

        adex = PGDAdversarialExample(**self.adex_data)
        adex.perturbation = self.perturbation_factory(adex)
        adex.transforms = [f(adex) for f in self.transforms]
        adex.compute()
        return adex

class PGDAdversarialExample(AdversarialExample):
    def __init__(self, 
        pos, edges, faces, 
        classifier:torch.nn.Module, 
        iterations:int, 
        classifier_args:dict,
        ground_truth:int,
        target:int=None):
        super().__init__(
            pos=pos,edges=edges,faces=faces,classifier=classifier,target=target,classifier_args=classifier_args)
        self.ground_truth:torch.LongTensor = torch.tensor(ground_truth, device=pos.device, dtype=torch.long)
        self.iterations = iterations
        self.perturbation:Perturbation = None
        self.transforms = []

    @property
    def gradient(self):
        y = self.target if self.is_targeted else self.ground_truth
        z = self.perturbed_logits.view(1,-1)
        criterion = torch.nn.CrossEntropyLoss()

        loss = criterion(z, y)
        loss.backward()
        gradient = self.perturbation.r.grad.clone().detach_()
        
        # reset the state of the perturbation tensor
        delta = self.perturbation.r
        delta.detach_().requires_grad_(True)
        delta.grad.detach_()
        delta.grad.zero_()
        return gradient*(float(not self.is_targeted)*2 - 1)

    @property
    def perturbed_pos(self):
        return self.perturbation.perturb_positions()

    def compute(self):
        for i in range(self.iterations):
            g = self.gradient
            for f in self.transforms: g = f(g)

            # compute step
            self.perturbation.r.data = self.perturbation.r.data + g
            #self.perturbation.r.data += 0.003*torch.sign(self.gradient)
            #self.perturbation.r.data = self.perturbation.r.data + 0.01*self.gradient/self.gradient.norm(p=2,keepdim=True, dim=0)


def generate_adversarial_example(
    mesh:Data, classifier:Module, alpha:float,
    clip_transform="none", lowband_transform="none", gradient_transform="none", **args) -> PGDAdversarialExample:
    
    transforms = []
    if gradient_transform != "none":
        if gradient_transform == "sign": transforms.append(SignTransform)
        elif gradient_transform == "normalized":transforms.append(NormalizedTransform)
        elif gradient_transform == "l2": transforms.append(L2Transform)
        else: raise ValueError()
    
    transforms.append(lambda x:AlphaTransform(x,alpha=alpha))

    if lowband_transform != "none":
        eigs_num = args[LowBandTransform.EIGS_NUMBER]
        if lowband_transform =="static":
            transforms.append(lambda x: LowBandTransform(x,eigs_num=eigs_num))
        elif lowband_transform =="dynamic":
            transforms.append(lambda x: DynamicLowBandTransform(x,eigs_num=eigs_num))
        else: raise ValueError()

    if clip_transform != "none":
        epsilon = args[ClipTransform.EPSILON]
        if clip_transform == "pointwise":
            transforms.append(lambda x:ClipTransform(x,epsilon=epsilon))
        elif clip_transform == "norm":
            transforms.append(lambda x:ClipNormsTransform(x,epsilon=epsilon))
        else: raise ValueError()
    
    builder = PGDBuilder(ground_truth=mesh.y).set_mesh(mesh.pos,mesh.edge_index.t(), mesh.face.t())
    builder.set_classifier(classifier).set_transforms(transforms)
    return builder.build(**args)


#=============================================================================
#-----------------------------------------------------------------------------
class Transform:
    def __init__(self, adv_example:PGDAdversarialExample):
        super().__init__()
        self.adv_example = adv_example

    def __call__(self, x:Tensor) -> Tensor:
        raise NotImplementedError

class AlphaTransform(Transform):
    def __init__(self, adv_example:PGDAdversarialExample, alpha:float): 
        super().__init__(adv_example)
        self.alpha:Tensor = torch.tensor(alpha, device=adv_example.device, dtype=adv_example.dtype_float)

    def __call__(self, x:Tensor)->Tensor: return self.alpha*x

class SignTransform(Transform):
    def __init__(self, adv_example:PGDAdversarialExample): super().__init__(adv_example)
    def __call__(self, x:Tensor)->Tensor: return torch.sign(x)

class NormalizedTransform(Transform):
    def __init__(self, adv_example:PGDAdversarialExample):super().__init__(adv_example)
    def __call__(self, x:Tensor)->Tensor: return x/x.norm(p=2, dim=1, keepdim=True)

class L2Transform(Transform):
    def __init__(self, adv_example:PGDAdversarialExample):super().__init__(adv_example)
    def __call__(self, x:Tensor)->Tensor: return x/x.norm(p=2, dim=0, keepdim=True)


class ClipTransform(Transform):
    EPSILON="epsilon"
    def __init__(self, adv_example:PGDAdversarialExample, epsilon:float):
        super().__init__(adv_example)
        self.epsilon = torch.tensor(epsilon, device=self.adv_example.device, dtype=self.adv_example.dtype_float)

    def __call__(self, x:Tensor) -> Tensor:
        return torch.max(torch.min(x, self.epsilon), -self.epsilon)


class ClipNormsTransform(ClipTransform):
    def __init__(self, adv_example:PGDAdversarialExample, epsilon:float):
        super().__init__(adv_example, epsilon)

    def __call__(self, x:Tensor) -> Tensor:
        norms = x.norm(dim=1,p=2)
        mask = norms > self.epsilon
        x[mask] = x[mask]*self.epsilon/norms[mask].view(-1,1)
        return x

class LowBandTransform(Transform):
    EIGS_NUMBER="eigs_num"
    def __init__(self, adv_example:PGDAdversarialExample, eigs_num:int=50):
        super().__init__(adv_example)
        self.eigs_num = eigs_num
        #self.area = torch.diag(self.adv_example.area[1])
        self.area_pos = self.adv_example.area[1]
        self.eigvals, self.eigvecs = eigenpairs(
            self.adv_example.pos, self.adv_example.faces, K=eigs_num)

    def __call__(self, x:Tensor) -> Tensor:
        #x_spectral = self.eigvecs.t().mm(self.area.mm(x))
        x_spectral = self.eigvecs.t().mm(self.area_pos.view(-1,1)*x)
        x_filtered = self.eigvecs.mm(x_spectral)
        return x_filtered

import scipy
import scipy.sparse.linalg  as slinalg

class DynamicLowBandTransform(Transform):
    def __init__(self, adv_example:PGDAdversarialExample, eigs_num:int=50):
        super().__init__(adv_example)
        self.eigs_num = eigs_num

    def _eigenpairs(self):
        ppos = self.adv_example.perturbed_pos.clone().detach_()
        faces = self.adv_example.faces
        K = self.eigs_num
        double_precision = True

        n = ppos.shape[0]
        device = ppos.device
        dtype = ppos.dtype
        dtypFEM = torch.float64 if double_precision else ppos.dtype
        stiff, area, lump = laplacebeltrami_FEM(ppos.to(dtypFEM), faces)

        stiff = stiff.coalesce()
        area = area.coalesce()

        si, sv = stiff.indices().cpu(), stiff.values().cpu()
        ai, av = area.indices().cpu(), area.values().cpu()

        ri,ci = si
        S = scipy.sparse.csr_matrix( (sv, (ri,ci)), shape=(n,n))

        ri,ci = ai
        A = scipy.sparse.csr_matrix( (av, (ri,ci)), shape=(n,n))

        eigvals, eigvecs = slinalg.eigsh(S, M=A, k=K, sigma=-1e-6)
        eigvals = torch.tensor(eigvals, device=device, dtype=dtype)
        eigvecs = torch.tensor(eigvecs, device=device, dtype=dtype)
        lump = lump.to(ppos.dtype)
        return eigvals, eigvecs, lump

    def __call__(self, x:Tensor) -> Tensor:        
        eigvals, eigvecs, area_pos = self._eigenpairs()
        x_spectral = eigvecs.t().mm(area_pos.view(-1,1)*x)
        x_filtered = eigvecs.mm(x_spectral)
        return x_filtered
