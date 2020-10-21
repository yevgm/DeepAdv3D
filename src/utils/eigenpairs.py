
import scipy
import scipy.sparse.linalg  as slinalg
import torch

from .laplacian import laplacebeltrami_FEM

def eigenpairs(pos:torch.Tensor, faces:torch.Tensor, K:int, double_precision:bool=False):
    r"""Compute first K eigenvalues and eigenvectors for the input mesh.
    
    """
    if pos.shape[-1] != 3:
        raise ValueError("Vertices positions must have shape [n,3]")
    if faces.shape[-1] != 3:
        raise ValueError("Face indices must have shape [m,3]")
    
    n = pos.shape[0]
    device = pos.device
    dtype = pos.dtype
    dtypFEM = torch.float64 if double_precision else pos.dtype
    stiff, area, lump = laplacebeltrami_FEM(pos.to(dtypFEM), faces)

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
    return eigvals, eigvecs