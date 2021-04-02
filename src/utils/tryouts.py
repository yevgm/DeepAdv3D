import torch

# differentiable sorting in torch
# for visualizations: https://www.desmos.com/calculator/rqjpdu6q54


def sigmoid(x, k):
    tensor_of_2 = torch.tensor([2]).type(torch.FloatTensor)
    tensor_of_k = torch.tensor([-k]).type(torch.FloatTensor)
    p = (torch.log10(tensor_of_2)) / (torch.log10(1 + torch.exp(tensor_of_k)))
    out = 1 + torch.exp(-k*x)
    out = torch.pow(out, -p)
    return out


def compute_sigmoid_matrix(vec: torch.Tensor, k):
    n = vec.shape[0]
    M = torch.zeros(n, n)
    for i in range(n):  # TODO: can flatten all [i,j] pairs and compute this without loops
        for j in range(n):
            ratio = vec[i] / vec[j]
            M[i, j] = sigmoid(ratio, k)
    M = M + 0.5 * torch.eye(n)
    return M


def sigmoid_torch_sort(vec: torch.Tensor, k):
    M = compute_sigmoid_matrix(vec, k)
    ranks_vec = torch.sum(M, 1)
    return ranks_vec


def gaussian(vec, center, var):
    pi = torch.acos(torch.zeros(1)).item() * 2  # ugly way of computing torch.pi since it doesn't exist
    const = torch.pow(torch.tensor([2*pi*var]), -0.5)
    distance = torch.pow(vec - center, 2) / (2*var)
    out = const * torch.exp(-distance)
    out = out / const
    return out


def sample_by_rank(vec, ranks_vec, rank, var=0.05):
    weights = gaussian(ranks_vec, rank, var=var)
    out = torch.sum(weights * vec)
    return out


if __name__ == "__main__":
    k = 10
    rank = 3
    a = torch.Tensor([7, 11, 15, 3])
    print("vector to sort: ", a)
    ranksvec = sigmoid_torch_sort(a, k)
    print("ranks of the vector: ", ranksvec)
    sample = sample_by_rank(a, ranksvec, rank, var=0.001)
    print("component of rank {} in the vec: {} ".format(rank, sample))