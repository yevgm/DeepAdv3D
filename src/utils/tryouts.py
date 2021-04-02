import torch

# differentiable sorting in torch


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


def sigmoid_torch_sort(vec : torch.Tensor, k):
    M = compute_sigmoid_matrix(vec, k)
    ranks_vec = torch.sum(M, 1)
    return ranks_vec


if __name__ == "__main__":
    k = 10
    a = torch.Tensor([2, 3, 5, 1])
    ranks_vec = sigmoid_torch_sort(a, k)
    print(ranks_vec)