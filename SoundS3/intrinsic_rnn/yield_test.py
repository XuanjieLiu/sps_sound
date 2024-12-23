import torch

def test(a, b):
    c = [a, b]
    yield c
    yield 2
    return c


if __name__ == '__main__':
    x = [1,2,3,4]
    tensor = torch.tensor(x)
    print(tensor)
    print(tensor[:tensor.size(0)-0])
