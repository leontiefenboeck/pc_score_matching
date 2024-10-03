from sklearn.datasets import make_moons
import torch

def center(data):

    mean = data.mean(axis=0)
    data_centered = data - mean

    return data_centered

def get_2d(dataset, n_samples, seed=42):
    z = torch.randn(n_samples, 2)

    if dataset == 'halfmoons':
        data, target = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
        return center(data)

    # from https://github.com/Ending2015a/toy_gradlogp/blob/master/toy_gradlogp/data.py
    if dataset == 'spirals':
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * torch.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y =   torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
        x = x + 0.1*z
        return center(x.numpy())  

    # from https://github.com/Ending2015a/toy_gradlogp/blob/master/toy_gradlogp/data.py
    if dataset == 'board':
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
        x2 = x2_ + x1.floor() % 2
        x = torch.stack([x1, x2], dim=1) * 2
        return center(x.numpy()) 
