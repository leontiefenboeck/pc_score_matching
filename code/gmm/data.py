from sklearn.datasets import make_moons
import torch


def get_2d(dataset, n_samples, seed=42):

    z = torch.randn(n_samples, 2)

    if dataset == 'moon':
        data, target = make_moons(n_samples=500, noise=0.1, random_state=seed)
        return data 

    # from https://github.com/Ending2015a/toy_gradlogp/blob/master/toy_gradlogp/data.py
    if dataset == 'spirals':
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * torch.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y =   torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
        return x + 0.1*z

    # from https://github.com/Ending2015a/toy_gradlogp/blob/master/toy_gradlogp/data.py
    if dataset == 'board':
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
        x2 = x2_ + x1.floor() % 2
        return torch.stack([x1, x2], dim=1) * 2

