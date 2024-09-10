import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import errno
from PIL import Image


def mkdir_p(path):
    """Linux mkdir -p"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def one_hot(x, K, dtype=torch.float):
    """One hot encoding"""
    with torch.no_grad():
        ind = torch.zeros(x.shape + (K,), dtype=dtype, device=x.device)
        ind.scatter_(-1, x.unsqueeze(-1), 1)
        return ind


def save_image_stack(samples, num_rows, num_columns, filename, margin=5, margin_gray_val=1., frame=0, frame_gray_val=0.0):
    """Save image stack in a tiled image"""

    # for gray scale, convert to rgb
    if len(samples.shape) == 3:
        samples = np.stack((samples,) * 3, -1)

    height = samples.shape[1]
    width = samples.shape[2]

    samples -= samples.min()
    samples /= samples.max()

    img = margin_gray_val * np.ones((height*num_rows + (num_rows-1)*margin, width*num_columns + (num_columns-1)*margin, 3))
    for h in range(num_rows):
        for w in range(num_columns):
            img[h*(height+margin):h*(height+margin)+height, w*(width+margin):w*(width+margin)+width, :] = samples[h*num_columns + w, :]

    framed_img = frame_gray_val * np.ones((img.shape[0] + 2*frame, img.shape[1] + 2*frame, 3))
    framed_img[frame:(frame+img.shape[0]), frame:(frame+img.shape[1]), :] = img

    img = Image.fromarray(np.round(framed_img * 255.).astype(np.uint8))

    img.save(filename)


def sample_matrix_categorical(p):
    """Sample many Categorical distributions represented as rows in a matrix."""
    with torch.no_grad():
        cp = torch.cumsum(p[:, 0:-1], -1)
        rand = torch.rand((cp.shape[0], 1), device=cp.device)
        rand_idx = torch.sum(rand > cp, -1).long()
        return rand_idx


def create_grid(x_lim = (-1.5, 2.5), y_lim = (-1, 1.5)):
    x = torch.linspace(x_lim[0], x_lim[1], 100)
    y = torch.linspace(y_lim[0], y_lim[1], 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.column_stack([X.flatten(), Y.flatten()])

    return (X, Y, grid)

def plot_eval(model, grid, algorithm):
    fig_model, ax_model = plt.subplots(2, 2, figsize=(10, 8))
    fig_model.suptitle(algorithm)

    plot_loss(model, ax_model[0, 0])
    plot_logp(model, ax_model[0, 1])
    plot_samples(model, ax_model[1, 0])
    plot_density(model, grid, ax_model[1, 1])

    return fig_model

def plot_samples(model, ax=plt):

    samples = model.sample(500)
    ax.set_title("Samples")
    ax.scatter(samples[:, 0], samples[:, 1])

def plot_density(model, grid, ax=plt):

    X, Y, grid = grid

    with torch.no_grad():
        density = torch.exp(model(grid))

    ax.set_title("Density")
    ax.contour(X, Y, density.reshape(X.shape), levels=200)

def plot_loss(model, ax=plt):
    ax.set_title("loss")
    ax.plot(model.get_loss())

def plot_logp(model, ax):
    ax.set_title("logp")
    ax.plot(model.get_logp())