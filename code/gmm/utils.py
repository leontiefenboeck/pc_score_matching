import torch
import matplotlib.pyplot as plt

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