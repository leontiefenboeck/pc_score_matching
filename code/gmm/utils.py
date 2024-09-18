import torch
import matplotlib.pyplot as plt

def create_grid(lim):

    x = torch.linspace(-lim, lim, 100)
    y = torch.linspace(-lim, lim, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.column_stack([X.flatten(), Y.flatten()])

    return (X, Y, grid)

def plot_data(x, dataset, bins, range_lim):

    if dataset == 'moons': range_lim = 2
    rang = [[-range_lim, range_lim], [-range_lim, range_lim]]

    plt.suptitle(f'Ground Truth')
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.hist2d(x[:, 0], x[:, 1], range=rang, bins=bins, cmap=plt.cm.plasma, vmin=0, vmax=3)

    plt.savefig(f"results/{dataset}/ground_truth.png", format='png')

def plot_samples(models, num_samples, dataset, bins, range_lim):

    if dataset == 'moons': range_lim = 2
    rang = [[-range_lim, range_lim], [-range_lim, range_lim]]

    fig, ax = plt.subplots(1, len(models), figsize=(len(models) * 5, 5))
    fig.suptitle("Samples")

    if len(models) == 1:
        ax = [ax]

    for i in range(len(models)):
        samples = models[i].sample(num_samples)
        # ax[i].set_facecolor('black')
        ax[i].set_title(f'{models[i].get_algorithm()}')
        ax[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax[i].hist2d(samples[:, 0], samples[:, 1], range=rang, bins=bins, cmap=plt.cm.plasma, vmin=0, vmax=3)

    plt.subplots_adjust(wspace=0.1) 
    plt.savefig(f"results/{dataset}/samples.png", format='png')

def plot_density(models, dataset):

    lim = 4
    if dataset == 'moons': lim = 2
    X, Y, grid = create_grid(lim)

    fig, ax = plt.subplots(1, len(models), figsize=(len(models) * 5, 5))
    fig.suptitle("Density")

    if len(models) == 1:
        ax = [ax]

    for i in range(len(models)):
        with torch.no_grad():
            density = torch.exp(models[i](grid))

        ax[i].set_facecolor('black')
        ax[i].set_title(f'{models[i].get_algorithm()}')
        ax[i].contour(X, Y, density.reshape(X.shape), levels=100, cmap=plt.cm.plasma)
        ax[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.subplots_adjust(wspace=0.1) 
    plt.savefig(f"results/{dataset}/densities.png", format='png')

def plot_losses(models, dataset):
    
    fig, ax = plt.subplots(len(models), 2, figsize=(10, 5 * len(models)))
    fig.suptitle('Losses vs. Logp')

    if len(models) == 1:
        ax = [ax]

    for i in range(len(models)):
        algorithm = models[i].get_algorithm()
        ax[i][0].set_title(f'{algorithm} Loss')
        ax[i][0].plot(models[i].get_loss())
        ax[i][1].set_title(f'{algorithm} negative Log Likelihood')
        ax[i][1].plot(models[i].get_logp())

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f"results/{dataset}/losses.png", format='png')
    
