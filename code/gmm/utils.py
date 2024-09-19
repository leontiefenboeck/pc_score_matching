import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
import json

def save_best_params(dataset, algorithm, best_params):
    
    filename='best_params.json'
    
    # Check if the file exists
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    if dataset not in data:
        data[dataset] = {}
    
    # Store the best parameters and score for the given algorithm
    data[dataset][algorithm] = {
        'K': best_params['K'],
        'lr': best_params['lr'],
        'epochs': best_params['epochs'],
        'll': best_params['ll']
    }
    
    # Save back to the file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_best_params(dataset, algorithm):

    filename = 'best_params.json'

    if not os.path.exists(filename):
        raise ValueError(f"File {filename} does not exist.")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Check if the dataset and algorithm exist in the file
    if dataset not in data or algorithm not in data[dataset]:
        raise ValueError(f"No parameters found for dataset: {dataset}, algorithm: {algorithm}")
    
    best_params = data[dataset][algorithm]

    return best_params['K'], best_params['lr'], best_params['epochs']

def create_grid(lim):

    x = torch.linspace(-lim, lim, 100)
    y = torch.linspace(-lim, lim, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.column_stack([X.flatten(), Y.flatten()])

    return (X, Y, grid)

def plot_data(x, dataset):

    bins = 100
    lim = 4

    if dataset == 'moons': lim = 2
    hist_range = [[-lim, lim], [-lim, lim]]

    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.hist2d(x[:, 0], x[:, 1], range=hist_range, bins=bins, cmap=plt.cm.inferno)

    plt.tight_layout()
    plt.savefig(f"results/{dataset}/ground_truth.png", format='png')

def plot_density_and_samples(models, dataset):

    bins = 100
    lim = 4

    if dataset == 'moons': lim = 2
    hist_range = [[-lim, lim], [-lim, lim]]

    X, Y, grid = create_grid(lim)

    fig, ax = plt.subplots(2, len(models), figsize=(len(models) * 5, 10))

    if len(models) == 1:
        ax = np.expand_dims(ax, 1)

    for i in range(len(models)):
        samples = models[i].sample(100000)

        K, lr, epochs = models[i].get_hyperparams()
        ll = models[i].get_ll()

        with torch.no_grad():
            density = torch.exp(models[i](grid))

        ax[0, i].set_title(f'{models[i].get_algorithm()} \n LL = {ll:.2f}', fontsize=20, pad=10, fontweight='bold')

        ax[0, i].contour(X, Y, density.reshape(X.shape), levels=100, cmap=plt.cm.inferno, linewidths=1.5)
        ax[0, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        ax[1, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax[1, i].hist2d(samples[:, 0], samples[:, 1], range=hist_range, bins=bins, cmap=plt.cm.inferno)

        ax[1, i].set_xlabel(f'K = {K}, lr = {lr}, epochs = {epochs}', fontsize=14, labelpad=10, fontweight='bold')

    # Set titles for the rows
    fig.text(0.09, 0.70, 'Densities', va='center', rotation='vertical', fontsize=25, fontweight='bold')
    fig.text(0.09, 0.30, 'Samples', va='center', rotation='vertical', fontsize=25, fontweight='bold')

    # Adjust spacing
    plt.subplots_adjust(wspace=0.001, hspace=0.001)
    plt.savefig(f"results/{dataset}/density_and_samples.png", format='png')

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
    
