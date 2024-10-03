import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.figure(figsize=(9, 6))

    plt.scatter(x[:, 0], x[:, 1], 
                c='dodgerblue',     # Set point color
                s=20,               # Set point size
                alpha=0.7,          # Add transparency to points
                edgecolor='k',      # Black edge for contrast
                linewidth=0.1)      # Thin edge
    
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axis('off')

    plt.savefig(f"results/{dataset}/data.png", format='png')

def plot_density_and_samples(experiments, dataset):

    bins = 100
    lim = 4

    if dataset == 'moons': lim = 2
    hist_range = [[-lim, lim], [-lim, lim]]

    X, Y, grid = create_grid(lim)

    fig, ax = plt.subplots(2, len(experiments), figsize=(len(experiments) * 5, 10))

    if len(experiments) == 1:
        ax = np.expand_dims(ax, 1)

    for i in range(len(experiments)):
        model, hyperparameters, algorithm, ll = experiments[i]

        samples = model.sample(100000)

        K, lr, epochs = hyperparameters

        with torch.no_grad():
            density = torch.exp(model(grid.to(model.device)))

        ax[0, i].set_title(f'{algorithm} \n LL = {ll:.2f}', fontsize=20, pad=10, fontweight='bold')

        ax[0, i].contour(X, Y, density.cpu().reshape(X.shape), levels=100, cmap=plt.cm.inferno, linewidths=1.5)
        ax[0, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        ax[1, i].hist2d(samples.cpu()[:, 0], samples.cpu()[:, 1], range=hist_range, bins=bins, cmap=plt.cm.inferno)
        ax[1, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        ax[1, i].set_xlabel(f'K = {K}, lr = {lr}, epochs = {epochs}', fontsize=14, labelpad=10, fontweight='bold')

    # Set titles for the rows
    fig.text(0.09, 0.70, 'Densities', va='center', rotation='vertical', fontsize=25, fontweight='bold')
    fig.text(0.09, 0.30, 'Samples', va='center', rotation='vertical', fontsize=25, fontweight='bold')

    plt.subplots_adjust(wspace=0.001, hspace=0.001)
    plt.savefig(f"results/{dataset}/density_and_samples.png", format='png')
    
def plot_logp(logps, dataset):

    max_len = max([len(logp) for logp, algorithm in logps])

    data = []
    for logp, algorithm in logps:
        padded_logp = np.pad(logp, (0, max_len - len(logp)), constant_values=np.nan)
        data.extend([(epoch, algorithm, value) for epoch, value in enumerate(padded_logp)])

    df = pd.DataFrame(data, columns=['Epoch', 'Algorithm', 'NLL'])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(5, 4))
    sns.lineplot(data=df, x='Epoch', y='NLL', hue='Algorithm', dashes=False, linewidth=1.2, alpha=0.8)
    
    plt.xlabel('Epochs', fontsize=8, labelpad=8)
    plt.ylabel('NLL', fontsize=8, labelpad=8)
    plt.xlabel(None)
    plt.ylabel(None)

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    
    plt.legend(fontsize=6)
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    
    plt.tight_layout()  
    plt.savefig(f"results/{dataset}/logp.png", format='png', dpi=300)
