import torch
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os 
import json
import time

def sm_loss(model, x):
    x.requires_grad_()

    logp = model(x).sum()

    score = autograd.grad(logp, x, create_graph=True)[0]
    loss1 = 0.5 * torch.norm(score, dim=-1) ** 2
    
    trace = torch.ones_like(loss1)
    for i in range(x.size(1)):
        trace += autograd.grad(score[:, i].sum(), x, create_graph=True)[0][:, i]
    
    loss = torch.mean(loss1 + trace)
    return loss
    
def ssm_loss(model, x, n_slices):
    x.requires_grad_()

    x = x.unsqueeze(0).expand(n_slices, *x.shape).contiguous().view(-1, *x.shape[1:])

    v = torch.randn_like(x).to(x.device)

    logp = model(x).sum()

    score = autograd.grad(logp, x, create_graph=True)[0]
    loss1 = 0.5 * (torch.norm(score, dim=-1) ** 2)

    grad2 = torch.autograd.grad(torch.sum(score * v), x, create_graph=True)[0]
    loss2 = torch.sum(v * grad2, dim=-1)

    return (loss1 + loss2).mean()

def train(model, x, algorithm, lr, epochs, n_slices):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  

    logps = []
    for _ in range(epochs):

        logps.append(-model.log_likelihood(x))

        if algorithm == 'EM':
            model.EM_step(x)
        else: 
            if algorithm == "GD":
                loss = -torch.mean(model(x))
            if algorithm == "SSM":
                loss = ssm_loss(model, x, n_slices)
            if algorithm == "SM":
                loss = sm_loss(model, x)
     
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if i % (epochs / 5) == 0: print(f'[{algorithm:{3}}]   [{i:{3}}] Train Log-Likelihood = {logps[-1]:.4f}')

    logps.append(-model.log_likelihood(x))
    # print(f'[{algorithm:{3}}]   [{epochs:{3}}] Train Log-Likelihood = {logps[-1]:.4f}')

    return logps

def save_best_params(dataset, algorithm, K, best_params):
    
    filename='best_params.json'
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    if dataset not in data: data[dataset] = {}
    if str(K) not in data[dataset]: data[dataset][str(K)] = {}
    
    data[dataset][str(K)][algorithm] = {
        'lr': best_params['lr'],
        'epochs': best_params['epochs']
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_best_params(dataset, algorithm, K):

    filename = 'best_params.json'

    if not os.path.exists(filename):
        raise ValueError(f"File {filename} does not exist.")
    
    with open(filename, 'r') as f:
        data = json.load(f)

    best_params = data[dataset][str(K)][algorithm]

    return best_params['lr'], best_params['epochs']

def create_grid(lim):

    x = torch.linspace(-lim, lim, 100)
    y = torch.linspace(-lim, lim, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.column_stack([X.flatten(), Y.flatten()])

    return (X, Y, grid)

def plot_data(x, dataset, cluster_centers=None):
    plt.figure(figsize=(9, 6))

    plt.scatter(x[:, 0], x[:, 1], 
                c='dodgerblue',     # Set point color
                s=20,               # Set point size
                alpha=0.7,          # Add transparency to points
                edgecolor='k',      # Black edge for contrast
                linewidth=0.1)      # Thin edge

    if cluster_centers is not None:
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                    c='red',            
                    s=100,              
                    marker='X',         
                    edgecolor='k',      
                    linewidth=1.5)      

    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axis('off')
    plt.tight_layout()

    if not os.path.exists(f'results/{dataset}/'): os.makedirs(f'results/{dataset}/')
    plt.savefig(f"results/{dataset}/data.png", format='png', bbox_inches='tight')

def plot_density_and_samples(experiments, dataset, K):

    bins = 100
    lim = 4

    if dataset == 'halfmoons': lim = 2
    hist_range = [[-lim, lim], [-lim, lim]]

    X, Y, grid = create_grid(lim)

    fig, ax = plt.subplots(2, len(experiments), figsize=(len(experiments) * 5, 10))

    if len(experiments) == 1:
        ax = np.expand_dims(ax, 1)

    for i in range(len(experiments)):
        model, hyperparameters, algorithm, ll = experiments[i]

        samples = model.sample(100000)

        K, lr, epochs = hyperparameters
        if lr == 0: lr = 'none'

        with torch.no_grad():
            density = model(grid.to(model.device))

        ax[0, i].set_title(f'{algorithm} \n LL = {ll:.2f}', fontsize=20, pad=10, fontweight='bold')

        ax[0, i].contourf(X, Y, density.cpu().reshape(X.shape), levels=30, cmap=plt.cm.inferno)
        ax[0, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        ax[1, i].hist2d(samples.cpu()[:, 0], samples.cpu()[:, 1], range=hist_range, bins=bins, cmap=plt.cm.inferno)
        ax[1, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # ax[1, i].set_xlabel(f'lr = {lr}, epochs = {epochs}', fontsize=14, labelpad=10, fontweight='bold')

    # Set titles for the rows
    fig.text(0.09, 0.70, 'Log-Densities', va='center', rotation='vertical', fontsize=25, fontweight='bold')
    fig.text(0.09, 0.30, 'Samples', va='center', rotation='vertical', fontsize=25, fontweight='bold')
    # fig.text(0.05, 0.50, f'K = {K}', va='center', rotation='vertical', fontsize=35, fontweight='bold')
    fig.text(0.5, 0.08, f'K = {K}, lr = {lr}, epochs = {epochs}', ha='center', fontsize=18, fontweight='bold')


    plt.subplots_adjust(wspace=0.001, hspace=0.001)
    if not os.path.exists(f'results/{dataset}/'): os.makedirs(f'results/{dataset}/')
    plt.savefig(f"results/{dataset}/{dataset}_{K}.png", format='png', bbox_inches='tight')
    
def plot_logp(logps, dataset):

    data = []
    for logp, algorithm in logps:
        data.extend([(epoch, algorithm, value) for epoch, value in enumerate(logp)])

    df = pd.DataFrame(data, columns=['Epoch', 'Algorithm', 'NLL'])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 9))

    palette = sns.color_palette("pastel", n_colors=len(logps))

    sns.lineplot(data=df, x='Epoch', y='NLL', hue='Algorithm', style='Algorithm', 
                 markers=['s', 'd', 'o', '^'], markersize=8, markevery=4, 
                 dashes=False, linewidth=5, alpha=1, palette=palette)
    
    plt.legend(fontsize=20, loc='upper right', title='Algorithms', title_fontsize=25, frameon=False)

    plt.xlabel('Epochs', fontsize=25, labelpad=20)
    plt.ylabel('Negative Log-Likelihood', fontsize=25, labelpad=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True, linestyle='--', linewidth=2, alpha=0.5)
    plt.tight_layout()

    if not os.path.exists(f'results/{dataset}/'): os.makedirs(f'results/{dataset}/')
    plt.savefig(f"results/{dataset}/logp.png", format='png', dpi=300, bbox_inches='tight')