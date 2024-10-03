import numpy as np
import torch 
import torch.autograd as autograd
from sklearn.cluster import KMeans

from gmm import GMM
import data
import utils

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = 'spirals'
algorithms = ['EM', 'SGD', 'SM', 'SSM']
# algorithms = ['EM', 'SGD']

use_best_parameters = False  # parameters after cross validation (used in thesis)

# ----------------------------- parameters -------------------------------
num_samples = 10000

K = 10                      # number of components 
lr = 0.01                   # learning rate
epochs = 100                # number of training epochs

n_slices = 1                # how many random vectors for sliced score matching

seed = 42

# ------------------------------- functions ------------------------------
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

    logps = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = torch.tensor(x, dtype=torch.float32).to(device)   

    for i in range(epochs):

        logps.append(-model.log_likelihood(x))

        if algorithm == 'EM':
            model.EM_step(x)
        else: 
            if algorithm == "SGD":
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

# -------------------------------- training ------------------------------
x = data.get_2d(dataset, num_samples * 2, seed)

np.random.seed(seed) 
np.random.shuffle(x)

x_test = torch.tensor(x[:num_samples], dtype=torch.float32).to(device)
x = x[num_samples:]

experiments, logps = [], []

kmeans = KMeans(K, random_state=seed)
kmeans.fit(x)
centers = kmeans.cluster_centers_

for a in algorithms:
    model = GMM(K, centers, device).to(device)

    if use_best_parameters: lr, epochs = utils.load_best_params(dataset, a)

    train_lopgs = train(model, x, a, lr, epochs, n_slices)
    logps.append((train_lopgs, a))

    test_logp = model.log_likelihood(x_test)
    print(f'[{a:{3}}] Test Log-Likelihood = {test_logp:.4f}')
    experiments.append((model, (K, lr, epochs), a, test_logp))

# ------------------------------ visualization ---------------------------
utils.plot_data(x, dataset)
utils.plot_logp(logps, dataset)
utils.plot_density_and_samples(experiments, dataset)