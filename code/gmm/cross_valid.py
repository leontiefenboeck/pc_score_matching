from gmm import GMM
import data
import utils
from sklearn.cluster import KMeans
import numpy as np
import torch

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = 'halfmoons'
algorithms = ['EM', 'GD', 'SM', 'SSM']
# algorithms = ['SSM']

# ----------------------------- parameters -------------------------------
num_samples = 10000

if dataset == 'halfmoons':
    Ks = [6, 10, 16]  
if dataset == 'spirals':
    Ks = [20, 40, 80]      
if dataset == 'board':
    Ks = [8, 60, 100]     

lr = [0.05, 0.025, 0.01]
epochs = [50, 100, 200]

seed = 42

# -------------------------------- training ----------------------------------
x = data.get_2d(dataset, num_samples * 2, seed)

np.random.seed(seed) 
np.random.shuffle(x)

x_test = torch.tensor(x[:num_samples], dtype=torch.float32).to(device)
x = torch.tensor(x[num_samples:], dtype=torch.float32).to(device)

experiments = []


for K in Ks:
    kmeans = KMeans(K, random_state=seed)
    kmeans.fit(x.detach().cpu())
    centers = kmeans.cluster_centers_
    
    for a in algorithms:

        best_params = None
        best_logp = -np.inf

        learning_rates = [0] if a == 'EM' else lr
        for l in learning_rates:
            for e in epochs:
                model = GMM(K, centers, device).to(device)
                utils.train(model, x, a, l, e, 1)

                logp = model.log_likelihood(x_test)
                print(f'[{a:{3}}] Test Log-Likelihood = {logp:.4f}')
                experiments.append((model, (K, l, e), a, logp))
                
                if logp > best_logp:
                    best_logp = logp
                    best_params = {'lr': l, 'epochs': e, 'll': best_logp}

        print(f'best log likelihood = {best_logp}')
        utils.save_best_params(dataset, a, K, best_params)

# utils.plot_density_and_samples(experiments, dataset)




