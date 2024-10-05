import numpy as np
import torch 
from sklearn.cluster import KMeans

from gmm import GMM
import data
import utils

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = 'spirals'
algorithms = ['EM', 'GD', 'SM', 'SSM']

use_best_parameters = False             # parameters after cross validation (used in thesis)
K_index = 0                             # use 0 for small, 1 for moderate and 2 for large K (when using best parameters)

random_init = False                      # initialize means and weights randomly
# ----------------------------- parameters -------------------------------
num_samples = 10000

K = 30                      # number of components 
lr = 0.01                   # learning rate
epochs = 200                # number of training epochs

n_slices = 1                # how many random vectors for sliced score matching

seed = 42
# torch.manual_seed(seed)
np.random.seed(seed) 

if dataset == 'halfmoons': Ks = [8, 12, 24]    
if dataset == 'spirals': Ks = [20, 40, 80]      
if dataset == 'board': Ks = [8, 60, 100]  

# -------------------------------- training ------------------------------
x = data.get_2d(dataset, num_samples * 2, seed)
np.random.shuffle(x)

if use_best_parameters: K = Ks[K_index]

if random_init:
    random_indices = np.random.choice(x.shape[0], K, replace=False)
    centers = x[random_indices]
else:
    kmeans = KMeans(K, random_state=seed)
    kmeans.fit(x)
    centers = kmeans.cluster_centers_

experiments, logps = [], []

x_test = torch.tensor(x[:num_samples], dtype=torch.float32).to(device)
x = torch.tensor(x[num_samples:], dtype=torch.float32).to(device)

for a in algorithms:

    model = GMM(device, K, centers, random_init).to(device)

    if use_best_parameters: lr, epochs = utils.load_best_params(dataset, a, K)

    train_lopgs = utils.train(model, x, a, lr, epochs, n_slices)
    logps.append((train_lopgs, a))

    test_logp = model.log_likelihood(x_test)
    print(f'[{a:{3}}] Test Log-Likelihood = {test_logp:.4f}')
    experiments.append((model, (K, lr, epochs), a, test_logp))

# ------------------------------ visualization ---------------------------
utils.plot_data(x.detach().cpu(), dataset, centers)
utils.plot_density_and_samples(experiments, dataset, K)
utils.plot_logp(logps, dataset, K)