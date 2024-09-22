from gmm import GMM
import data
import utils
from sklearn.cluster import KMeans
import numpy as np

dataset = 'board'
algorithm = 'EM'

# ----------------------------- parameters -------------------------------
num_samples = 10000

if dataset == 'moons':
    Ks = [8, 12, 14]  
if dataset == 'spirals':
    Ks = [30, 40, 50]      
if dataset == 'board':
    Ks = [60, 80, 100]     

lr = [0.1, 0.01]
epochs = [50, 100, 200]

seed = 42

# -------------------------------- training ----------------------------------
x = data.get_2d(dataset, num_samples * 2, seed)

np.random.seed(seed) 
np.random.shuffle(x)

x_test = x[:num_samples]
x = x[num_samples:]

models = []

best_params = None
best_logp = -np.inf

for K in Ks:
    kmeans = KMeans(K, random_state=seed)
    kmeans.fit(x)
    centers = kmeans.cluster_centers_
    for e in epochs:
        learning_rates = [None] if algorithm == 'EM' else lr
        for l in learning_rates:
            model = GMM(K, centers)
            model.fit(x, algorithm, e, l)
            models.append(model)

            logp = model.log_likelihood(x_test)
            print(logp)
            
            if logp > best_logp:
                best_logp = logp
                best_params = {'K': K, 'lr': l, 'epochs': e, 'll': best_logp}

print(f'best log likelihood = {best_logp}')
utils.save_best_params(dataset, algorithm, best_params)
utils.plot_density_and_samples(models, dataset)




