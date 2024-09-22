from gmm import GMM
import data
import utils
from sklearn.cluster import KMeans
import numpy as np

dataset = 'board'
algorithm = ['EM', 'SGD', 'SM', 'SSM']
# falgorithm = ['SM']

use_best_parameters = True  # parameters after cross validation (used in thesis)

# ----------------------------- parameters -------------------------------
num_samples = 10000

K = 100                      # number of components 
lr = 0.01                   # learning rate
epochs = 200                # number of training epochs

n_slices = 1                # how many random vectors for sliced score matching

seed = 42

# -------------------------------- training ------------------------------
x = data.get_2d(dataset, num_samples * 2, seed)

np.random.seed(seed) 
np.random.shuffle(x)

x_test = x[:num_samples]
x = x[num_samples:]

models = []

if use_best_parameters: 
    for a in algorithm:
        K, lr, epochs = utils.load_best_params(dataset, a)

        kmeans = KMeans(K, random_state=seed)
        kmeans.fit(x)
        centers = kmeans.cluster_centers_

        model = GMM(K, centers)
        model.fit(x, a, epochs, lr, n_slices)
        models.append(model)

else: 

    kmeans = KMeans(K, random_state=seed)
    kmeans.fit(x)
    centers = kmeans.cluster_centers_

    for a in algorithm:
        model = GMM(K, centers)
        model.fit(x, a, epochs, lr, n_slices)

        models.append(model)

# ------------------------------ eval ------------------------------------
for m in models:
    logp = m.log_likelihood(x_test)
    print(f'[{m.get_algorithm()}] Log Likelihood = {logp}')

# ------------------------------ visualization ---------------------------
utils.plot_data(x, dataset)
utils.plot_logp(models, dataset)
utils.plot_density_and_samples(models, dataset)
utils.plot_losses(models, dataset)