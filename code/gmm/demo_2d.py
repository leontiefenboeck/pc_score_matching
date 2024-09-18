from gmm import GMM
import data
import utils
from sklearn.cluster import KMeans

dataset = 'board'
algorithm = ['EM', 'SGD', 'SM', 'SSM']
# algorithm = ['EM']

# ----------------------------- parameters -------------------------------
num_samples = 5000

K = 50 # number of components 

lr = 0.01
epochs = 100

n_slices = 1 # how many random vectors for sliced score matching

seed = 42

# -------------------------------- data ----------------------------------
x = data.get_2d(dataset, num_samples)

kmeans = KMeans(K, random_state=seed)
kmeans.fit(x)
centers = kmeans.cluster_centers_

# -------------------------------- training ------------------------------
models = []

for a in algorithm:
    model = GMM(K, centers)
    model.fit(x, a, epochs, lr, n_slices)
    models.append(model)

# ------------------------------ visualization ---------------------------

bins = 200
range_lim = 4

utils.plot_data(x, dataset, bins, range_lim)
utils.plot_density(models, dataset)
utils.plot_samples(models, num_samples, dataset, bins, range_lim)
utils.plot_losses(models, dataset)