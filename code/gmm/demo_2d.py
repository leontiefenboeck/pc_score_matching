from gmm import GMM
import data
import utils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans

# TODO better visualization for thesis

dataset = 'spirals'
algorithm = ['EM', 'SGD', 'SM', 'SSM']
# algorithm = ['SM']

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

fig, ax = plt.subplots(1, len(models) + 1, figsize=(25, 5))

range_lim = 4
bins = 200

rang = [[-range_lim, range_lim], [-range_lim, range_lim]]
ax[0].hist2d(x[:, 0], x[:, 1], range=rang, bins=bins, cmap=plt.cm.viridis)
ax[0].set_title(f'ground truth')

for i in range(len(models)):
    samples = models[i].sample(num_samples)
    ax[i + 1].hist2d(samples[:, 0], samples[:, 1], range=rang, bins=bins, cmap=plt.cm.viridis)
    ax[i + 1].set_title(f'samples - {algorithm[i]}')

for a in ax:
    a.axis('off')

plt.subplots_adjust(wspace=0.1) 
plt.savefig(f"results/{dataset}/samples.png", format='png')

for i in range(len(models)):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'{algorithm[i]}')
    utils.plot_loss(models[i], ax[0])
    utils.plot_logp(models[i], ax[1])
    plt.savefig(f"results/{dataset}/{algorithm[i]}_losses.png", format='png')