from models.gmm import GMM
import data
import utils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans

# TODO better visualization for thesis

datasets = ['moon', 'spirals', 'board']
components = [8, 100, 100]

algos = ['EM', 'SGD', 'SM', 'SSM']

seed = 42
n_slices = 1 # for sliced score matching

epochs = 100
lr = 0.01

for d, K in zip(datasets, components):

    x = data.get_2d(d)

    kmeans = KMeans(K, random_state=seed)
    kmeans.fit(x)
    centers = kmeans.cluster_centers_

    plt.scatter(x[:, 0], x[:, 1], c='blue', cmap=plt.cm.RdYlBu)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100, label='Cluster Centers')
    plt.show()

    models = []

    for a in algos:
        model = GMM(K, centers)
        model.fit(x, a, epochs, lr, n_slices)
        models.append(model)

    grid = utils.create_grid()

    figs_models = []
    for m, a in zip(models, algos):
        figs_models.append(utils.plot_eval(m, grid, a))

    plt.show()
