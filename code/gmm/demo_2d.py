from gmm import GMM
import data
import utils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans

# TODO better visualization for thesis

dataset_num = 0

datasets = ['moon', 'spirals', 'board']
dataset = datasets[dataset_num]
components = [8, 100, 100]
K = components[dataset_num]

algos = ['EM', 'SGD', 'SM', 'SSM']
algos = ['SSM']

seed = 42
n_slices = 1 # for sliced score matching

epochs = 500
lr = 0.01

x = data.get_2d(dataset)

kmeans = KMeans(K, random_state=seed)
kmeans.fit(x)
centers = kmeans.cluster_centers_

# plt.scatter(x[:, 0], x[:, 1], c='blue', cmap=plt.cm.RdYlBu)
# plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100, label='Cluster Centers')
# plt.show()

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
