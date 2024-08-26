import torch 
import torch.nn as nn
import torch.distributions as dist
import torch.autograd as autograd

from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

K = 10
algos = ["SM", "SGD"]

class GMM(nn.Module):

    def __init__(self, x, n_components, n_features, algorithm="SGD", means_init = 0):
        super(GMM, self).__init__()
        self.n_components = n_components
        self.n_features = n_features

        if means_init: 
            self.means = nn.Parameter(torch.tensor(means_init))
        else:
            self.means = self.init_mean(x, n_components)

        self.chol_var = nn.Parameter(torch.eye(n_features).repeat(n_components, 1, 1))
        self.pi = nn.Parameter(torch.ones(n_components) / n_components)

        if algorithm == "EM": self.EM(x)
        else: self.train_torch(x, algorithm)

    def init_mean(self, x, n_components):
        kmeans = KMeans(n_components)
        kmeans.fit(x.detach())
        return nn.Parameter(torch.tensor(kmeans.cluster_centers_))
    
    def forward(self, x):
        log_likelihoods = torch.zeros(x.size(0), self.n_components)
        # self.chol_var.data = torch.clamp(self.chol_var.data, min=1e-6)

        for k in range(self.n_components):
            lower_triangular = torch.tril(self.chol_var[k])
            var_k = lower_triangular @ lower_triangular.t() + 1e-6 * torch.eye(self.n_features)
            log_probs = dist.MultivariateNormal(self.means[k], var_k).log_prob(x)
            log_likelihoods[:, k] = log_probs


        weighted_log_likelihoods = log_likelihoods + torch.log_softmax(self.pi, dim=-1)
        log_likelihood = torch.logsumexp(weighted_log_likelihoods, dim=1)

        return log_likelihood
    
    def sample(self, num_samples):
        cat_dist = dist.Categorical(torch.softmax(self.pi, dim=-1))
        component_indices = cat_dist.sample((num_samples,))

        samples = torch.zeros(num_samples, self.n_features)
        for k in range(self.n_components): 
            mask = (component_indices == k)
            num_component_samples = mask.sum()
            
            if num_component_samples > 0:
                var_k = self.chol_var[k] @ self.chol_var[k].t() + 1e-6 * torch.eye(self.n_features)
                component_samples = dist.MultivariateNormal(self.means[k].float(), var_k.float()).sample((num_component_samples,))
                samples[mask] = component_samples

        return samples
    
    def train_torch(self, x, algorithm, epochs=1000, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        losses = []
        x.requires_grad_()
        for _ in range(epochs):
            optimizer.zero_grad()

            if algorithm == "SGD":
                loss = -torch.mean(self(x))
            if algorithm == "SSM":
                loss = self.ssm_loss(x)
            if algorithm == "SM":
                loss = self.sm_loss(x)

            losses.append(loss.item())
            loss.backward()

            # nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
            optimizer.step()
        
        print(-torch.mean(self(x)))

    def sm_loss(self, x):
        x.requires_grad_()

        logp = self(x).sum()
        score = autograd.grad(logp, x, create_graph=True)[0]
        loss1 = 0.5 * torch.norm(score, dim=-1) ** 2
        
        trace = torch.ones_like(loss1)
        for i in range(x.size(1)):
            trace += autograd.grad(score[:, i].sum(), x, create_graph=True)[0][:, i]
        
        loss = torch.mean(loss1 + trace)
        return loss
    
    def ssm_loss(self, x):
        x.requires_grad_()
        v = torch.randn_like(x)

        logp = self(x)
        score = autograd.grad(logp.sum(), x, create_graph=True)[0]

        gradv = score * v
        loss1 = 0.5 * (torch.sum(gradv, dim=-1) ** 2)

        grad2 = torch.autograd.grad(torch.sum(gradv), x, create_graph=True)[0]
        loss2 = torch.sum(v * grad2, dim=-1)

        return (loss1 + loss2).mean()

    def EM(self, x, epochs=100, tol=1e-6):
        x = x.detach()  
        
        for _ in range(epochs):
            log_likelihoods = torch.zeros(x.size(0), self.n_components)

            for k in range(self.n_components):
                var_k = self.chol_var[k] @ self.chol_var[k].t() + 1e-6 * torch.eye(self.n_features)
                mvn = dist.MultivariateNormal(self.means[k], var_k)
                log_likelihoods[:, k] = mvn.log_prob(x)

            weighted_log_likelihoods = log_likelihoods + torch.log(self.pi + 1e-10)
            log_likelihoods = torch.logsumexp(weighted_log_likelihoods, dim=1, keepdim=True)
            resp = torch.exp(weighted_log_likelihoods - log_likelihoods)

            resum = torch.sum(resp, dim=0)

            self.means.data = resp.T @ x / resum.unsqueeze(1)
            self.pi.data = resum / x.size(0)

            for k in range(self.n_components):
                diff = x - self.means[k]
                var_k = (resp[:, k] * diff.T @ diff) / resum[k]
                self.chol_var.data[k] = torch.linalg.cholesky(var_k + 1e-6 * torch.eye(self.n_features))

# Sample data
data, target = make_moons(n_samples=500, noise=0.1, random_state=42)

data = torch.tensor(data, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.long)

fig_data, ax_data = plt.subplots(1, 2, figsize=(15, 5))
fig_data.suptitle("Halfmoon Dataset and Kmeans centroids")
ax_data[0].scatter(data[:, 0], data[:, 1], c=target, cmap=plt.cm.RdYlBu)

x = torch.linspace(-1.5, 2.5, 100)
y = torch.linspace(-1, 1.5, 100)
X, Y = torch.meshgrid(x, y, indexing='ij')
points = torch.column_stack([X.flatten(), Y.flatten()])

# plot clusters
kmeans = KMeans(K)
kmeans.fit(data)
centers = kmeans.cluster_centers_
ax_data[1].scatter(data[:, 0], data[:, 1], c='blue', label='Data Points')
ax_data[1].scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100, label='Cluster Centers')

figs_models = []

for a in algos:
    model = GMM(data, K, 2, a)

    fig_model, ax_model = plt.subplots(1, 2, figsize=(15, 5))
    fig_model.suptitle(a)

    samples = model.sample(500)
    ax_model[0].set_title("Samples")
    ax_model[0].scatter(samples[:, 0], samples[:, 1])

    with torch.no_grad():
        density = model(points)

    ax_model[1].set_title("Density")
    ax_model[1].contour(X.detach(), Y.detach(), density.reshape(X.shape).detach().numpy(), levels=200)

    figs_models.append(fig_model)

pdf = PdfPages('figures.pdf')
pdf.savefig(fig_data)
for f in figs_models:
    pdf.savefig(f)
pdf.close()
