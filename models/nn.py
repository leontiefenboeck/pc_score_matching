import torch 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

sigma_1 = 0.001
sigma_L = 1
L = 100

algo = "SM"

# set random seed 
# does score work better than energy 

def get_sigmas(sigma_1, sigma_L, L):
    return torch.tensor(np.exp(np.linspace(np.log(sigma_1),np.log(sigma_L), L)))

def sm_loss(net, x):
    # tr(nabla2x logp(x)) + 0.5 * norm( nablax logp(x)) ** 2

    # d logp / dx = -df(x) / dx = s
    # tr(ds / dx) + 0.5 * norm(s) ** 2

    x.requires_grad_()
 
    # energy = -net(x).sum()
    score = net(x)
    loss1 = 0.5 * torch.norm(score, dim=-1) ** 2
    
    trace = torch.zeros_like(loss1)
    for i in range(x.size(1)):
        trace += autograd.grad(score[:, i].sum(), x, create_graph=True)[0][:, i]
    
    loss = torch.mean(loss1 + trace)
    return loss

def ssm_loss(net, x):
    # vT * nabla2x logp(x) * v + 0.5 * (vT * nablax logp(x)) ** 2 
    v = torch.randn_like(x)

    # more v 

    x.requires_grad_()

    energy = -net(x)
    score = autograd.grad(energy.sum(), x, create_graph=True)[0]

    gradv = score * v
    loss1 = 0.5 * (torch.sum(gradv, dim=-1) ** 2)

    grad2 = autograd.grad(gradv.sum(), x, create_graph=True)[0]
    loss2 = torch.sum(v * grad2, dim=-1)

    loss = torch.mean(loss1 + loss2)
    return loss

def dsm_loss(net, x):

    all_sigmas = get_sigmas(sigma_1, sigma_L, L)

    sigmas = all_sigmas[torch.randint(0, len(all_sigmas), (x.size(0), 1))].to(x.dtype)
    noise = torch.randn_like(x) * sigmas
    x_hat = (x + noise).requires_grad_()

    energy = -net(torch.cat((x_hat, sigmas), dim=1))
    net_scores = torch.autograd.grad(energy.sum(), x_hat, create_graph=True)[0]

    scores = noise / (sigmas**2)
    sigmas = sigmas.reshape(-1)

    loss = (sigmas**2) * (torch.norm(scores + net_scores, dim=1)**2)

    return loss.mean()

def train(net, x, iterations = 2000, lr = 0.01):

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []

    for _ in range(iterations):
        optimizer.zero_grad()

        if algo == "DSM":
            loss = dsm_loss(net, x)
        if algo == "SSM":
            loss = ssm_loss(net, x)
        if algo == "SM":
            loss = sm_loss(net, x)

        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()

    return losses 

def evaluate_energy(net):

    x = torch.linspace(-1.5, 2.5, 100)
    y = torch.linspace(-1, 1.5, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.column_stack([X.flatten(), Y.flatten()])
    points = points.requires_grad_()

    if algo == "DSM":
        energy = net(torch.cat((points,  sigma_1 * torch.ones_like(X.flatten()).reshape(-1, 1)), dim=1))
    else :
        energy = net(points)

    scores_x = energy[:, 0].reshape(X.shape).detach().numpy()
    scores_y = energy[:, 1].reshape(Y.shape).detach().numpy()
    

    step = 2  # Change step to control arrow density
    X_downsampled = X[::step, ::step].detach().numpy()
    Y_downsampled = Y[::step, ::step].detach().numpy()
    scores_x_downsampled = scores_x[::step, ::step]
    scores_y_downsampled = scores_y[::step, ::step]

    plt.quiver(X_downsampled, Y_downsampled, scores_x_downsampled, scores_y_downsampled,
                        np.hypot(scores_x_downsampled, scores_y_downsampled))
    plt.show()

data, target = make_moons(n_samples=1000, noise=0.1, random_state=42)

data = torch.tensor(data, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.long)

hidden_size = 64
if algo == "DSM":
    Net = nn.Sequential(nn.Linear(3, hidden_size), nn.ELU(), nn.Linear(hidden_size, hidden_size), nn.ELU(), nn.Linear(hidden_size, 1))
else: 
    Net = nn.Sequential(nn.Linear(2, hidden_size), nn.SELU(), nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Linear(hidden_size, 2))

train_loss = train(Net, data)
print(f'Train Loss = {train_loss[-1]}')
plt.plot(train_loss)
plt.show()
evaluate_energy(Net)

