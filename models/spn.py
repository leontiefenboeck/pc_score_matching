import torch 
import torch.nn as nn
import torch.distributions as dist
import torch.autograd as autograd

class Sum(nn.Module):

    def __init__(self, nodes):
        super(Sum, self).__init__()
        self.n_components = len(nodes)

        self.leafs = nn.ModuleList(nodes)
        self.pi = nn.Parameter(torch.ones(self.n_components) / self.n_components)

    def fit(self, x, algorithm, epochs=1000, lr=0.001, n_slices=1):

        x = torch.tensor(x, dtype=torch.float32) 

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        x.requires_grad_()
        for _ in range(epochs):
            optimizer.zero_grad()

            if algorithm == "SGD":
                loss = -torch.mean(self(x))
            if algorithm == "SSM":
                loss = self.ssm_loss(x, n_slices)
            if algorithm == "SM":
                loss = self.sm_loss(x)

            self.loss_curve.append(loss.item())
            self.logp_curve.append(-torch.mean(self(x)).detach().item())
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
        
        print(f'{algorithm}-logp = {self.logp_curve[-1]}')

    def forward(self, x):
        logp = torch.zeros(x.size(0), self.n_components)

        for k in range(self.n_components):
            logp[:, k] = self.leafs[k](x)

        weighted_logp = logp + torch.log_softmax(self.pi, dim=-1)
        logp = torch.logsumexp(weighted_logp, dim=1)

        return logp
    
    def log_likelihood(self, x):
        return -torch.mean(self(x)).detach().item()
    
    def get_loss(self):
        return self.loss_curve
    
    def get_logp(self):
        return self.logp_curve
    
    def sample(self, num_samples):
        cat_dist = dist.Categorical(torch.softmax(self.pi, dim=-1))
        component_indices = cat_dist.sample((num_samples,))

        samples = torch.zeros(num_samples, self.n_features)
        for k in range(self.n_components): 
            mask = (component_indices == k)
            num_component_samples = mask.sum()
            
            if num_component_samples > 0:
                component_samples = self.leafs[k].sample(num_component_samples)
                samples[mask] = component_samples

        return samples

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
    
    def ssm_loss(self, x, n_slices):
        x = x.unsqueeze(0).expand(n_slices, *x.shape).contiguous().view(-1, *x.shape[1:])
        x.requires_grad_(True)

        v = torch.randn_like(x)
        v = v / torch.norm(v, dim=-1, keepdim=True)

        logp = self(x)
        score = autograd.grad(logp.sum(), x, create_graph=True)[0]

        gradv = score * v
        loss1 = 0.5 * (torch.sum(gradv, dim=-1) ** 2)

        grad2 = torch.autograd.grad(torch.sum(gradv), x, create_graph=True)[0]
        loss2 = torch.sum(v * grad2, dim=-1)

        return (loss1 + loss2).mean()

class Product(nn.Module):

    def __init__(self, nodes):
        super(Product, self).__init__()
        self.nodes = nn.ModuleList(nodes)

    def forward(self, x):
        product = torch.ones(x.size(0), dtype=x.dtype)
        
        for node in self.nodes:
            product *= node(x)
        
        return product

class Leaf(nn.Module):
    def __init__(self, n_features, mean_init = 0):
        super(Leaf, self).__init__()
        self.n_features = n_features

        if mean_init: 
            self.mean = nn.Parameter(torch.tensor(mean_init))
        else:
            self.mean = nn.Parameter(torch.rand(n_features))

        self.chol_var = nn.Parameter(torch.eye(n_features).repeat(1, 1))

    def forward(self, x):
        lower_triangular = torch.tril(self.chol_var)
        var_k = lower_triangular @ lower_triangular.t() + 1e-6 * torch.eye(self.n_features)
        logp = dist.MultivariateNormal(self.mean, var_k).log_prob(x)
        return logp 
    
    def sample(self, num_samples):
        lower_triangular = torch.tril(self.chol_var)
        var_k = lower_triangular @ lower_triangular.t() + 1e-6 * torch.eye(self.n_features)
        component_samples = dist.MultivariateNormal(self.mean.float(), var_k.float()).sample((num_samples,))
        return component_samples
    
