import torch 
import torch.nn as nn
import torch.distributions as dist
import torch.autograd as autograd

class GMM(nn.Module):

    def __init__(self, n_components, means_init=0, n_features=2):
        super(GMM, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.loss_curve = []
        self.logp_curve = []

        if means_init.any(): 
            self.means = nn.Parameter(torch.tensor(means_init))
        else:
            self.means = nn.Parameter(torch.rand(means_init))

        self.chol_var = nn.Parameter(torch.eye(n_features).repeat(n_components, 1, 1))
        self.pi = nn.Parameter(torch.ones(n_components) / n_components)

    def fit(self, x, algorithm, epochs, lr, n_slices):

        x = torch.tensor(x, dtype=torch.float32) 

        if algorithm == "EM": 
            self.EM(x, epochs)
        else: 
            self.train_torch(x, algorithm, epochs, lr, n_slices)

    def forward(self, x):
        log_likelihoods = torch.zeros(x.size(0), self.n_components)

        for k in range(self.n_components):
            lower_triangular = torch.tril(self.chol_var[k])
            var_k = lower_triangular @ lower_triangular.t() + 1e-6 * torch.eye(self.n_features)
            log_probs = dist.MultivariateNormal(self.means[k], var_k).log_prob(x)
            log_likelihoods[:, k] = log_probs

        weighted_log_likelihoods = log_likelihoods + torch.log_softmax(self.pi, dim=-1)
        log_likelihood = torch.logsumexp(weighted_log_likelihoods, dim=1)

        return log_likelihood
    
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
                var_k = self.chol_var[k] @ self.chol_var[k].t() + 1e-6 * torch.eye(self.n_features)
                component_samples = dist.MultivariateNormal(self.means[k].float(), var_k.float()).sample((num_component_samples,))
                samples[mask] = component_samples

        return samples
    
    def train_torch(self, x, algorithm, epochs, lr, n_slices=1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        x.requires_grad_()
        for i in range(epochs):
            
            if algorithm == "SGD":
                loss = -torch.mean(self(x))
            if algorithm == "SSM":
                loss = self.ssm_loss(x, n_slices)
            if algorithm == "SM":
                loss = self.sm_loss(x)

            self.loss_curve.append(loss.item())
            self.logp_curve.append(-torch.mean(self(x)).detach().item())
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % (epochs / 10) == 0: print(f'[{i}]       {algorithm}-logp = {self.logp_curve[-1]}')

        print(f'[{epochs}]       {algorithm}-logp = {self.logp_curve[-1]}') 

    def sm_loss(self, x):

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

        v = torch.randn_like(x)
        # v = v / torch.norm(v, dim=-1, keepdim=True)

        logp = self(x)

        score = autograd.grad(logp.sum(), x, create_graph=True)[0]
        loss1 = 0.5 * (torch.norm(score, dim=-1) ** 2)

        grad2 = torch.autograd.grad(torch.sum(score * v), x, create_graph=True)[0]
        loss2 = torch.sum(v * grad2, dim=-1)

        return (loss1 + loss2).mean()

    def EM(self, x, epochs=100):
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

        print(f'EM-logp = {-torch.mean(self(x)).detach().item()}')