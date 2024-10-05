import torch 
import torch.nn as nn
import torch.distributions as dist

class GMM(nn.Module):

    def __init__(self, device, K, means_init, random_weights = False, n_features=2):
        super(GMM, self).__init__()
        self.device = device
        self.K = K
        self.n_features = n_features

        if random_weights:
            pi_init = torch.rand(K, dtype=torch.float32)  
            self.pi = nn.Parameter(pi_init / pi_init.sum())
        else:
            self.pi = nn.Parameter(torch.ones(K) / K)

        self.means = nn.Parameter(torch.tensor(means_init))
        self.chol_var = nn.Parameter(torch.eye(n_features).repeat(K, 1, 1))

    def forward(self, x):
        log_likelihoods = torch.zeros(x.size(0), self.K, device=x.device)  # Ensure it's on the same device as x

        for k in range(self.K):
            lower_triangular = torch.tril(self.chol_var[k])
            var_k = lower_triangular @ lower_triangular.t() + 1e-6 * torch.eye(self.n_features, device=x.device)  # On the same device
            log_probs = dist.MultivariateNormal(self.means[k], var_k).log_prob(x)
            log_likelihoods[:, k] = log_probs

        weighted_log_likelihoods = log_likelihoods + torch.log_softmax(self.pi, dim=-1)
        log_likelihood = torch.logsumexp(weighted_log_likelihoods, dim=1)

        return log_likelihood

    def log_likelihood(self, x):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            return torch.mean(self(x)).item()
        
    def sample(self, num_samples):
        cat_dist = dist.Categorical(torch.softmax(self.pi, dim=-1))
        component_indices = cat_dist.sample((num_samples,))

        samples = torch.zeros(num_samples, self.n_features, device=self.means.device)  # Ensure it's on the same device
        for k in range(self.K): 
            mask = (component_indices == k)
            num_component_samples = mask.sum()
            
            if num_component_samples > 0:
                var_k = self.chol_var[k] @ self.chol_var[k].t() + 1e-6 * torch.eye(self.n_features, device=self.means.device)  # On the same device
                component_samples = dist.MultivariateNormal(self.means[k].float(), var_k.float()).sample((num_component_samples,))
                samples[mask] = component_samples

        return samples

    def EM_step(self, x):
        log_likelihoods = torch.zeros(x.size(0), self.K, device=x.device)  

        for k in range(self.K):
            var_k = self.chol_var[k] @ self.chol_var[k].t() + 1e-6 * torch.eye(self.n_features, device=x.device) 
            mvn = dist.MultivariateNormal(self.means[k], var_k)
            log_likelihoods[:, k] = mvn.log_prob(x)

        weighted_log_likelihoods = log_likelihoods + torch.log(self.pi + 1e-10)
        log_likelihoods = torch.logsumexp(weighted_log_likelihoods, dim=1, keepdim=True)
        resp = torch.exp(weighted_log_likelihoods - log_likelihoods)

        resum = torch.sum(resp, dim=0)

        self.means.data = (resp.T @ x) / resum.unsqueeze(1)
        self.pi.data = resum / x.size(0)

        for k in range(self.K):
            diff = x - self.means[k]
            var_k = (resp[:, k] * diff.T @ diff) / resum[k]
            self.chol_var.data[k] = torch.linalg.cholesky(var_k + 1e-6 * torch.eye(self.n_features, device=x.device))  