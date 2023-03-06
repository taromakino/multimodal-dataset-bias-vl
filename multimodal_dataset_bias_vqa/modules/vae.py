import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from modules.stats import diag_gaussian_log_prob


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        module_list = []
        last_in_dim = input_dim
        for hidden_dim in hidden_dims:
            module_list.append(nn.Linear(last_in_dim, hidden_dim))
            module_list.append(nn.GELU())
            last_in_dim = hidden_dim
        module_list.append(nn.Linear(last_in_dim, output_dim))
        self.module_list = nn.Sequential(*module_list)

    def forward(self, *args):
        return torch.squeeze(self.module_list(torch.hstack(args)))


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.mu_net = MLP(input_dim, hidden_dims, output_dim)
        self.logvar_net = MLP(input_dim, hidden_dims, output_dim)

    def forward(self, *args):
        return self.mu_net(*args), F.softplus(self.logvar_net(*args))


class Vae(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, output_dim, n_components, n_samples):
        super().__init__()
        self.n_samples = n_samples
        self.q_z_xy_net = GaussianMLP(2 * input_dim + output_dim, hidden_dims, latent_dim)
        self.q_z_x_net = GaussianMLP(2 * input_dim, hidden_dims, latent_dim)
        self.p_y_xz_net = MLP(2 * input_dim + latent_dim, hidden_dims, output_dim)
        self.logits_c = nn.Parameter(torch.ones(n_components))
        self.mu_z_c = nn.Parameter(torch.zeros(n_components, latent_dim))
        self.logvar_z_c = nn.Parameter(torch.zeros(n_components, latent_dim))
        nn.init.xavier_normal_(self.mu_z_c)
        nn.init.xavier_normal_(self.logvar_z_c)


    def sample_z(self, mu, var):
        sd = var.sqrt()
        eps = torch.randn(self.n_samples, *sd.shape).to(sd.get_device())
        return mu + eps * sd


    def forward(self, x, y):
        batch_size = len(x) # For assertions
        # z ~ q(z|x,y)
        mu_z_xy, var_z_xy = self.q_z_xy_net(x, y)
        z = self.sample_z(mu_z_xy, var_z_xy)
        log_q_z_xy = diag_gaussian_log_prob(z, mu_z_xy, var_z_xy).view(-1)
        assert log_q_z_xy.shape == (self.n_samples * batch_size,)
        # E_q(z|x,y)[log p(y|x,z)]
        x = torch.repeat_interleave(x[None], repeats=self.n_samples, dim=0)
        y = torch.repeat_interleave(y[None], repeats=self.n_samples, dim=0)
        x, y, z = x.view(-1, x.shape[-1]), y.view(-1, y.shape[-1]), z.view(-1, z.shape[-1])
        logits_y_xz = self.p_y_xz_net(x, z)
        log_p_y_xz = F.binary_cross_entropy_with_logits(logits_y_xz, y, reduction="none").sum(dim=1)
        assert log_p_y_xz.shape == (self.n_samples * batch_size,)
        # KL(q(z|x,y) || p(z))
        dist_c = D.Categorical(logits=self.logits_c)
        var_z_c = F.softplus(self.logvar_z_c)
        dist_z_c = D.Independent(D.Normal(self.mu_z_c, var_z_c.sqrt()), 1)
        dist_z = D.MixtureSameFamily(dist_c, dist_z_c)
        log_p_z = dist_z.log_prob(z)
        assert log_p_z.shape == (self.n_samples * batch_size,)
        kl = (log_q_z_xy - log_p_z).mean()
        elbo = log_p_y_xz.mean() - kl
        logits_y_xz = logits_y_xz.view((self.n_samples, batch_size, -1))
        return {
            "loss": -elbo,
            "kl": kl,
            "logits_y_xz": logits_y_xz[0]
        }