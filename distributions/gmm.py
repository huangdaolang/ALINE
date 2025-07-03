import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, Normal, constraints

class MixtureOfGaussians(Distribution):
    arg_constraints = {
        'means': constraints.real_vector,
        'stds': constraints.positive,
        'weights': constraints.simplex
    }

    def __init__(self, means, stds, weights, validate_args=None):
        # Ensure means, stds, and weights have the same shape
        assert means.shape == stds.shape
        # assert means.shape[:-1] == weights.shape

        self.means = means      # [B, D, N]
        self.stds = stds        # [B, D, N]
        self.weights = weights    # [B, N]

        # Create a categorical distribution for the mixing coefficients
        self.mixing_distribution = Categorical(weights)
        self.component_distribution = Normal(means, stds)

        super(MixtureOfGaussians, self).__init__(batch_shape=means.shape[:-1], validate_args=validate_args)

    def log_prob(self, x):
        """ Log prob

        Args:
            x [B, K, D]: sample values

        Returns:
            [B, K]
        """
        x = x.permute(1, 0, 2)  # [K, B, D]
        # log probability of `x` under each Gaussian component
        log_probs = self.component_distribution.log_prob(x.unsqueeze(-1))                       # [K, B, D, N]
        log_probs = log_probs.sum(-2)                                                           # [K, B, N]
        log_probs = log_probs.permute(1, 0, 2)                                                  # [B, K, N]
        # Weighted sum of log_probs with the mixing coefficients
        log_probs = torch.logsumexp(log_probs + torch.log(self.weights.unsqueeze(-2)), dim=-1)  # [B, K]
        return log_probs

    def sample(self, sample_shape=torch.Size()):
        # Sample component indices according to the mixing probabilities
        component_indices = self.mixing_distribution.sample(sample_shape)                       # [..., B]
        # Sample from the selected Gaussian component
        samples = self.component_distribution.sample(sample_shape)                              # [..., B, D, N]
        return torch.gather(samples, -1, component_indices.unsqueeze(-1).unsqueeze(-1)).squeeze(-1) # [..., B, D]

    def rsample(self, sample_shape=torch.Size()):
        # rsample is needed for reparameterization trick (e.g., for variational inference)
        component_indices = self.mixing_distribution.sample(sample_shape)
        base_samples = self.component_distribution.rsample(sample_shape)
        return torch.gather(base_samples, -1, component_indices.unsqueeze(-1)).squeeze(-1)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MixtureOfGaussians, _instance)
        batch_shape = torch.Size(batch_shape)
        new.means = self.means.expand(batch_shape + self.event_shape)
        new.stds = self.stds.expand(batch_shape + self.event_shape)
        new.weights = self.weights.expand(batch_shape + self.event_shape[:-1])
        new.mixing_distribution = Categorical(new.weights)
        new.component_distribution = Normal(new.means, new.stds)
        super(MixtureOfGaussians, new).__init__(batch_shape, validate_args=False)
        return new

    @property
    def mean(self):
        # Return the mean of the mixture distribution
        return torch.sum(self.weights * self.means, dim=-1)

    @property
    def variance(self):
        # Return the variance of the mixture distribution
        mean_sq = torch.sum(self.weights * (self.means ** 2 + self.stds ** 2), dim=-1)
        return mean_sq - self.mean ** 2