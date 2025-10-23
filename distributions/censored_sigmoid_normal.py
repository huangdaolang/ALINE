import torch
from torch.distributions import Distribution, Normal, TransformedDistribution, SigmoidTransform, constraints
from torch.distributions.utils import broadcast_all

def is_bad(x):
    return torch.isnan(x).any() or torch.isinf(x).any()

class CensoredSigmoidNormal(Distribution):
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "lower_lim": constraints.real,
        "upper_lim": constraints.real,
    }

    has_rsample = True  # Enables reparameterized sampling

    def __init__(self, loc, scale, lower_lim, upper_lim, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self.upper_lim, self.lower_lim = broadcast_all(upper_lim, lower_lim)

        self.normal = Normal(self.loc, self.scale)
        self.transform = SigmoidTransform()
        self.base_dist = TransformedDistribution(self.normal, [self.transform])

        batch_shape = self.base_dist.batch_shape
        event_shape = self.base_dist.event_shape

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def z(self, value):
        return (self.transform.inv(value) - self.loc) / self.scale

    @property
    def support(self):
        return constraints.interval(self.lower_lim.min(), self.upper_lim.max())

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        x = self.base_dist.sample(sample_shape)
        return torch.clamp(x, min=self.lower_lim, max=self.upper_lim)

    def rsample(self, sample_shape=torch.Size()):
        x = self.base_dist.rsample(sample_shape)
        return torch.clamp(x, min=self.lower_lim, max=self.upper_lim)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        # Broadcast all inputs to same shape
        value, upper_lim, lower_lim = torch.broadcast_tensors(value, self.upper_lim, self.lower_lim)

        log_prob = self.base_dist.log_prob(value)

        # Compute log_cdf values at limits
        upper_cdf = 1. - self.base_dist.cdf(upper_lim)
        lower_cdf = self.base_dist.cdf(lower_lim)

        crit = 2 * torch.finfo(value.dtype).tiny

        mask_upper = upper_cdf < crit
        mask_lower = lower_cdf < crit

        z_upper = self.z(upper_lim)
        z_lower = self.z(lower_lim)

        asymptotic_upper = self.base_dist.log_prob(upper_lim) - (crit + z_upper.abs()).log()
        asymptotic_lower = self.base_dist.log_prob(lower_lim) - (crit + z_lower.abs()).log()

        upper_logcdf = upper_cdf.clone().log()
        lower_logcdf = lower_cdf.clone().log()

        upper_logcdf[mask_upper] = asymptotic_upper[mask_upper]
        lower_logcdf[mask_lower] = asymptotic_lower[mask_lower]

        # Fill in special values based on value mask
        log_prob = torch.where(value == upper_lim, upper_logcdf, log_prob)
        log_prob = torch.where(value == lower_lim, lower_logcdf, log_prob)
        log_prob = torch.where(value > upper_lim, float('-inf'), log_prob)
        log_prob = torch.where(value < lower_lim, float('-inf'), log_prob)

        if is_bad(log_prob):
            raise ArithmeticError("NaN in log_prob")

        return log_prob

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)

        cdf_val = self.base_dist.cdf(value)
        cdf_val[value >= self.upper_lim] = 1.
        cdf_val[value < self.lower_lim] = 0.
        return cdf_val

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(CensoredSigmoidNormal, _instance)

        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.upper_lim = self.upper_lim.expand(batch_shape)
        new.lower_lim = self.lower_lim.expand(batch_shape)

        new.normal = Normal(new.loc, new.scale)
        new.transform = self.transform
        new.base_dist = TransformedDistribution(new.normal, [new.transform])

        super(CensoredSigmoidNormal, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
