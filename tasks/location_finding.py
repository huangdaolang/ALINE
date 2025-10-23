import torch
import torch.nn as nn
import torch.distributions as dist
from attrdictionary import AttrDict
from tasks.base_task import Task


class HiddenLocation(Task):
    """Simulate Location Finding Experiment"""

    def __init__(
            self,
            name: str = "Location",
            dim_x: int = 2,  # dimension of location
            dim_y: int = 1,  # dimension of outcome
            embedding_type="theta",  # mode of the experiment
            n_target_theta: int = 2,  # number of theta, treat each coordinate as one theta
            n_context_init: int = 1,  # number of initial context points
            n_query_init: int = 200,  # number of initial query points
            K: int = 1,  # number of source points
            theta_loc=None,  # prior on theta
            theta_cov=None,  # prior on theta
            theta_dist="uniform",  # prior distribution type
            design_scale=None,  # scale of the design space
            outcome_scale=10,  # scale of the experiment outcomes
            noise_scale=0.5,
            base_signal: float = 0.1,  # param of signal
            max_signal: float = 1e-4,  # param of signal
            **kwargs,

    ) -> None:
        super(HiddenLocation, self).__init__(dim_x=dim_x, dim_y=dim_y, n_target_theta=n_target_theta, K=K)

        # prior of theta
        self.theta_dist = theta_dist

        low = torch.zeros((K, self.dim_x))
        high = torch.ones((K, self.dim_x))

        if theta_dist == "normal":
            self.theta_loc = theta_loc if theta_loc is not None else torch.zeros((K, dim_x))
            self.theta_cov = theta_cov if theta_cov is not None else torch.eye(dim_x)
            if dim_x == 1:
                self.theta_prior = dist.Normal(self.theta_loc, self.theta_cov)
            else:
                self.theta_prior = dist.MultivariateNormal(
                    self.theta_loc, self.theta_cov
                )

            low = torch.ones((K, self.dim_x)) * -4
            high = torch.ones((K, self.dim_x)) * 4

        elif theta_dist == "uniform":
            self.theta_loc = theta_loc if theta_loc is not None else torch.zeros((K, dim_x))  # low
            self.theta_cov = theta_cov if theta_cov is not None else torch.ones((K, dim_x))  # scale: high - low
            self.theta_prior = dist.Uniform(
                self.theta_loc, self.theta_cov
            )
        else:
            raise ValueError(f"Prior distribution type {theta_dist} is not supported!")

        # sampler of the data
        self.data_sampler = dist.Uniform(low, high)

        # scale of design space
        self.design_scale = (
            design_scale if design_scale is not None else torch.max(self.theta_cov)
        )
        # signal params
        noise_scale = noise_scale * torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("noise_scale", noise_scale)
        self.base_signal = base_signal
        self.max_signal = max_signal

        self.n_target_theta = n_target_theta
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init

        self.K = K
        assert self.n_target_theta == self.K * self.dim_x, "n_theta must be equal to K * dim_x"

        self.conditional = dist.Normal(0, self.noise_scale)

    @torch.no_grad()
    def sample_theta(self, batch_size):
        """ Sample latent variable from the prior

        Args:
            batch_size (int, tuple, or list):

        """
        if isinstance(batch_size, int):
            shape = [batch_size]  # Convert int to list
        elif isinstance(batch_size, tuple):
            shape = list(batch_size)  # Convert tuple to list

        theta = self.theta_prior.sample(shape)

        return theta

    @torch.no_grad()
    def sample_data(self, batch_size, n_data):
        """ Sample designs """
        # data = self.theta_prior.sample([batch_size, n_data])
        data = self.data_sampler.sample([batch_size, n_data])
        return data[..., 0, :]  # [B, N, K, D] -> [B, N, D]

        # return data.reshape(batch_size, n_data, self.dim_x)  # [B, N, 2]

    def total_density(self, xi, theta):
        """Total density

        Shape:
            xi: [:, D] - [B, D] or [1/L, B, T, D]
            theta: [:, K, D] - [B, K, D] or [L, B, T, K, D]

        Returns:
            density: [:, 1]
        """
        # two norm squared
        sq_two_norm = (
            (xi.unsqueeze(-2).expand(theta.shape) - theta).pow(2).sum(axis=-1)
        )  # [:, K]
        sq_two_norm_inverse = (self.max_signal + sq_two_norm).pow(-1)
        # sum over the K sources, add base signal and take log.
        density = torch.log(
            self.base_signal + sq_two_norm_inverse.sum(-1, keepdim=True)
        )  # [:, 1]

        return density

    def forward(self, xi, theta):
        """ Experiment's outcome
            Using Differentiable Sampling for Reparameterization Trick!

        Args:
            xi [B, D]: normalised design
            theta [B, K, D]: sources

        Returns:
            observations: [B, 1]
        """

        signal = self.total_density(xi, theta)  # [B, 1]
        # Add noise
        noised_signal = dist.Normal(signal, self.noise_scale).rsample()
        return noised_signal

    def log_likelihood(self, y, xi, theta):
        """Log likelihood from gaussian noise

        Args:
            y [:, 1]
            xi [:, D]: real designs
            theta [:, K, D]

        Returns:
            log_prob: log likelihoods, [:, 1]
        """
        # uncorrupted signal
        signal = self.total_density(xi, theta)
        # Calculate the log likelihood
        log_prob = dist.Normal(signal, self.noise_scale).log_prob(y)
        return log_prob

    @torch.no_grad()
    def sample_batch(self, batch_size, with_query=True):
        """Sample a batch of data"""
        theta = self.sample_theta(batch_size)  # [B, K, D]

        if not with_query:
            self.n_query_init = 1

        num_samples = self.n_context_init + self.n_query_init

        # normalised design
        x = self.sample_data(batch_size, num_samples)
        y = self.forward(self.unnormalise_design(x),
                         theta.unsqueeze(1).expand(batch_size, num_samples, self.K, self.dim_x))  # [B, T, 1]

        # reshape
        theta = theta.reshape(batch_size, self.n_target_theta, 1)  # Reshape to match [B, K*D, 1]

        batch = AttrDict()
        batch.context_x = x[:, :self.n_context_init]
        batch.context_y = y[:, :self.n_context_init]
        batch.query_x = x[:, self.n_context_init:]
        batch.query_y = y[:, self.n_context_init:]
        batch.target_all = batch.target_theta = theta
        batch.n_target_theta = self.n_target_theta

        return batch

    def __str__(self) -> str:
        info = self.__dict__.copy()
        del_keys = []

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]
        return f"HiddenLocation({', '.join('{}={}'.format(key, val) for key, val in info.items())})"


if __name__ == "__main__":
    task = HiddenLocation()
    task.sample_batch(10)
