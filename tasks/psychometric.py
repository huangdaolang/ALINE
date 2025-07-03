import torch
import torch.nn as nn
import torch.distributions as dist
from attrdictionary import AttrDict
from tasks.base_task import Task


class PsychometricTask(Task):
    """Simulate Psychometric Function Experiment"""

    def __init__(
            self,
            name: str = "Psychometric",
            dim_x: int = 1,  # dimension of stimulus intensity
            dim_y: int = 1,  # dimension of outcome (binary)
            embedding_type="theta",  # mode of the experiment
            n_target_theta: int = 4,  # number of parameters: [alpha, beta, gamma, lambda]
            n_context_init: int = 5,  # number of initial context points
            n_query_init: int = 300,  # number of initial query points
            design_scale: int = 5,  # scale of the design space
            **kwargs
    ) -> None:
        super(PsychometricTask, self).__init__(dim_x=dim_x, dim_y=dim_y, n_target_theta=n_target_theta)

        self.dim_x = dim_x
        self.n_target_theta = n_target_theta
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init

        # Define priors for each parameter
        # 1. Threshold (alpha) - Normal or Uniform
        self.alpha_lower = -3.0
        self.alpha_upper = 3.0
        self.alpha_prior = dist.Uniform(self.alpha_lower, self.alpha_upper)

        # 2. Slope (beta) - Gamma (must be positive)
        # self.beta_k = 2.0  # shape parameter
        # self.beta_theta = 0.5  # scale parameter
        # self.beta_prior = dist.Gamma(self.beta_k, 1.0 / self.beta_theta)
        self.beta_lower = 0.1
        self.beta_upper = 2.0
        self.beta_prior = dist.Uniform(self.beta_lower, self.beta_upper)

        # 3. Guess rate (gamma) - Beta
        self.gamma_lower = 0.1
        self.gamma_upper = 0.9  # Expect low guess rate (for yes/no) or use fixed (for nAFC)
        self.gamma_prior = dist.Uniform(self.gamma_lower, self.gamma_upper)

        # 4. Lapse rate (lambda) - Beta
        self.lambda_lower = 0
        self.lambda_upper = 0.5  # Expect very low lapse rate
        self.lambda_prior = dist.Uniform(self.lambda_lower, self.lambda_upper)

        self.design_scale = design_scale

    @torch.no_grad()
    def sample_theta(self, batch_size):
        """Sample parameters from the prior"""
        # Sample all parameters
        alpha = self.alpha_prior.sample([batch_size])
        beta = self.beta_prior.sample([batch_size])
        gamma = self.gamma_prior.sample([batch_size])
        lmbda = self.lambda_prior.sample([batch_size])

        # Stack parameters
        theta = torch.stack([alpha, beta, gamma, lmbda], dim=1)

        return theta.reshape(batch_size, 4, 1)  # [B, 4, 1]

    @torch.no_grad()
    def sample_data(self, batch_size, n_data):
        """Sample stimulus intensities"""
        # For adaptive methods, this would be more complex
        # Here we simply sample uniform stimuli across the range
        data = torch.rand(batch_size, n_data, self.dim_x) * 2 * self.design_scale - self.design_scale

        return data  # [B, N, 1]

    def psychometric_function(self, x, theta):
        """Evaluate the psychometric function

        Args:
            x: stimulus intensity [B, 1] or [B, T, 1]
            theta: parameters [B, 4, 1] or [B, 4] or [L, B, 4]

        Returns:
            probability of correct response [B, 1] or [B, T, 1]
        """
        # Reshape theta if needed
        if len(theta.shape) == 2:
            theta = theta.unsqueeze(-1)  # [B, 4] -> [B, 4, 1]

        # Extract parameters
        alpha = theta[:, 0, :]  # threshold
        beta = theta[:, 1, :]  # slope
        gamma = theta[:, 2, :]  # guess rate
        lmbda = theta[:, 3, :]  # lapse rate

        # Calculate internal value using Gumbel function (like in Prins 2013)
        z = (x - alpha) / beta
        F = 1 - torch.exp(-10 ** (z))  # Gumbel function

        # Apply guessing and lapsing
        p = lmbda*gamma + (1 - lmbda) * F

        return p

    def to_design_space(self, xi):
        """Convert normalized design to actual stimulus intensity"""
        return xi

    def normalise_outcomes(self, y):
        """Normalize binary outcomes if needed"""
        return y

    def forward(self, xi, theta):
        """Generate binary response

        Args:
            xi: normalized stimulus intensity [B, 1]
            theta: parameters [B, 4, 1]

        Returns:
            binary response [B, 1]
        """
        xi = self.to_design_space(xi)  # Convert to actual stimulus intensity

        # Get probability of response
        p = self.psychometric_function(xi, theta)

        # Sample Bernoulli response
        y = torch.bernoulli(p)

        return y

    def log_likelihood(self, y, xi, theta):
        """Calculate log likelihood of observation

        Args:
            y: binary response [B, 1] or [L, B, T, 1]
            xi: stimulus intensity [B, 1] or [L, B, T, 1]
            theta: parameters [B, 4, 1] or [L, B, 4, 1]

        Returns:
            log likelihood [B, 1] or [L, B, T, 1]
        """
        # Get probability from psychometric function
        p = self.psychometric_function(xi, theta)

        # Bernoulli log likelihood
        log_prob = y * torch.log(p + 1e-10) + (1 - y) * torch.log(1 - p + 1e-10)

        return log_prob

    def sample_batch(self, batch_size):
        """Sample a batch of data"""
        theta = self.sample_theta(batch_size)
        x = self.sample_data(batch_size, self.n_context_init + self.n_query_init)
        y = torch.empty(batch_size, self.n_context_init + self.n_query_init, self.dim_y)

        # Generate responses for each stimulus
        for i in range(self.n_context_init + self.n_query_init):
            y[:, i, :] = self.forward(x[:, i, :], theta)

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
        return f"PsychometricTask({', '.join('{}={}'.format(key, val) for key, val in info.items())})"


if __name__ == "__main__":
    task = PsychometricTask()
    print(task.sample_batch(10))