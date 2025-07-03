import torch
import torch.nn as nn
import torch.distributions as dist
from attrdictionary import AttrDict
from tasks.base_task import Task
from distributions import CensoredSigmoidNormal


class CESTask(Task):
    """Simulate Constant Elasticity of Substitution (CES) utility experiments"""

    def __init__(
            self,
            name: str = "CES",
            dim_x: int = 6,  # dimension of input (2 baskets x 3 commodities)
            dim_y: int = 1,  # dimension of outcome (preference rating)
            embedding_type="theta",  # mode of the experiment
            n_theta: int = 5,  # number of parameters: [rho, alpha1, alpha2, alpha3, u]
            n_context_init: int = 5,  # number of initial context points
            n_query_init: int = 300,  # number of initial query points
            design_scale: int = 100,  # scale of the design space (baskets in [0,100])
            noise_scale: float = 0.005,  # sigma_eta in the model
            epsilon: float = 2 ** (-22),  # epsilon for response clipping
            **kwargs
    ) -> None:
        super(CESTask, self).__init__(dim_x=dim_x, dim_y=dim_y, n_theta=n_theta)

        self.basket_dim = 3  # Each basket has 3 commodities
        self.n_theta = n_theta
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.design_scale = design_scale
        self.noise_scale = noise_scale
        self.epsilon = epsilon

        # Define priors for each parameter following the paper
        # 1. rho - Beta prior on [0,1]
        self.rho_a = 1.0
        self.rho_b = 1.0
        self.rho_prior = dist.Beta(self.rho_a, self.rho_b)

        # 2. alpha - Dirichlet prior
        self.alpha_concentration = torch.ones(self.basket_dim)
        self.alpha_prior = dist.Dirichlet(self.alpha_concentration)

        # 3. u - Log-normal prior
        self.u_mu = 1.0
        self.u_sigma = 3.0
        self.log_u_prior = dist.Normal(self.u_mu, self.u_sigma)

    @torch.no_grad()
    def sample_theta(self, batch_size):
        """ Sample parameters from the prior
        
        Args:
            batch_size (int, tuple, or list)
        """
        if isinstance(batch_size, int):
            shape = [batch_size]  # Convert int to list
        elif isinstance(batch_size, tuple):
            shape = list(batch_size)  # Convert tuple to list

        # Sample rho ~ Beta(1,1)
        rho = self.rho_prior.sample(shape)
        # regularization trick
        rho = 0.01 + 0.99 * rho  # [B, 1]

        # Sample alpha ~ Dirichlet(1,1,1)
        alpha = self.alpha_prior.sample(shape)  # [B, 3]

        # Sample log(u) ~ N(1,3) then transform to u
        log_u = self.log_u_prior.sample(shape)
        # u = torch.exp(log_u)

        # Stack parameters into a single tensor [B, 5]
        theta = torch.cat([
            rho.unsqueeze(-1),  # [B, 1]
            alpha,  # [B, 3]
            log_u.unsqueeze(-1)  # [B, 1]
        ], dim=-1)  # [B, 5]

        return theta

    @torch.no_grad()
    def sample_data(self, batch_size, n_data):
        """Sample basket pairs"""
        # Sample uniform baskets across the range [0, design_scale]
        # Each data point consists of two baskets (x, x')
        baskets1 = torch.rand(batch_size, n_data, self.basket_dim) * self.design_scale # TOCHECK: design scale
        baskets2 = torch.rand(batch_size, n_data, self.basket_dim) * self.design_scale

        # Combine both baskets into a single tensor
        data = torch.cat([baskets1, baskets2], dim=-1)  # [B, N, 6]

        return data  # [B, N, 6]

    def utility(self, x, rho, alpha):
        """Calculate CES utility for a basket U(x) = (sum_i alpha_i * x_i^rho)^(1/rho)

        Args:
            x: basket of goods [B, basket_dim] or [B, T, basket_dim]
            rho: elasticity parameter [(L,) B, 1]
            alpha: weights for each good [(L,) B, (T), basket_dim] # or [B, 1, basket_dim]

        Returns:
            utility value [B, 1] or [B, T, 1]
        """
        # Calculate CES utility
        x_pow_rho = x ** rho

        # Compute weighted sum
        weighted_sum = torch.sum(alpha * x_pow_rho, dim=-1, keepdim=True)

        # Compute utility U(x) = (weighted_sum)^(1/rho)
        utility = weighted_sum ** (1. / rho)

        return utility


    def normalise_design(self, x):
        """Normalize design if needed"""
        return x
    
    def normalise_outcomes(self, y):
        """Normalize preference ratings if needed"""
        return y

    def forward(self, xi, theta):
        """Generate preference rating

        Args:
            xi: basket pairs [B, 6] or [B, T, 6]
            theta: parameters [B, 5, 1]

        Returns:
            preference rating [B, 1] or [B, T, 1]
        """
        # Extract parameters
        rho = theta[..., 0:1]  # [B, (T), 1]
        alpha = theta[..., 1:4]  # [B, (T), 3]
        log_u = theta[..., 4:5]  # [B, (T), 1]
        u = torch.exp(log_u)

        xi = torch.clamp(xi, min=0.01, max=100.0)
        basket1 = xi[..., :self.basket_dim]  # [B, (T), 3]
        basket2 = xi[..., self.basket_dim:]  # [B, (T), 3]
        

        # Calculate utility for each basket
        u1 = self.utility(basket1, rho, alpha)  # [B, 1] or [B, T, 1]
        u2 = self.utility(basket2, rho, alpha)  # [B, 1] or [B, T, 1]

        # Calculate utility difference
        utility_diff = u1 - u2  # [B, 1] or [B, T, 1]

        # Calculate the mean of the response distribution
        mu_eta = utility_diff * u  # [B, 1] or [B, T, 1]

        # Calculate the standard deviation (noise level)
        basket_diff = basket1 - basket2
        basket_dist = torch.norm(basket_diff, dim=-1, p=2, keepdim=True)  # [B, 1] or [B, T, 1]
        sigma_eta = (1 + basket_dist) * self.noise_scale * u  # [B, 1] or [B, T, 1]

        y = CensoredSigmoidNormal(mu_eta, sigma_eta, self.epsilon, 1-self.epsilon).rsample()

        return y

    def log_likelihood(self, y, xi, theta):
        """Calculate log likelihood of observation

        Args:
            y: preference rating [1, B, T, 1]
            xi: basket pairs [1, B, T, 6]
            theta: parameters [L, B, (T), 5]

        Returns:
            log likelihood [L, B, T, 1]
        """
        # Extract parameters
        rho = theta[..., 0:1]  # [L, B, (T), 1]
        alpha = theta[..., 1:4]  # [L, B, (T), 3]
        log_u = theta[..., 4:5]  # [L, B, (T), 1]
        u = torch.exp(log_u)

        # Split the input into two baskets
        xi = torch.clamp(xi, min=0.01, max=100.0)
        basket1 = xi[..., :self.basket_dim]  # [1, B, T, 3]
        basket2 = xi[..., self.basket_dim:]  # [1, B, T, 3]

        # Calculate utility for each basket
        u1 = self.utility(basket1, rho, alpha)  # [L, B, T, 1]
        u2 = self.utility(basket2, rho, alpha)  # [L, B, T, 1]

        # Calculate utility difference
        utility_diff = u1 - u2  # [L, B, T, 1]

        # Calculate the mean and std of the response distribution
        mu_eta = utility_diff * u  # [L, B, T, 1]

        # Calculate the standard deviation (noise level)
        basket_diff = basket1 - basket2
        basket_dist = torch.norm(basket_diff, dim=-1, p=2, keepdim=True)  # [L, B, T, 1]
        sigma_eta = (1 + basket_dist) * self.noise_scale * u  # [L, B, T, 1]
        
        log_prob = CensoredSigmoidNormal(mu_eta, sigma_eta, self.epsilon, 1-self.epsilon).log_prob(y)

        return log_prob

    @torch.no_grad()
    def sample_batch(self, batch_size):
        """Sample a batch of normalised data"""
        theta = self.sample_theta(batch_size)
        # reshape theta
        theta = theta.reshape(batch_size, self.n_theta, 1)  # [B, 5, 1]
        x = self.sample_data(batch_size, self.n_context_init + self.n_query_init)

        # Generate responses for each basket pair parallely
        y = self.forward(x, theta.squeeze(-1).unsqueeze(-2)) # theta [B, T, 5]

        # Normalize x
        x = self.normalise_design(x)

        batch = AttrDict()
        batch.context_x = x[:, :self.n_context_init]
        batch.context_y = y[:, :self.n_context_init]
        batch.query_x = x[:, self.n_context_init:]
        batch.query_y = y[:, self.n_context_init:]
        batch.target_all = batch.target_theta = theta
        batch.n_theta = self.n_theta

        return batch

    def __str__(self) -> str:
        info = self.__dict__.copy()
        del_keys = []

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]
        return f"CESTask({', '.join('{}={}'.format(key, val) for key, val in info.items() if not callable(val))})"


if __name__ == "__main__":
    task = CESTask()
    print(task.sample_batch(10))