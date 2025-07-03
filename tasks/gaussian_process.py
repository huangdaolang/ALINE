import torch
import numpy as np
import torch.nn as nn
import torch.distributions as dist
from attrdictionary import AttrDict
from tasks.base_task import Task


class GPTask(Task):
    """Gaussian Process Task for amortized inference with support for isotropic/anisotropic kernels"""

    def __init__(
            self,
            name: str = "GP",
            dim_x: int = 1,  # dimension of input
            dim_y: int = 1,  # dimension of output
            embedding_type="mix",  # mode of the experiment: "data", "theta", or "mix"
            n_context_init: int = 5,  # number of initial context points
            n_query_init: int = 10,  # number of initial query points
            n_target_theta: int = 2,  # number of parameters: [lengthscale, variance]
            n_target_data: int = 5,  # number of target points (for data and mix modes)
            design_scale=None,  # scale of the design space
            noise_scale: float = 0.01,  # noise scale
            p_iso: float = 0.5,  # probability of isotropic kernel
            kernel_weights=None,
            lengthscale_lower=0.1,
            lengthscale_upper=2.0,
            **kwargs
    ) -> None:
        super().__init__(dim_x=dim_x, dim_y=dim_y)

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.n_target_theta = n_target_theta
        self.n_target_data = n_target_data

        self.embedding_type = embedding_type
        self.jitter = 1e-5
        self.p_iso = p_iso  # probability of isotropic kernel
        # self.kernel_weights = kernel_weights if kernel_weights is not None else [0.25, 0.25, 0.25, 0.25]
        self.kernel_weights = kernel_weights if kernel_weights is not None else [1/3, 0, 1/3, 1/3]
        self.kernel_types = ["rbf", "matern12", "matern32", "matern52"]

        if self.embedding_type in ["mix", "theta"]:
            if self.n_target_theta != dim_x + 1:
                raise ValueError("n_target_theta must be equal to dim_x + 1 for theta or mix embedding type")
        else:
            self.n_target_theta = 0

        # Define priors for GP hyperparameters
        # Lengthscale prior (now for each dimension)
        base_ls_factor = torch.sqrt(torch.tensor(dim_x, dtype=torch.float))
        self.lengthscale_lower = lengthscale_lower * base_ls_factor
        self.lengthscale_upper = lengthscale_upper * base_ls_factor

        # Variance
        self.scale_lower = 0.1
        self.scale_upper = 1.0

        self.noise_scale = noise_scale

        self.design_scale = torch.tensor(design_scale) if design_scale is not None else torch.tensor(5.0)

    @torch.no_grad()
    def sample_theta(self, batch_size):
        """Sample hyperparameters from the prior, supporting both isotropic and anisotropic kernels"""
        # Sample per-dimension lengthscales
        ls_range = self.lengthscale_upper - self.lengthscale_lower
        length_scales = self.lengthscale_lower + ls_range * torch.rand(batch_size, self.dim_x)

        # Determine if each batch uses isotropic kernel (same lengthscale for all dimensions)
        is_iso = torch.bernoulli(torch.ones(batch_size) * self.p_iso).bool()

        new_length_scales = length_scales.clone()
        for b in range(batch_size):
            if is_iso[b]:
                new_length_scales[b, :] = length_scales[b, 0]
        length_scales = new_length_scales

        # Sample variance
        scale = self.scale_lower + (self.scale_upper - self.scale_lower) * torch.rand(batch_size)

        # Stack all parameters: [lengthscales, variance]
        theta = torch.cat([
            length_scales,  # [B, D]
            scale.unsqueeze(1),  # [B, 1]
        ], dim=1)

        return theta.unsqueeze(2)  # [B, D+2, 1]

    @torch.no_grad()
    def sample_data(self, batch_size, n_data):
        """Sample input points from the domain"""
        # Simple uniform random sampling in the design space
        return torch.rand(batch_size, n_data, self.dim_x) * 2 * self.design_scale - self.design_scale

    def to_design_space(self, xi):
        """Convert normalized design to actual input domain"""
        return xi * self.design_scale

    def normalise_outcomes(self, y):
        """Normalize outcomes if needed"""
        return y

    def rbf_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute RBF kernel between x1 and x2 with support for both isotropic and anisotropic kernels

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x] (can be all same for isotropic)
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        # Reshape for broadcasting: [N, 1, dim_x] - [1, M, dim_x]
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        # Calculate squared distance with per-dimension lengthscales
        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        # Sum across feature dimensions for RBF kernel
        sq_dist = weighted_sq_diff.sum(dim=-1)  # [N, M]

        # Apply RBF kernel formula
        kernel = scale * torch.exp(-0.5 * sq_dist)

        return kernel  # [N, M]

    def matern12_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute Matérn 1/2 kernel (exponential kernel) between x1 and x2

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        # Reshape for broadcasting
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        # Calculate weighted distances with per-dimension lengthscales
        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        # Compute Euclidean distance
        dist = torch.sqrt(weighted_sq_diff.sum(dim=-1))  # [N, M]

        # Apply Matérn 1/2 kernel formula
        kernel = scale * torch.exp(-dist)

        return kernel  # [N, M]

    def matern32_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute Matérn 3/2 kernel between x1 and x2

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        # Reshape for broadcasting
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        # Calculate weighted distances with per-dimension lengthscales
        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        # Compute Euclidean distance
        dist = torch.sqrt(weighted_sq_diff.sum(dim=-1))  # [N, M]

        # Apply Matérn 3/2 kernel formula
        sqrt3 = torch.sqrt(torch.tensor(3.0, device=x1.device))
        kernel = scale * (1 + sqrt3 * dist) * torch.exp(-sqrt3 * dist)

        return kernel  # [N, M]

    def matern52_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute Matérn 5/2 kernel between x1 and x2

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        # Reshape for broadcasting
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        # Calculate weighted distances with per-dimension lengthscales
        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        # Compute Euclidean distance
        dist = torch.sqrt(weighted_sq_diff.sum(dim=-1))  # [N, M]

        # Apply Matérn 5/2 kernel formula
        sqrt5 = torch.sqrt(torch.tensor(5.0, device=x1.device))
        kernel = scale * (1 + sqrt5 * dist + (5.0 / 3.0) * dist ** 2) * torch.exp(-sqrt5 * dist)

        return kernel  # [N, M]

    def compute_kernel_matrix(self, x1, x2, lengthscales, scale, kernel_type):
        """
        Compute kernel matrix based on the specified kernel type

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)
            kernel_type: one of "rbf", "matern12", "matern32", "matern52"

        Returns:
            kernel matrix [N, M]
        """
        if kernel_type == "rbf":
            return self.rbf_kernel(x1, x2, lengthscales, scale)
        elif kernel_type == "matern12":
            return self.matern12_kernel(x1, x2, lengthscales, scale)
        elif kernel_type == "matern32":
            return self.matern32_kernel(x1, x2, lengthscales, scale)
        elif kernel_type == "matern52":
            return self.matern52_kernel(x1, x2, lengthscales, scale)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def sample_kernel_type(self, batch_size):
        """
        Sample kernel types for each batch element based on weights

        Args:
            batch_size: Number of batches to generate kernel types for

        Returns:
            List of kernel types, one for each batch element
        """
        # Convert weights to PyTorch tensor and normalize
        weights = torch.tensor(self.kernel_weights, dtype=torch.float)
        weights = weights / weights.sum()

        # Sample indices based on weights
        indices = torch.multinomial(weights, batch_size, replacement=True)

        # Map indices to kernel types
        kernel_types = [self.kernel_types[idx] for idx in indices]

        return kernel_types

    def generate_gp_data(self, x, theta):
        """
        Generate GP data for all points at once using proper correlation

        Args:
            x: input locations [B, N, dim_x]
            theta: GP hyperparameters [B, dim_x+2, 1]

        Returns:
            GP function values with noise [B, N, 1]
        """
        batch_size, n_points, _ = x.shape

        # Extract parameters:
        # First dim_x elements are lengthscales, then variance, then noise
        lengthscales = theta[:, :self.dim_x, 0]  # [B, dim_x]
        scale = theta[:, self.dim_x, 0]  # [B]

        # Sample kernel types for each batch element
        kernel_types = self.sample_kernel_type(batch_size)

        # Result tensor
        samples = torch.zeros(batch_size, n_points, 1)

        # Process each batch independently
        for b in range(batch_size):
            # Compute kernel matrix with batch-specific lengthscales and kernel type
            K = self.compute_kernel_matrix(
                x[b], x[b], lengthscales[b], scale[b], kernel_types[b]
            )  # [N, N]

            # Add jitter for numerical stability
            K = K + self.jitter * torch.eye(n_points)

            # Sample from multivariate normal
            try:
                # Try Cholesky decomposition method (most efficient)
                L = torch.linalg.cholesky(K)
                z = torch.randn(n_points)
                f = torch.matmul(L, z)
            except:
                # Alternative: use MultivariateNormal directly
                mvn = dist.MultivariateNormal(
                    loc=torch.zeros(n_points),
                    covariance_matrix=K
                )
                f = mvn.sample()

            # Add observation noise
            samples[b, :, 0] = f + self.noise_scale * torch.randn(n_points)

        return samples

    def forward(self, xi, theta):
        """
        Generate observations from a GP

        Args:
            xi: normalized input locations [B, N, dim_x] or [B, dim_x]
            theta: GP hyperparameters [B, dim_x+2, 1]

        Returns:
            noisy observations [B, N, 1] or [B, 1]
        """
        # Convert to actual input domain
        x = self.to_design_space(xi)

        # Check if input is for single points per batch or multiple points
        is_single_point_per_batch = len(x.shape) == 2

        if is_single_point_per_batch:
            # Handle single point per batch case
            batch_size = x.shape[0]
            x = x.unsqueeze(1)  # [B, 1, dim_x]

            # Generate data
            y = self.generate_gp_data(x, theta)

            # Return with original shape
            return y.squeeze(1)
        else:
            # Handle multiple points per batch case
            return self.generate_gp_data(x, theta)

    def sample_batch(self, batch_size):
        """Sample a batch of data based on the mode"""
        # Initialize the dictionary to return
        batch = AttrDict()

        # Sample hyperparameters
        theta = self.sample_theta(batch_size)

        # Sample input points based on the mode
        if self.embedding_type == "theta":
            # Only context and query points
            n_total = self.n_context_init + self.n_query_init
            x = self.sample_data(batch_size, n_total)

            # Generate all observations at once (more efficient)
            y = self.generate_gp_data(x, theta)

            # Split into context and query
            batch.context_x = x[:, :self.n_context_init]
            batch.context_y = y[:, :self.n_context_init]
            batch.query_x = x[:, self.n_context_init:]
            batch.query_y = y[:, self.n_context_init:]

            # For theta mode, target is the hyperparameters
            batch.target_all = batch.target_theta = theta

        elif self.embedding_type == "data":
            # Context, query, and target points
            n_total = self.n_context_init + self.n_query_init + self.n_target_data
            x = self.sample_data(batch_size, n_total)

            # Generate all observations at once
            y = self.generate_gp_data(x, theta)

            # Split into context, query, and target
            batch.context_x = x[:, :self.n_context_init]
            batch.context_y = y[:, :self.n_context_init]
            batch.query_x = x[:, self.n_context_init:self.n_context_init + self.n_query_init]
            batch.query_y = y[:, self.n_context_init:self.n_context_init + self.n_query_init]
            batch.target_x = x[:, self.n_context_init + self.n_query_init:]
            batch.target_y = y[:, self.n_context_init + self.n_query_init:]

            # For data mode, target is the target data
            batch.target_all = batch.target_y
            batch.target_theta = None  # No target theta in data mode

        else:  # "mix" mode
            # Context, query, and target points
            n_total = self.n_context_init + self.n_query_init + self.n_target_data
            x = self.sample_data(batch_size, n_total)

            # Generate all observations at once
            y = self.generate_gp_data(x, theta)

            # Split into context, query, and target
            batch.context_x = x[:, :self.n_context_init]
            batch.context_y = y[:, :self.n_context_init]
            batch.query_x = x[:, self.n_context_init:self.n_context_init + self.n_query_init]
            batch.query_y = y[:, self.n_context_init:self.n_context_init + self.n_query_init]
            batch.target_x = x[:, self.n_context_init + self.n_query_init:]
            batch.target_y = y[:, self.n_context_init + self.n_query_init:]

            # For mix mode, both data and theta are targets
            batch.target_theta = theta


            # Combine target data and theta for target_all
            batch.target_all = torch.cat([batch.target_y, batch.target_theta], dim=1)

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
        return f"GPTask({', '.join('{}={}'.format(key, val) for key, val in info.items())})"



if __name__ == "__main__":
    # Import visualization libraries
    import matplotlib.pyplot as plt
    import matplotlib
    import argparse

    parser = argparse.ArgumentParser(description='GPTask demonstration or offline data generation')
    parser.add_argument('--dim_x', type=int, default=1, help='Input dimension')
    parser.add_argument('--embedding_type', type=str, default='mix', help='Embedding type: data, theta, or mix')
    args = parser.parse_args()

    # Create the GP task
    task = GPTask(
        dim_x=args.dim_x,
        embedding_type=args.embedding_type,
        n_context_init=1,
        n_query_init=500,
        n_target_data=200,
        noise_scale=0.01,
        design_scale=5
    )

    batch = task.sample_batch(20)

    plt.figure(figsize=(15, 10))

    matplotlib.rcParams.update({
        'font.family': 'times',
        'font.size': 14.0,
        'lines.linewidth': 2,
        'lines.antialiased': True,
        'axes.facecolor': 'fdfdfd',
        'axes.edgecolor': '777777',
        'axes.linewidth': 1,
        'axes.titlesize': 'medium',
        'axes.labelsize': 'medium',
        'axes.axisbelow': True,
        'xtick.major.size': 0,  # major tick size in points
        'xtick.minor.size': 0,  # minor tick size in points
        'xtick.major.pad': 6,  # distance to major tick label in points
        'xtick.minor.pad': 6,  # distance to the minor tick label in points
        'xtick.color': '333333',  # color of the tick labels
        'xtick.labelsize': 'medium',  # fontsize of the tick labels
        'xtick.direction': 'in',  # direction: in or out
        'ytick.major.size': 0,  # major tick size in points
        'ytick.minor.size': 0,  # minor tick size in points
        'ytick.major.pad': 6,  # distance to major tick label in points
        'ytick.minor.pad': 6,  # distance to the minor tick label in points
        'ytick.color': '333333',  # color of the tick labels
        'ytick.labelsize': 'medium',  # fontsize of the tick labels
        'ytick.direction': 'in',  # direction: in or out
        'axes.grid': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 1,
        'legend.fancybox': True,
        'legend.fontsize': 'Small',
        'figure.figsize': (2.5, 2.5),
        'figure.facecolor': '1.0',
        'figure.edgecolor': '0.5',
        'hatch.linewidth': 0.1,
        'text.usetex': True
    })

    # Only visualize 1D GPs
    if args.dim_x == 1:
        for i in range(20):
            # Create subplot
            plt.subplot(5, 4, i + 1)


            # Plot the sampled points
            # plt.scatter(batch.context_x[i], batch.context_y[i], c='blue', label='Context' if i == 0 else "", s=10)

            query_x_i = batch.query_x[i].squeeze().numpy()  # Squeeze to 1D numpy array
            query_y_i = batch.query_y[i].squeeze().numpy()  # Squeeze to 1D numpy array

            sorted_indices = np.argsort(query_x_i)
            query_x_sorted = query_x_i[sorted_indices]
            query_y_sorted = query_y_i[sorted_indices]
            plt.plot(query_x_sorted, query_y_sorted, c='C0', label='Context' if i == 0 else "")

            # plt.plot(batch.query_x[i], batch.query_y[i], c='red', label='Query' if i == 0 else "", s=10)
            plt.ylim(-3, 3)
            # if hasattr(batch, 'target_x'):
            #     plt.scatter(batch.target_x[i], batch.target_y[i], c='green', label='Target' if i == 0 else "", s=10)

            # Add title with hyperparameters
            lengthscale = batch.target_theta[i, 0, 0].item()
            variance = batch.target_theta[i, 1, 0].item()
            plt.title(f'Sample {i + 1}: ls = {lengthscale:.2f}, scale = {variance:.2f}')
            # plt.text(-5, 2, f'lengthscale={lengthscale:.2f}', fontsize=10)
            # plt.text(-5, 1, f'scale={variance:.2f}', fontsize=10)

            # if i == 0:
            #     plt.legend()

            plt.grid(False)

        plt.tight_layout()
        plt.savefig("../figures/gp_samples.pdf", dpi=300)
        plt.show()