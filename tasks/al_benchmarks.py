import torch
import numpy as np
from attrdictionary import AttrDict
from tasks.base_task import Task


class BenchmarkTask(Task):
    """Benchmark Task for sampling from standard test functions"""

    def __init__(
            self,
            name: str = "Benchmark",
            dim_x: int = 1,  # dimension of input
            dim_y: int = 1,  # dimension of output
            n_context_init: int = 5,  # number of initial context points
            n_query_init: int = 10,  # number of initial query points
            n_target_data: int = 5,  # number of target points
            design_scale=5.0,  # scale of the design space (-design_scale to design_scale)
            noise_scale=0.1,  # noise level for function observations
            **kwargs
    ) -> None:
        super().__init__(dim_x=dim_x, dim_y=dim_y, n_target_theta=0)

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.n_target_data = n_target_data
        self.noise_scale = noise_scale
        self.design_scale = torch.tensor(design_scale)

        # Register known benchmark functions
        self.benchmark_functions = {
            'forrester': {
                'dim': 1,
                'domain': (0.0, 1.0),
                'func': self._forrester
            },
            'branin': {
                'dim': 2,
                'domain': [(0.0, 1.0), (0.0, 1.0)],
                'func': self._branin
            },
            'gramacy1d': {
                'dim': 1,
                'domain': (0.5, 2.5),  # Corrected domain
                'func': self._gramacy1d
            },
            'gramacy2d': {
                'dim': 2,
                'domain': [(-2.0, 6.0), (-2.0, 6.0)],
                'func': self._gramacy2d
            },
            'higdon': {
                'dim': 1,
                'domain': (0.0, 20.0),
                'func': self._higdon
            },
            'ackley2d': {
                'dim': 2,
                'domain': [(-2.0, 2.0), (-2.0, 2.0)],
                'func': self._ackley
            },
            'three_hump_camel': {
                'dim': 2,
                'domain': [(-2.0, 2.0), (-2.0, 2.0)],
                'func': self._three_hump_camel
            },
            'goldstein_price': {
                'dim': 2,
                'domain': [(-2.0, 2.0), (-2.0, 2.0)],
                'func': self._goldstein_price
            }
        }

    def _forrester(self, x):
        """
        Forrester function (1D)
        Domain: [0, 1]
        """
        return (torch.pow(6 * x - 2, 2) * torch.sin(12 * x - 4)) / 5

    def _branin(self, x):
        r"""
        Branin function as rescaled by Picheny et al. (2012), on domain [0,1]^2.
        """
        # Unpack input
        x1 = x[..., 0]
        x2 = x[..., 1]

        # Map [0,1] -> [0,15]
        x1p = 15.0 * x1
        x2p = 15.0 * x2

        # Branin constants
        a = 1.0
        b = 5.1 / (4.0 * np.pi ** 2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        # Standard Branin terms
        term1 = a * (x2p - b * x1p ** 2 + c * x1p - r) ** 2
        term2 = s * (1.0 - t) * torch.cos(x1p)
        branin_val = term1 + term2 + s

        # Picheny rescaling: shift and scale
        return (branin_val - 44.81) / 51.95

    def _gramacy1d(self, x):
        """
        Gramacy & Lee function (1D)
        Domain: [0.5, 2.5]
        """
        return (torch.sin(10 * np.pi * x) / (2 * x) + (x - 1) ** 4)/3

    def _gramacy2d(self, x):
        """
        Gramacy & Lee function (2D)
        Domain: [-2, 6] x [-2, 6]
        """
        x1, x2 = x[..., 0], x[..., 1]
        return x1 * torch.exp(-x1 ** 2 - x2 ** 2)

    def _higdon(self, x):
        """
        Higdon function (1D)
        Domain: [0, 20]
        """
        # Create a piecewise function similar to numpy's piecewise
        result = torch.zeros_like(x)

        # For x < 10
        mask_lt_10 = x < 10
        result[mask_lt_10] = torch.sin(np.pi * x[mask_lt_10] / 5) + 0.2 * torch.cos(4 * np.pi * x[mask_lt_10] / 5)

        # For x >= 10
        mask_ge_10 = ~mask_lt_10
        result[mask_ge_10] = x[mask_ge_10] / 10 - 1

        return result

    def _ackley(self, x):
        """
        Ackley function (2D)
        Domain: [-5, 5] for each variable
        """
        x1, x2 = x[..., 0], x[..., 1]
        term1 = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))
        term2 = -torch.exp(0.5 * (torch.cos(2 * np.pi * x1) + torch.cos(2 * np.pi * x2)))
        return (term1 + term2 + 20 + np.e) / 5

    def _three_hump_camel(self, x):
        """
        Three-Hump Camel function (2D)
        Domain: [-2, 2] for each variable
        Output range: [0, ~2]
        """
        x1, x2 = x[..., 0], x[..., 1]
        term1 = 2 * x1 ** 2
        term2 = -1.05 * x1 ** 4
        term3 = x1 ** 6 / 6
        term4 = x1 * x2
        term5 = x2 ** 2

        return term1 + term2 + term3 + term4 + term5

    def _goldstein_price(self, x):
        """
        Goldstein-Price function (2D), scaled to have output range close to [-2, 2]
        Original domain: [-2, 2] for each variable
        Scaled output range: approximately [-2, 2]
        """
        x1, x2 = x[..., 0], x[..., 1]

        # First part
        part1a = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)

        # Second part
        part2a = 30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)

        # Original function
        original = part1a * part2a

        # Log transform and scale to get reasonable values
        transformed = torch.log(original) - 6  # subtract 6 to center around 0
        scaled = transformed / 4  # scale down to [-2, 2] range

        return scaled

    def _scale_input_to_domain(self, x_norm, benchmark_name):
        """
        Scales normalized inputs from [-design_scale, design_scale] to the benchmark function's domain
        """
        benchmark = self.benchmark_functions[benchmark_name]

        # Get the domain for the function
        domain = benchmark['domain']

        # If the domain is a tuple, it's the same for all dimensions
        if isinstance(domain, tuple):
            lower, upper = domain
            domain_range = upper - lower
            domain_mid = (upper + lower) / 2

            # Scale from [-design_scale, design_scale] to [lower, upper]
            return (x_norm / self.design_scale) * (domain_range / 2) + domain_mid

        # If the domain is a list of tuples, each dimension has a different range
        else:
            x_scaled = torch.zeros_like(x_norm)
            for i, (lower, upper) in enumerate(domain):
                domain_range = upper - lower
                domain_mid = (upper + lower) / 2

                # Scale this dimension
                x_scaled[..., i] = (x_norm[..., i] / self.design_scale) * (domain_range / 2) + domain_mid

            return x_scaled

    def forward(self, xi, benchmark_name):
        """
        Generate observations from a benchmark function

        Args:
            xi: normalized input locations [B, N, dim_x] or [B, dim_x]
            benchmark_name: name of the benchmark function to use

        Returns:
            noisy observations [B, N, 1] or [B, 1]
        """
        if benchmark_name not in self.benchmark_functions:
            raise ValueError(f"Unknown benchmark function: {benchmark_name}")

        # Get the benchmark function
        benchmark = self.benchmark_functions[benchmark_name]
        func = benchmark['func']

        # Check input dimensions
        if benchmark['dim'] != self.dim_x:
            raise ValueError(
                f"Benchmark function {benchmark_name} requires {benchmark['dim']} dimensions, but got {self.dim_x}")

        # Scale inputs to the appropriate domain
        x_scaled = self._scale_input_to_domain(xi, benchmark_name)
        # Check if input is for single points per batch or multiple points
        is_single_point_per_batch = len(x_scaled.shape) == 2

        if is_single_point_per_batch:
            # Handle single point per batch case
            batch_size = x_scaled.shape[0]
            x_scaled = x_scaled.unsqueeze(1)  # [B, 1, dim_x]

            # Evaluate function
            y = func(x_scaled)  # [B, 1]

            # Add noise and ensure shape is [B, 1]
            if len(y.shape) < 2:
                y = y.unsqueeze(-1)

            y = y + self.noise_scale * torch.randn_like(y)

            # Return with original shape
            return y
        else:
            # Handle multiple points per batch case
            # Evaluate function
            y = func(x_scaled)  # [B, N]

            # Add noise and ensure shape is [B, N, 1]
            if len(y.shape) < 3:
                y = y.unsqueeze(-1)

            y = y + self.noise_scale * torch.randn_like(y)

            return y

    def sample_data(self, batch_size, n_data):
        """Sample input points from the domain"""
        return torch.rand(batch_size, n_data, self.dim_x) * 2 * self.design_scale - self.design_scale

    def sample_batch(self, batch_size, benchmark_name, mode="data"):
        """Sample a batch of data from a specific benchmark function"""
        # Check if benchmark exists
        if benchmark_name not in self.benchmark_functions:
            raise ValueError(f"Unknown benchmark function: {benchmark_name}")

        # Check dimension compatibility
        benchmark = self.benchmark_functions[benchmark_name]
        if benchmark['dim'] != self.dim_x:
            raise ValueError(
                f"Benchmark function {benchmark_name} requires {benchmark['dim']} dimensions, but got {self.dim_x}")

        # Initialize the dictionary to return
        batch = AttrDict()

        # Sample input points
        n_total = self.n_context_init + self.n_query_init + self.n_target_data
        x = self.sample_data(batch_size, n_total)

        # Generate all observations at once
        y = self.forward(x, benchmark_name)

        # Split into context, query, and target
        batch.context_x = x[:, :self.n_context_init]
        batch.context_y = y[:, :self.n_context_init]
        batch.query_x = x[:, self.n_context_init:self.n_context_init + self.n_query_init]
        batch.query_y = y[:, self.n_context_init:self.n_context_init + self.n_query_init]
        batch.target_x = x[:, self.n_context_init + self.n_query_init:]
        batch.target_y = y[:, self.n_context_init + self.n_query_init:]

        if mode == "data":
            batch.target_all = batch.target_y
            batch.target_theta = None
            batch.n_target_theta = 0
        elif mode == "mix":
            batch.target_theta = torch.zeros(batch_size, self.dim_x + 1, 1)
            batch.target_all = torch.cat([batch.target_y, batch.target_theta], dim=1)


        return batch

    def __str__(self) -> str:
        info = self.__dict__.copy()
        del_keys = []

        for key in info.keys():
            if key[0] == "_" or key == "benchmark_functions":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        return f"BenchmarkTask({', '.join('{}={}'.format(key, val) for key, val in info.items())})"


if __name__ == "__main__":
    # Test the benchmark task
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a benchmark task
    task = BenchmarkTask(
        dim_x=1,
        n_context_init=5,
        n_query_init=10,
        n_target_data=20,
        noise_scale=0.1
    )

    # Sample a batch from the Forrester function
    batch = task.sample_batch(1, 'higdon')

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(batch.context_x[0].numpy(), batch.context_y[0].numpy(), c='blue', label='Context')
    plt.scatter(batch.query_x[0].numpy(), batch.query_y[0].numpy(), c='red', label='Query')
    plt.scatter(batch.target_x[0].numpy(), batch.target_y[0].numpy(), c='green', label='Target')

    # Plot the true function (densely sampled)
    x_dense = torch.linspace(-5, 5, 500).reshape(-1, 1)
    with torch.no_grad():
        y_dense = task.forward(x_dense, 'higdon')

    # Ensure they're 1D arrays for plotting
    x_plot = x_dense.reshape(-1).numpy()
    y_plot = y_dense.reshape(-1).numpy()

    plt.plot(x_plot, y_plot, 'k-', label='True Function')

    plt.title('Forrester Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Test additional functions if 2D visualization is needed
    if task.dim_x == 2:
        # For 2D functions like Branin
        from mpl_toolkits.mplot3d import Axes3D

        # Sample a batch from the Branin function
        task.dim_x = 2  # Change to 2D for this test
        batch = task.sample_batch(1, 'branin')

        # Create a meshgrid for visualization
        x1 = np.linspace(-5, 5, 50)
        x2 = np.linspace(-5, 5, 50)
        X1, X2 = np.meshgrid(x1, x2)

        # Convert to torch tensors
        grid_points = torch.tensor(np.vstack([X1.flatten(), X2.flatten()]).T, dtype=torch.float32)

        # Evaluate the function
        with torch.no_grad():
            Z = task.forward(grid_points, 'branin').numpy().reshape(X1.shape)

        # 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)

        # Plot the sampled points
        ax.scatter(batch.context_x[0, :, 0].numpy(), batch.context_x[0, :, 1].numpy(),
                   batch.context_y[0].numpy(), c='blue', s=50, label='Context')

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x1, x2)')
        ax.set_title('Branin Function')
        plt.colorbar(surf)
        plt.legend()
        plt.show()