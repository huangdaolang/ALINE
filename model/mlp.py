import torch
import torch.nn as nn


class EncoderNetwork(nn.Module):
    """ Encoder network
        input_dim -> encoding_dim
    """

    def __init__(self, design_dim, osbervation_dim, hidden_dim, encoding_dim):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.design_dim = design_dim
        input_dim = self.design_dim + osbervation_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, xi, y):

        inputs = torch.cat([xi, y], dim=-1)


        x = self.linear1(inputs)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


class EmitterNetwork(nn.Module):
    """ Emitter network 
        encoding_dim -> design_dim_flat
    """

    def __init__(self, encoding_dim, design_dim):
        super().__init__()
        self.design_dim = design_dim
        self.linear = nn.Linear(encoding_dim, self.design_dim)

    def forward(self, r):
        xi = self.linear(r)
        return xi



class SetEquivariantDesignNetwork(nn.Module):
    """
    DAD (Encoder + Emitter)

    """
    def __init__(
        self,
        encoder_network,
        emission_network,
        dim_x,
        dim_y,
        empty_value
    ):
        super().__init__()
        self.encoder = encoder_network
        self.emitter = emission_network
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.register_buffer("prototype", empty_value.clone())
        self.register_parameter("empty_value", nn.Parameter(empty_value))


    def forward(self, xi, y):
        """ Generate design

        Args:
            xi [B, t, D]
            y [B, t, 1]

        """
        B, t, _ = xi.shape
        # Generate the first design
        if t == 0:
            sum_encoding = self.empty_value.new_zeros((B, self.encoder.encoding_dim))

        else:
            # Pooling
            sum_encoding = self.encoder(xi, y).sum(1)
        output = self.emitter(sum_encoding)

        return output
    
    @torch.no_grad()
    def run_trace(self, experiment, T, M):
        """ Run M parallel experiments and record the traces

        Args:
            experiment (BED): experiment simulator
            T (int): number of steps in a experiment trajectory
            M (int): number of rollouts
        """
        self.eval()

        theta = experiment.sample_theta((M, ))

        # history of an experiment
        xi_designs = torch.empty((M, T, self.dim_x))  # normalised designs [1, T, D_x]
        y_outcomes = torch.empty((M, T, self.dim_y))  # [1, T, D_y]

        # T-steps experiment
        for t in range(T):
            xi = self.forward(xi_designs[:, :t], y_outcomes[:, :t])     # [B, D_x]
            y = experiment(xi, theta)                                   # [B, D_y]

            xi_designs[:, t] = xi
            y_outcomes[:, t] = y

        # convert designs to design space
        xi_designs = experiment.to_design_space(xi_designs)

        return theta, xi_designs, y_outcomes