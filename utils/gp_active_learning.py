import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from typing import Dict, Tuple, List, Callable, Optional, Union
from attrdictionary import AttrDict
import time


def uncertainty_sampling(
        gp: GaussianProcessRegressor,
        X_pool: np.ndarray,
        **kwargs
) -> np.ndarray:
    """
    Uncertainty sampling acquisition function: selects the point with highest predictive variance

    Args:
        gp: Trained Gaussian Process model
        X_pool: Pool of candidate points to select from

    Returns:
        Acquisition function values for each candidate point
    """
    _, std = gp.predict(X_pool, return_std=True)
    return std


def gp_ucb(
        gp: GaussianProcessRegressor,
        X_pool: np.ndarray,
        beta: float = 1.0,
        **kwargs
) -> np.ndarray:
    """
    Gaussian Process Upper Confidence Bound (GP-UCB) acquisition function.

    Selects the point that maximizes the upper confidence bound, balancing
    exploitation (high mean) and exploration (high uncertainty).

    Formula: UCB(x) = mu(x) + beta * std(x)

    Args:
        gp: Trained Gaussian Process model.
        X_pool: Pool of candidate points to select from.
        beta: Controls the exploration-exploitation trade-off. A common
              value is 1.96 for the 95% confidence interval.
        **kwargs: (Allows for other parameters to be passed without error).

    Returns:
        Acquisition function values for each candidate point.
    """
    mean, std = gp.predict(X_pool, return_std=True)
    beta = np.sqrt(0.1)
    return mean + beta * std


def variance_reduction(
        gp: GaussianProcessRegressor,
        X_pool: np.ndarray,
        X_test: np.ndarray,
        **kwargs
) -> np.ndarray:
    """
    Variance reduction acquisition function: selects the point that would
    maximize the reduction in variance across the test set.

    This function computes the expected reduction in the global (integrated)
    posterior variance when adding a candidate point to the training set.
    It uses the GP posterior covariance:

        Cov_post(x', x) = k(x', x) - k(x', X_train) @ K_inv @ k(X_train, x)

    and approximates the integration over X_test by a discrete sum.

    Args:
        gp: Trained Gaussian Process model (must have been fitted; uses gp.X_train_).
        X_pool: Pool of candidate points to select from, shape (n_pool, d).
        X_test: Test (or representative) points to approximate the integral over the input space.
        **kwargs: (Reserved for future extensions.)

    Returns:
        Acquisition function values for each candidate point (numpy array of shape (n_pool,)).
    """
    # Ensure training data is available from the fitted GP
    X_train = gp.X_train_

    # Compute the training kernel matrix and add noise variance (gp.alpha) on the diagonal.
    K_train = gp.kernel_(X_train, X_train)
    K_train = K_train + np.eye(K_train.shape[0]) * gp.alpha
    K_inv = np.linalg.inv(K_train)

    # Precompute kernel evaluations between test points and training points (for speed).
    K_test_train = gp.kernel_(X_test, X_train)  # shape: (n_test, n_train)

    acquisition_values = np.zeros(len(X_pool))

    for i, x_cand in enumerate(X_pool):
        x_cand_reshaped = x_cand.reshape(1, -1)

        # 1. k(X_test, x_cand)
        K_test_cand = gp.kernel_(X_test, x_cand_reshaped)  # shape: (n_test, 1)

        # 2. k(X_train, x_cand)
        K_train_cand = gp.kernel_(X_train, x_cand_reshaped)  # shape: (n_train, 1)

        # 3. posterior covariance：Cov_post(X_test, x_cand) = k(X_test, x_cand) - k(X_test, X_train) @ (K_inv @ k(X_train, x_cand))
        cov_post = K_test_cand - K_test_train @ (K_inv @ K_train_cand)  # shape: (n_test, 1)

        # 4. sum over cov_post^2
        numerator = np.sum(cov_post ** 2)

        _, std_x_cand = gp.predict(x_cand_reshaped, return_std=True)
        predictive_variance = std_x_cand[0] ** 2

        if predictive_variance < 1e-10:
            predictive_variance = 1e-10

        acquisition_values[i] = numerator / predictive_variance

    return acquisition_values


def epig(
    gp: GaussianProcessRegressor,
    X_pool: np.ndarray,
    X_test: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    EPIG (Expected Predictive Information Gain) acquisition function for GP regression.
    Selects the point x in X_pool that maximizes the expected reduction in predictive
    variance (entropy) on the test set X_test.

    Formula (latent function perspective):
        EPIG(x) = (1/2) * (1 / n_test) * sum over x* in X_test of:
            log( sigma^2_{t-1}(x*)
                 / [ sigma^2_{t-1}(x*) - Cov_{t-1}(x*, x)^2 / ( sigma^2_{t-1}(x) + noise ) ] )

    Implementation details:
    - We use the current GP model's posterior to get:
        sigma^2_{t-1}(x*), sigma^2_{t-1}(x), Cov_{t-1}(x*, x).
    - Then compute the ratio of prior-to-posterior variance for each x*, take log, and average.
    - Return an EPIG score for each candidate in X_pool.

    Args:
        gp: Fitted GaussianProcessRegressor (must have gp.X_train_).
        X_pool: Pool of candidate points to select from, shape (n_pool, d).
        X_test: Test (or representative) points, shape (n_test, d), for approximating the EPIG expectation.
        **kwargs: Unused, reserved for future.

    Returns:
        A 1D numpy array of shape (n_pool,) with the EPIG value for each candidate.
    """
    # -- 1. Gather training data from the GP
    X_train = gp.X_train_
    alpha = gp.alpha  # Noise term in the GP

    # -- 2. Build the kernel matrix on the training set + noise on diagonal
    K_train = gp.kernel_(X_train, X_train)
    K_train[np.diag_indices_from(K_train)] += alpha
    K_inv = np.linalg.inv(K_train)

    # -- 3. Precompute needed kernel blocks for X_test
    # Posterior variance of f(x*) = k(x*, x*) - k(x*, X_train) K_inv k(X_train, x*)
    # We'll do it in a vectorized manner
    K_test_test = gp.kernel_(X_test, X_test)        # shape (n_test, n_test)
    K_test_train = gp.kernel_(X_test, X_train)      # shape (n_test, n_train)

    # Diagonal of the prior (posterior) var for X_test (latent function, no noise added yet)
    # var_test[i] = k(x*_i, x*_i) - k(x*_i, X_train) K_inv k(X_train, x*_i)
    # We'll compute this by columns, then just keep the diagonal.
    tmp = np.dot(K_test_train, K_inv)               # shape (n_test, n_train)
    # shape (n_test,) each is k(x*_i, x*_i) - row_i dot k(X_train, x*_i)
    var_test = np.diag(K_test_test) - np.sum(tmp * K_test_train, axis=1)

    # -- 4. For each candidate x_cand in X_pool, compute EPIG
    acquisition_values = np.zeros(len(X_pool))

    for i, x_cand in enumerate(X_pool):
        x_cand_reshaped = x_cand.reshape(1, -1)

        # (a) Posterior var at x_cand (latent function)
        #     var_cand = k(x_cand, x_cand) - k(x_cand, X_train) K_inv k(X_train, x_cand)
        K_cand_cand = gp.kernel_(x_cand_reshaped, x_cand_reshaped)[0, 0]
        K_train_cand = gp.kernel_(X_train, x_cand_reshaped)       # shape (n_train, 1)
        tmp_cand = np.dot(K_inv, K_train_cand)
        var_cand_latent = K_cand_cand - (K_train_cand.T @ tmp_cand)[0, 0]

        # (b) Cov_{t-1}[f(x*), f(x_cand)] for each x*
        #     cov_x_test_cand[j] = k(x*_j, x_cand) - k(x*_j, X_train) K_inv k(X_train, x_cand)
        K_test_cand = gp.kernel_(X_test, x_cand_reshaped)  # shape (n_test, 1)
        cov_x_test_cand = K_test_cand - (tmp @ K_train_cand)
        cov_sq = cov_x_test_cand.ravel() ** 2  # shape (n_test,)

        pred_var_cand = var_cand_latent + alpha
        pred_var_test = var_test + alpha
        numerator = pred_var_cand * pred_var_test
        denominator = numerator - cov_sq

        denominator = np.maximum(denominator, 1e-15)
        numerator = np.maximum(numerator, 1e-15)

        ratio = numerator / denominator
        ratio = np.maximum(ratio, 1.0)
        log_ratio = np.log(ratio)

        # (d) EPIG(x_cand) = (1/2) * average_j [ log_ratio_j ]
        epig_value = 0.5 * np.mean(log_ratio)
        acquisition_values[i] = epig_value

    return acquisition_values


def bald_sampling(
        gp: GaussianProcessRegressor,
        X_pool: np.ndarray,
        **kwargs
) -> np.ndarray:
    """
    Bayesian Active Learning by Disagreement (BALD)
    Approximation using predictive entropy

    Args:
        gp: Trained Gaussian Process model
        X_pool: Pool of candidate points to select from

    Returns:
        Acquisition function values for each candidate point
    """
    _, std = gp.predict(X_pool, return_std=True)
    noise_var = gp.alpha

    # For GPs: EIG = 0.5 * log(1 + σ²_x/σ²_noise)
    acquisition_values = 0.5 * np.log(1 + np.square(std) / noise_var)
    return acquisition_values


def random_sampling(
        gp: GaussianProcessRegressor,
        X_pool: np.ndarray,
        **kwargs
) -> np.ndarray:
    """
    Random sampling acquisition function (baseline)

    Args:
        gp: Trained Gaussian Process model
        X_pool: Pool of candidate points to select from

    Returns:
        Random acquisition function values for each candidate point
    """
    return np.random.rand(len(X_pool))


def visualize_active_learning_process_2d(
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_selected: List[np.ndarray],
        y_selected: List[np.ndarray],
        X_target: np.ndarray,
        y_target: np.ndarray,
        models: List[GaussianProcessRegressor],
        acquisition_function: str,
        n_iterations: int,
        figsize: Tuple[int, int] = (18, 12)
) -> None:
    """
    Visualize the active learning process across iterations for 2D input data

    Args:
        X_context: Initial context points, shape (n_context, 2)
        y_context: Initial context values, shape (n_context,)
        X_selected: List of selected points at each iteration, each of shape (1, 2)
        y_selected: List of selected values at each iteration, each of shape (1,)
        X_target: Target/test points, shape (n_target, 2)
        y_target: Target/test values, shape (n_target,)
        models: List of GP models at each iteration
        acquisition_function: Name of the acquisition function used
        n_iterations: Number of iterations to visualize
        figsize: Figure size
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scipy.interpolate import griddata
    from mpl_toolkits.mplot3d import Axes3D

    # Determine the number of rows and columns for the subplot grid
    n_cols = min(4, n_iterations)
    n_rows = (min(n_iterations, len(models)) + n_cols - 1) // n_cols

    fig = plt.figure(figsize=figsize)

    # Determine the range for x and y axes
    x1_min, x2_min = np.min(X_target, axis=0)
    x1_max, x2_max = np.max(X_target, axis=0)

    # Add some padding
    padding = 0.05 * max(x1_max - x1_min, x2_max - x2_min)
    x1_min -= padding
    x1_max += padding
    x2_min -= padding
    x2_max += padding

    # Create a grid for predictions
    grid_size = 50
    x1_grid = np.linspace(x1_min, x1_max, grid_size)
    x2_grid = np.linspace(x2_min, x2_max, grid_size)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    # Plot for each iteration
    for i in range(min(n_iterations, len(models))):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')

        # Get the model for this iteration
        gp = models[i]

        # Predict across the grid
        try:
            y_mean, y_std = gp.predict(X_grid, return_std=True)

            # Reshape for plotting
            Z_mean = y_mean.reshape(X1_grid.shape)
            Z_std = y_std.reshape(X1_grid.shape)

            # Plot the surface
            surf = ax.plot_surface(X1_grid, X2_grid, Z_mean, cmap=cm.viridis,
                                   alpha=0.8, linewidth=0, antialiased=True)

            # Add color bar
            cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.05)
            cbar.set_label('Predicted Mean')

            # Calculate MSE if we have target data
            if y_target is not None and len(y_target) > 0:
                target_preds = gp.predict(X_target)
                mse = np.mean((target_preds - y_target) ** 2)
                rmse = np.sqrt(mse)
            else:
                rmse = None

        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            # Fall back to scatter plot if prediction fails
            ax.scatter(X_target[:, 0], X_target[:, 1], y_target, alpha=0.1, color='gray')
            rmse = None

        # Plot known points (initial context)
        if len(X_context) > 0:
            ax.scatter(X_context[:, 0], X_context[:, 1], y_context,
                       color='black', marker='x', s=50, label='Initial Points')

        # Plot selected points up to this iteration
        if i > 0 and len(X_selected) > 0:
            # Combine all previously selected points
            prev_X = np.vstack(X_selected[:i])
            prev_y = np.hstack(y_selected[:i])

            ax.scatter(prev_X[:, 0], prev_X[:, 1], prev_y,
                       color='red', marker='o', s=50, label='Selected Points')

            # Highlight the most recent selected point
            ax.scatter(X_selected[i - 1][0], X_selected[i - 1][1], y_selected[i - 1],
                       color='red', marker='o', s=100, edgecolor='black', linewidth=2)

        # Set title and labels
        # if i < len(X_selected):
        #     if rmse is not None:
        #         ax.set_title(
        #             f'Iteration {i + 1}: RMSE={rmse:.4f}\nx=({X_selected[i][0, 0]:.2f}, {X_selected[i][0, 1]:.2f}), y={y_selected[i][0]:.2f}')
        #     else:
        #         ax.set_title(
        #             f'Iteration {i + 1}: x=({X_selected[i][0, 0]:.2f}, {X_selected[i][0, 1]:.2f}), y={y_selected[i][0]:.2f}')
        # else:
        #     if rmse is not None:
        #         ax.set_title(f'Iteration {i + 1}: RMSE={rmse:.4f}')
        #     else:
        #         ax.set_title(f'Iteration {i + 1}')

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')

        # Set consistent view angle
        ax.view_init(elev=30, azim=45)

        # Add legend only to the first plot
        if i == 0:
            ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.suptitle(f'Active Learning with {acquisition_function.replace("_", " ").title()}', fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()


def visualize_active_learning_process(
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_selected: List[np.ndarray],
        y_selected: List[np.ndarray],
        X_target: np.ndarray,
        y_target: np.ndarray,
        models: List[GaussianProcessRegressor],
        acquisition_function: str,
        n_iterations: int,
        figsize: Tuple[int, int] = (25, 20)
) -> None:
    """
    Visualize the active learning process across iterations for 1D or 2D input data

    Args:
        X_context: Initial context points, shape (n_context, dim)
        y_context: Initial context values, shape (n_context,)
        X_selected: List of selected points at each iteration
        y_selected: List of selected values at each iteration
        X_target: Target/test points, shape (n_target, dim)
        y_target: Target/test values, shape (n_target,)
        models: List of GP models at each iteration
        acquisition_function: Name of the acquisition function used
        n_iterations: Number of iterations to visualize
        figsize: Figure size
    """
    # Check input dimension
    input_dim = X_context.shape[1] if len(X_context) > 0 else X_target.shape[1]

    if input_dim == 1:
        # Original 1D visualization
        plt.figure(figsize=figsize)

        # Determine the range for x-axis (for plotting)
        x_min = min(np.min(X_target), np.min(X_context))
        x_max = max(np.max(X_target), np.max(X_context))
        x_grid = np.linspace(x_min, x_max, 100).reshape(-1, 1)

        # Plot for each iteration
        for i in range(min(n_iterations, len(models))):
            plt.subplot(5, 6, i + 1)

            # Get the model for this iteration
            gp = models[i]

            # Predict across the range
            y_mean, y_std = gp.predict(x_grid, return_std=True)

            # Plot prediction and uncertainty
            plt.plot(x_grid, y_mean, 'b-', label='Predicted Mean' if i == 0 else "")
            plt.fill_between(
                x_grid.ravel(),
                y_mean - 2 * y_std,
                y_mean + 2 * y_std,
                alpha=0.2, color='b',
                label='95% CI' if i == 0 else ""
            )

            # Plot known points (initial context)
            plt.plot(X_context, y_context, 'kx', label='Initial Points' if i == 0 else "")

            # Plot selected points up to this iteration
            if i > 0:
                # All previously selected points
                plt.plot(
                    np.vstack(X_selected[:i]),
                    np.hstack(y_selected[:i]),
                    'ro',
                    label='Selected Points' if i == 1 else ""
                )

                # Most recent selected point
                plt.plot(X_selected[i - 1], y_selected[i - 1], 'ro', markersize=10, markeredgecolor='k')

            # Plot test/target set
            plt.plot(X_target, y_target, 'g.', alpha=0.5, label='Test Set' if i == 0 else "")

            if i < len(X_selected):
                plt.title(f'Iteration {i + 1}: x={X_selected[i][0]:.2f}, y={y_selected[i]:.2f}')
            else:
                plt.title(f'Iteration {i + 1}')

            if i == 0:
                plt.legend(loc='best')

            plt.ylim([np.min(y_target) - 0.5, np.max(y_target) + 0.5])

        plt.tight_layout()
        plt.suptitle(f'Active Learning with {acquisition_function.replace("_", " ").title()}', fontsize=16)
        plt.subplots_adjust(top=0.92)
        plt.show()

    elif input_dim == 2:
        # Call the 2D visualization function
        visualize_active_learning_process_2d(
            X_context, y_context, X_selected, y_selected,
            X_target, y_target, models, acquisition_function,
            n_iterations, figsize
        )
    else:
        raise ValueError(f"Visualization not supported for {input_dim}D input data")


def visualize_final_model(
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_selected: List[np.ndarray],
        y_selected: List[np.ndarray],
        X_target: np.ndarray,
        y_target: np.ndarray,
        final_model: GaussianProcessRegressor,
        acquisition_function: str,
        mse: float,
        figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Visualize the final model after active learning

    Args:
        X_context: Initial context points
        y_context: Initial context values
        X_selected: List of all selected points
        y_selected: List of all selected values
        X_target: Target/test points
        y_target: Target/test values
        final_model: Final GP model
        acquisition_function: Name of the acquisition function used
        mse: Mean squared error on the test set
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Create fine grid for visualization
    x_min = min(np.min(X_target), np.min(X_context))
    x_max = max(np.max(X_target), np.max(X_context))
    x_grid = np.linspace(x_min, x_max, 100).reshape(-1, 1)

    # Predict across the range
    y_mean, y_std = final_model.predict(x_grid, return_std=True)

    # Plot prediction and uncertainty
    plt.plot(x_grid, y_mean, 'b-', label='Final Prediction')
    plt.fill_between(
        x_grid.ravel(),
        y_mean - 2 * y_std,
        y_mean + 2 * y_std,
        alpha=0.2, color='b',
        label='95% Confidence Interval'
    )

    # Plot initial points
    plt.plot(X_context, y_context, 'kx', label='Initial Known Points')

    # Plot all selected points
    if X_selected:
        plt.plot(
            np.vstack(X_selected),
            np.hstack(y_selected),
            'ro',
            label='Selected Points'
        )

    # Plot test set (targets)
    plt.plot(X_target, y_target, 'g.', alpha=0.5, label='Test Set')

    plt.title(f'Final Model After {acquisition_function.replace("_", " ").title()} (MSE: {mse:.4f})')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def active_learning_with_gp(
        batch: AttrDict,
        acquisition_function: str = 'uncertainty',
        n_iterations: int = 30,
        visualize: bool = True,
        kernel: Optional[RBF] = None,
        alpha: float = 1e-2,
        n_restarts_optimizer: int = 5
) -> Tuple[np.array, np.array]:
    """
    Active learning with Gaussian Process using different acquisition functions

    Args:
        batch: Dictionary/structure containing context_x, context_y, query_x, query_y, target_x, target_y
             (each with shape [batch_size, n_points, dim])
        acquisition_function: Strategy for selecting points. Options:
                             'uncertainty': Uncertainty sampling (highest variance)
                             'variance_reduction': Expected variance reduction
                             'bald': Bayesian Active Learning by Disagreement
                             'random': Random selection (baseline)
        n_iterations: Number of iterations (points to select)
        visualize: Whether to visualize the learning process (only for first batch element)
        kernel: GP kernel to use (if None, a default RBF kernel will be used)
        alpha: Noise level
        n_restarts_optimizer: Number of restarts for GP hyperparameter optimization

    Returns:
        avg_log_probs: List of average log probabilities at each iteration
        avg_rmse: List of root mean squared errors at each iteration
    """
    # Define acquisition functions mapping
    acquisition_functions = {
        'uncertainty': uncertainty_sampling,
        'variance_reduction': variance_reduction,
        'bald': bald_sampling,
        'random': random_sampling,
        'epig': epig,
        'ucb': gp_ucb,
    }

    # Get the requested acquisition function
    if acquisition_function not in acquisition_functions:
        raise ValueError(f"Unknown acquisition function: {acquisition_function}. "
                         f"Available options: {list(acquisition_functions.keys())}")

    acquisition_func = acquisition_functions[acquisition_function]

    # Convert batch attributes to numpy arrays
    context_x = batch.context_x.cpu().numpy() if hasattr(batch.context_x, 'cpu') else batch.context_x
    context_y = batch.context_y.cpu().numpy() if hasattr(batch.context_y, 'cpu') else batch.context_y
    query_x = batch.query_x.cpu().numpy() if hasattr(batch.query_x, 'cpu') else batch.query_x
    query_y = batch.query_y.cpu().numpy() if hasattr(batch.query_y, 'cpu') else batch.query_y
    target_x = batch.target_x.cpu().numpy() if hasattr(batch.target_x, 'cpu') else batch.target_x
    target_y = batch.target_y.cpu().numpy() if hasattr(batch.target_y, 'cpu') else batch.target_y

    # Get batch size and prepare arrays for metrics
    batch_size = context_x.shape[0]
    all_log_probs = np.zeros((n_iterations, batch_size))
    all_rmse = np.zeros((n_iterations, batch_size))
    all_time = np.zeros((n_iterations, batch_size))


    # For visualization (using only the first batch element)
    if visualize:
        all_sampled_x = []
        all_sampled_y = []
        all_models = []

    # Process each batch element separately
    for b in range(batch_size):
        # Extract data for this batch element
        b_context_x = context_x[b]
        b_context_y = context_y[b]
        b_query_x = query_x[b]
        b_query_y = query_y[b]
        b_target_x = target_x[b]
        b_target_y = target_y[b]

        # Ensure proper dimensions for inputs
        if b_context_x.shape[-1] == 1:
            b_context_x = b_context_x.reshape(-1, 1)
        if b_context_y.shape[-1] == 1:
            b_context_y = b_context_y.reshape(-1)
        if b_query_x.shape[-1] == 1:
            b_query_x = b_query_x.reshape(-1, 1)
        if b_query_y.shape[-1] == 1:
            b_query_y = b_query_y.reshape(-1)
        if b_target_x.shape[-1] == 1:
            b_target_x = b_target_x.reshape(-1, 1)
        if b_target_y.shape[-1] == 1:
            b_target_y = b_target_y.reshape(-1)

        # Current training data for the model
        X_train = b_context_x.copy()
        y_train = b_context_y.copy()

        # Current candidate points in the pool
        X_pool = b_query_x.copy()
        y_pool = b_query_y.copy()

        # Iterate through the active learning loop
        for i in range(n_iterations):
            start_time = time.time()

            kernel = C(0.5, (0.1, 2.0)) * RBF(0.5, (0.1, 3))
                      # + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-3, 1e-1)))
            # Define and fit GP model
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restarts_optimizer,
                alpha=1e-4,
                random_state=1  # For reproducibility
            )

            # Train the model
            gp.fit(X_train, y_train)

            # Store model for visualization (only for first batch)
            if visualize and b == 0:
                all_models.append(gp)

            # Calculate metrics on target set
            y_pred, y_std = gp.predict(b_target_x, return_std=True)

            # Calculate RMSE
            rmse = np.sqrt(np.mean((y_pred - b_target_y) ** 2))
            all_rmse[i, b] = rmse

            # Compute log probability using Gaussian likelihood with numerical stability
            # Clip std to prevent division by zero
            y_std = np.maximum(y_std, 1e-6)
            # Clip squared errors to prevent extremely negative values
            sq_err = np.minimum(((b_target_y - y_pred) / y_std) ** 2, 100)
            log_prob = np.mean(-0.5 * np.log(2 * np.pi) - np.log(y_std) - 0.5 * sq_err)
            all_log_probs[i, b] = log_prob

            # Calculate acquisition function values
            if acquisition_function in ['epig','variance_reduction']:
                # For variance reduction, we need the test points as well
                acquisition_values = acquisition_func(gp, X_pool, X_test=b_target_x)
                # print(acquisition_values)
            else:
                acquisition_values = acquisition_func(gp, X_pool)

            # Select the point with maximum acquisition value
            next_idx = np.argmax(acquisition_values)
            next_x = X_pool[next_idx].reshape(1, -1)  # Keep 2D shape for consistency
            next_y = y_pool[next_idx]

            end_time = time.time()
            all_time[i, b] = end_time - start_time

            # Record sampling history (only for first batch)
            if visualize and b == 0:
                all_sampled_x.append(next_x.reshape(-1))
                all_sampled_y.append(next_y)

            # Add the selected point to the training set
            X_train = np.vstack((X_train, next_x))
            y_train = np.append(y_train, next_y)

            # Remove the selected point from the pool
            X_pool = np.delete(X_pool, next_idx, axis=0)
            y_pool = np.delete(y_pool, next_idx)



    # Average across batch dimension to get final metrics for each iteration
    avg_log_probs = np.mean(all_log_probs, axis=1).tolist()
    avg_rmse = np.mean(all_rmse, axis=1).tolist()

    # Visualize process for the first batch element if requested
    if visualize and len(all_sampled_x) > 0:
        # Process visualization
        b_context_x = context_x[0]
        b_context_y = context_y[0]
        b_target_x = target_x[0]
        b_target_y = target_y[0]

        # Ensure proper dimensions
        if b_context_x.shape[-1] == 1:
            b_context_x = b_context_x.reshape(-1, 1)
        if b_context_y.shape[-1] == 1:
            b_context_y = b_context_y.reshape(-1)
        if b_target_x.shape[-1] == 1:
            b_target_x = b_target_x.reshape(-1, 1)
        if b_target_y.shape[-1] == 1:
            b_target_y = b_target_y.reshape(-1)

        visualize_active_learning_process(
            b_context_x, b_context_y,
            all_sampled_x, all_sampled_y,
            b_target_x, b_target_y,
            all_models,
            acquisition_function,
            n_iterations
        )

        # Plot metrics
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, n_iterations + 1), avg_log_probs)
        plt.xlabel('Iteration')
        plt.ylabel('Average Log Probability')
        plt.title(f'Log Probability vs. Iteration ({acquisition_function})')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, n_iterations + 1), avg_rmse)
        plt.xlabel('Iteration')
        plt.ylabel('Average RMSE')
        plt.title(f'RMSE vs. Iteration ({acquisition_function})')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    return all_log_probs, all_rmse
    # return all_log_probs, all_rmse, all_time


def compare_acquisition_methods(
        batch: Dict,
        n_iterations: int = 15,
        visualize_individual: bool = False,
        visualize_comparison: bool = True,
        kernel: Optional[RBF] = None,
        alpha: float = 1e-4
) -> Dict:
    """
    Compare different acquisition methods on the same dataset

    Args:
        batch: Dictionary/structure containing context_x, context_y, query_x, query_y, target_x, target_y
        n_iterations: Number of iterations for each method
        visualize_individual: Whether to visualize individual learning processes
        visualize_comparison: Whether to visualize comparison of methods
        kernel: GP kernel to use (if None, a default RBF kernel will be used)
        alpha: Noise level for GP

    Returns:
        Dictionary containing results for each method
    """
    methods = ['uncertainty', 'variance_reduction', 'bald', 'random']
    results = {}

    for method in methods:
        print(f"\nRunning active learning with {method}...")
        avg_log_probs, avg_rmse = active_learning_with_gp(
            batch,
            acquisition_function=method,
            n_iterations=n_iterations,
            visualize=visualize_individual,
            kernel=kernel,
            alpha=alpha
        )

        results[method] = {
            'log_probs': avg_log_probs,
            'rmse': avg_rmse
        }

    # Visualize comparison of methods if requested
    if visualize_comparison:
        # Plot log probabilities and RMSE for all methods
        plt.figure(figsize=(12, 10))

        # Plot log probabilities
        plt.subplot(2, 1, 1)
        for method in methods:
            plt.plot(range(1, n_iterations + 1), results[method]['log_probs'],
                     label=method.replace('_', ' ').title(), linewidth=2)

        plt.xlabel('Iteration')
        plt.ylabel('Average Log Probability')
        plt.title('Log Probability vs. Iteration by Acquisition Method')
        plt.legend()
        plt.grid(True)

        # Plot RMSE
        plt.subplot(2, 1, 2)
        for method in methods:
            plt.plot(range(1, n_iterations + 1), results[method]['rmse'],
                     label=method.replace('_', ' ').title(), linewidth=2)

        plt.xlabel('Iteration')
        plt.ylabel('Average RMSE')
        plt.title('RMSE vs. Iteration by Acquisition Method')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Print final metrics
        print("\nFinal Metrics (after all iterations):")
        print(f"{'Method':<20} {'Final Log Prob':<15} {'Final RMSE':<10}")
        print("-" * 50)
        for method in methods:
            print(f"{method:<20} {results[method]['log_probs'][-1]:<15.4f} {results[method]['rmse'][-1]:<10.4f}")

    return results


if __name__ == "__main__":
    # Example usage with a simple toy dataset
    from sklearn.datasets import make_friedman1

    # Create synthetic data
    X, y = make_friedman1(n_samples=200, n_features=1, noise=0.1, random_state=42)


    # Split into context, query, and target
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self


    # Create multiple batch elements for testing
    batch_size = 10

    # Lists to store data for each batch element
    all_context_x = []
    all_context_y = []
    all_query_x = []
    all_query_y = []
    all_target_x = []
    all_target_y = []

    for b in range(batch_size):
        # Use different random seed for each batch element
        seed = 42 + b
        np.random.seed(seed)

        # Use only 5 points as initial context
        context_indices = np.random.choice(X.shape[0], 5, replace=False)
        context_x = X[context_indices]
        context_y = y[context_indices]

        # Use 100 points as query pool
        remaining_indices = np.setdiff1d(np.arange(X.shape[0]), context_indices)
        query_indices = np.random.choice(remaining_indices, 100, replace=False)
        query_x = X[query_indices]
        query_y = y[query_indices]

        # Use the rest as target
        target_indices = np.setdiff1d(np.arange(X.shape[0]), np.concatenate([context_indices, query_indices]))
        target_x = X[target_indices]
        target_y = y[target_indices]

        # Append to lists
        all_context_x.append(context_x)
        all_context_y.append(context_y)
        all_query_x.append(query_x)
        all_query_y.append(query_y)
        all_target_x.append(target_x)
        all_target_y.append(target_y)

    # Create batch with all elements
    batch = AttrDict({
        'context_x': np.array(all_context_x),
        'context_y': np.array(all_context_y),
        'query_x': np.array(all_query_x),
        'query_y': np.array(all_query_y),
        'target_x': np.array(all_target_x),
        'target_y': np.array(all_target_y)
    })

    # Run active learning with uncertainty sampling
    avg_log_probs, avg_rmse = active_learning_with_gp(
        batch,
        acquisition_function='uncertainty',
        n_iterations=15,
        visualize=True
    )

    print(f"Final average log probability: {avg_log_probs[-1]:.4f}")
    print(f"Final average RMSE: {avg_rmse[-1]:.4f}")

    # Compare all methods
    results = compare_acquisition_methods(
        batch,
        n_iterations=15,
        visualize_individual=False,
        visualize_comparison=True
    )