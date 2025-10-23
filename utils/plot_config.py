import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import torch
import os
import copy
from utils import compute_ll, create_target_mask, select_targets_by_mask
from utils.misc import *
from sklearn.metrics import mean_squared_error


def apply_style(use_tex=True):
    plt.rcParams.update({
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
        'text.usetex': use_tex
    })


def plot_metrics_with_confidence(
        metrics_data_list,
        metric_names,
        x_range=None,
        colors=None,
        markers=None,
        title="",
        xlabel="Number of Steps",
        ylabel="RMSE",
        figsize=(5, 4),
        marker_frequency=5,
        legend_loc='upper right'
):
    # Ensure all input data is numpy arrays
    metrics_data_list = [
        data.detach().numpy() if torch.is_tensor(data) else data
        for data in metrics_data_list
    ]

    # Get number of time steps
    T = metrics_data_list[0].shape[0]

    # Set default x_range if not provided
    if x_range is None:
        x_range = (1, T)

    # Create x-axis array
    x = np.arange(x_range[0], x_range[1] + 1)

    # Set default colors if not provided
    if colors is None:
        colors = ['#8172b3', '#937860', '#4c72b0', '#dd8452', '#55a868', '#c44e52', ]

    # Set default markers if not provided
    if markers is None:
        markers = ['o', 's', '^', 'D', 'X', 'P']

    # Set style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("colorblind")
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Helper function to calculate confidence intervals
    def get_confidence_interval(data):
        n_trials = data.shape[1]
        std = data.std(axis=1)
        ci = 1.96 * std / np.sqrt(n_trials)  # 1.96 for 95% CI under normal approximation
        return ci

    # Helper function to plot with confidence bands
    def plot_with_confidence(ax, x, mean, ci, label, color, marker):
        ax.plot(x, mean, label=label, color=color, linewidth=2,
                marker=marker, markevery=marker_frequency, markersize=7, markeredgecolor='white')
        ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15)

    # Plot each metric
    for i, (data, label) in enumerate(zip(metrics_data_list, metric_names)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        mean = data.mean(axis=1)
        ci = get_confidence_interval(data)

        # Ensure data length matches x-axis
        if mean.shape[0] < len(x):
            # Pad with NaN or truncate as needed
            padded_mean = np.full(len(x), np.nan)
            padded_mean[:mean.shape[0]] = mean

            padded_ci = np.full(len(x), np.nan)
            padded_ci[:ci.shape[0]] = ci

            plot_with_confidence(ax, x, padded_mean, padded_ci, label, color, marker)
        elif mean.shape[0] > len(x):
            # Truncate to match x length
            plot_with_confidence(ax, x, mean[:len(x)], ci[:len(x)], label, color, marker)
        else:
            plot_with_confidence(ax, x, mean, ci, label, color, marker)

    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    # Add grid with lower opacity
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend with a nice border
    legend = ax.legend(frameon=True, framealpha=1, edgecolor='gray',
                       fontsize=10, loc=legend_loc)
    legend.get_frame().set_linewidth(0.5)

    # Set axis limits
    # ax.set_ylim(bottom=0)
    ax.set_xlim(x_range[0], x_range[1])

    # Add minor ticks
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Adjust layout
    plt.tight_layout()

    return fig, ax



def calculate_targeted_log_prob_and_save_plots(eval_set, cfg, model, experiment, T, acquisition="aae", attend_to="data",
                                               visualize=True, save_path="."):
    """
    Visualize model predictions over multiple time steps, saving each step as a separate image.

    Args:
        eval_set: The evaluation dataset.
        cfg: Configuration object.
        model: The trained model.
        T: Total number of time steps to run.
        acquisition: Acquisition strategy ("aae", "random", or "uncertainty_sampling").
        attend_to: Which targets to attend to ("data" or "theta").
        visualize: Whether to generate and save visualizations.
        save_path: Directory to save the generated plots.
    """
    if visualize and not os.path.exists(save_path):
        os.makedirs(save_path)

    batch = copy.deepcopy(eval_set)
    batch_size = batch.context_x.shape[0]
    mask_type = cfg.task.mask_type[0]

    target_mask = create_target_mask(
        mask_type,
        cfg.task.embedding_type,
        cfg.task.n_target_data,
        cfg.task.n_target_theta,
        cfg.task.n_selected_targets,
        cfg.task.predefined_masks,
        None,
        cfg.task.mask_index,
        attend_to,
    )
    if acquisition != "aae":
        batch.target_mask = None
    else:
        batch.target_mask = target_mask

    log_probs = torch.zeros((T, batch_size), device=batch.context_x.device)
    rmse_values = torch.zeros((T, batch_size), device=batch.context_x.device)
    all_time = torch.zeros((T), device=batch.context_x.device)

    batch_idx = 0  # Work with the first batch for visualization

    for t in range(T):
        if cfg.time_token:
            batch.t = torch.tensor([t / T])

        outs = model.forward(batch)
        design_out = outs.design_out
        posterior_out = outs.posterior_out
        posterior_out_query = outs.posterior_out_query

        target_ll = compute_ll(
            batch.target_all,
            posterior_out.mixture_means,
            posterior_out.mixture_stds,
            posterior_out.mixture_weights
        )

        weighted_means = torch.sum(posterior_out.mixture_means * posterior_out.mixture_weights, dim=-1)
        squared_errors = (batch.target_all.squeeze(-1) - weighted_means) ** 2

        if mask_type == "none":
            masked_target_ll = target_ll
            masked_target_rmse = torch.sqrt(torch.mean(squared_errors, dim=-1))
        else:
            masked_target_ll = select_targets_by_mask(target_ll, target_mask)
            masked_squared_errors = select_targets_by_mask(squared_errors, target_mask)
            masked_target_rmse = torch.sqrt(torch.mean(masked_squared_errors, dim=-1))

        log_probs[t] = masked_target_ll.mean(dim=-1)
        rmse_values[t] = masked_target_rmse

        if acquisition == "aae":
            index = design_out.idx
        elif acquisition == "random":
            query_size = batch.query_x.shape[1]
            index = torch.randint(0, query_size, (batch_size, 1), device=batch.query_x.device)
        elif acquisition == "uncertainty_sampling":
            uncertainties = calculate_gmm_variance(
                posterior_out_query.mixture_means,
                posterior_out_query.mixture_stds,
                posterior_out_query.mixture_weights
            )
            index = torch.argmax(uncertainties, dim=1, keepdim=True)
        else:
            raise NotImplementedError


        next_design_x = torch.gather(
            batch.query_x, 1,
            index.unsqueeze(2).expand(batch_size, 1, cfg.task.dim_x)
        )
        batch = experiment.update_batch(batch, index)

        if visualize:
            fig, ax = plt.subplots(figsize=(8, 6))

            x_values = batch.target_x[batch_idx].detach().cpu()
            y_values = batch.target_y[batch_idx].detach().cpu()
            means = posterior_out.mixture_means[batch_idx].detach().cpu()
            stds = posterior_out.mixture_stds[batch_idx].detach().cpu()
            weights = posterior_out.mixture_weights[batch_idx].detach().cpu()

            all_x, all_means, all_lower, all_upper = [], [], [], []
            for i in range(x_values.shape[0]):
                x_val = x_values[i, 0].item()
                component_means = means[i].numpy()
                component_stds = stds[i].numpy()
                component_weights = weights[i].numpy()
                weighted_mean = np.sum(component_weights * component_means)
                weighted_variance = np.sum(
                    component_weights * (component_stds ** 2 + (component_means - weighted_mean) ** 2))
                weighted_std = np.sqrt(weighted_variance)
                all_x.append(x_val)
                all_means.append(weighted_mean)
                all_lower.append(weighted_mean - 2 * weighted_std)
                all_upper.append(weighted_mean + 2 * weighted_std)

            all_x, all_means, all_lower, all_upper = np.array(all_x), np.array(all_means), np.array(
                all_lower), np.array(all_upper)
            sort_indices = np.argsort(all_x)
            all_x, all_means, all_lower, all_upper = all_x[sort_indices], all_means[sort_indices], all_lower[
                sort_indices], all_upper[sort_indices]
            all_gt = y_values.numpy().reshape(-1)[sort_indices]
            mse = mean_squared_error(all_gt, all_means)

            ax.plot(all_x, all_means, 'C0', label='Prediction')
            ax.fill_between(all_x, all_lower, all_upper, color='C0', alpha=0.2)
            ax.plot(all_x, all_gt, 'C3', label='Ground Truth')
            ax.scatter(all_x, all_gt, color='black', s=10, label='Targets')
            context_x = batch.context_x[batch_idx].detach().cpu().numpy()
            context_y = batch.context_y[batch_idx].detach().cpu().numpy()
            ax.scatter(context_x, context_y, color='C2', s=30, marker='o', label='Context')
            next_x = next_design_x[batch_idx].detach().cpu().numpy()
            ax.axvline(x=next_x, color="r", linestyle="--", linewidth=1.5, label="Next Query")

            ax.set_title(f'Step {t + 1}, RMSE = {mse:.4f}', fontsize=16)
            ax.set_xlabel('x', fontsize=16)
            ax.set_ylabel('y', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.5, 2)

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(labels),
                       fontsize='small')

            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.show()
            plt.savefig(os.path.join(save_path, f"step_{t+1:03d}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

    return log_probs, rmse_values