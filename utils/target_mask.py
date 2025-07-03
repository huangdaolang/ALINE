import torch
import random


def create_target_mask(mask_type,
                       embedding_type,
                       n_target_data,
                       n_target_theta,
                       n_selected_targets,
                       predefined_masks,
                       predefined_mask_weights,
                       mask_index,
                       attend_to):
    """
    Create a target mask based on configuration settings.

    This function creates a boolean mask of length n_target, where True values
    indicate targets that should be attended to during inference.

    Args:

        - embedding_type: 'data', 'theta', or 'mix'
        - n_target_data: Number of target data points
        - n_target_theta: Number of target theta parameters
        - mask_type: 'all', 'partial', 'none', 'predefined', or 'split'
        - n_selected_targets: For 'partial' mask_type, number of targets to select
        - predefined_masks: List of predefined masks to choose from
        - mask_index: Index of predefined mask to use (if None, randomly select)
        - attend_to: For 'split' mask_type, whether to attend to 'data' or 'theta'

    Returns:
        torch.Tensor: Boolean tensor of shape [n_target] where True values indicate
                     targets to attend to
    """

    # Total number of targets
    n_target = n_target_data + n_target_theta

    # Initialize mask
    mask = torch.zeros(n_target, dtype=torch.bool)

    # Determine mask based on type
    if mask_type == 'all':
        # Attend to all targets
        mask.fill_(True)

    elif mask_type == 'none':
        # Don't attend to any targets (ACE case)
        mask.fill_(False)

    elif mask_type == 'partial':
        # For data mode: attend to random subset of data
        if embedding_type == 'data':
            # Randomly select n_selected_targets indices
            indices = torch.randperm(n_target)[:n_selected_targets]
            mask[indices] = True

        # For theta mode: attend to random subset of theta
        elif embedding_type == 'theta':
            indices = torch.randperm(n_target)[:n_selected_targets]
            mask[indices] = True

    elif mask_type == 'predefined':
        # Use predefined mask patterns
        if mask_index is not None:
            # Use specific mask by index
            predefined_mask = predefined_masks[mask_index]
        else:
            # Weighted random selection of predefined mask
            if predefined_mask_weights is not None and len(predefined_mask_weights) == len(predefined_masks):
                # Convert weights to probabilities
                weights = torch.tensor(predefined_mask_weights, dtype=torch.float)
                probabilities = weights / weights.sum()
                # Sample according to weights
                index = torch.multinomial(probabilities, 1).item()
                predefined_mask = predefined_masks[index]
            else:
                # Uniform random selection if no weights provided
                predefined_mask = random.choice(predefined_masks)
        # Convert predefined mask to boolean tensor
        for i, should_attend in enumerate(predefined_mask):
            if i < n_target and should_attend:
                mask[i] = True

    elif mask_type == 'split':
        # For mix mode: attend to either all data or all theta
        if embedding_type == 'mix':
            # Decide whether to attend to data or theta
            if attend_to is not None:
                attend_to_data = attend_to == 'data'
            else:
                attend_to_data = random.choice([True, False])

            if attend_to_data:
                # Attend to all data points
                mask[:n_target_data] = True
            else:
                # Attend to all theta parameters
                mask[n_target_data:] = True

    return mask


def select_targets_by_mask(target_results, target_mask):
    """
    Select target results based on the target mask.

    Args:
        target_results (torch.Tensor): Tensor of shape [batch_size, n_target, ...]
        target_mask (torch.Tensor): Boolean tensor of shape [n_target]

    Returns:
        torch.Tensor: Tensor of shape [batch_size, num_selected, ...] containing only
                     the results for selected targets
    """
    # Get the indices of True values in the mask
    selected_indices = torch.where(target_mask)[0]

    # Select these indices from the target results
    selected_results = target_results[:, selected_indices]

    return selected_results


def get_masking_description(cfg):
    """
    Generate a human-readable description of the current masking configuration.
    Useful for logging and debugging.

    Args:
        cfg: Configuration object containing masking parameters

    Returns:
        str: Description of the mask configuration
    """
    if cfg.task.mask_type == 'all':
        return "Attending to all targets"
    elif cfg.task.mask_type == 'none':
        return "Not attending to any targets"
    elif cfg.task.mask_type == 'partial':
        return f"Attending to {cfg.task.n_selected_targets} randomly selected targets"
    elif cfg.task.mask_type == 'predefined':
        if hasattr(cfg.task, 'mask_index') and cfg.task.mask_index is not None:
            return f"Using predefined mask #{cfg.task.mask_index}"
        else:
            return "Using randomly selected predefined mask"
    elif cfg.task.mask_type == 'split' and cfg.task.embedding_type == 'mix':
        if hasattr(cfg.task, 'attend_to') and cfg.task.attend_to is not None:
            return f"Attending to all {cfg.task.attend_to} targets"
        else:
            return "Attending to either all data or all theta targets (random choice)"
    return "Unknown masking configuration"


def test_create_target_mask():
    """Test the create_target_mask function with various configurations."""
    from attrdictionary import AttrDict

    # Create some test configs
    test_configs = [
        # All targets mask (data mode)
        {"name": "All targets (data mode)",
         "cfg": AttrDict({"embedding_type": "data",
                          "task": AttrDict({"mask_type": "all", "n_target_data": 5, "n_target_theta": 0})}),
         "expected_mask": torch.ones(5, dtype=torch.bool),
         "expected_desc": "Attending to all targets"},

        # All targets mask (theta mode)
        {"name": "All targets (theta mode)",
         "cfg": AttrDict({"embedding_type": "theta",
                          "task": AttrDict({"mask_type": "all", "n_target_data": 0, "n_target_theta": 4})}),
         "expected_mask": torch.ones(4, dtype=torch.bool),
         "expected_desc": "Attending to all targets"},

        # All targets mask (mix mode)
        {"name": "All targets (mix mode)",
         "cfg": AttrDict({"embedding_type": "mix",
                          "task": AttrDict({"mask_type": "all", "n_target_data": 3, "n_target_theta": 2})}),
         "expected_mask": torch.ones(5, dtype=torch.bool),
         "expected_desc": "Attending to all targets"},

        # No targets mask
        {"name": "No targets",
         "cfg": AttrDict({"embedding_type": "data",
                          "task": AttrDict({"mask_type": "none", "n_target_data": 5, "n_target_theta": 0})}),
         "expected_mask": torch.zeros(5, dtype=torch.bool),
         "expected_desc": "Not attending to any targets"},

        # Partial mask
        {"name": "Partial targets",
         "cfg": AttrDict({"embedding_type": "data",
                          "task": AttrDict({"mask_type": "partial", "n_target_data": 5, "n_target_theta": 0, "n_selected_targets": 2,})}),
         "expected_count": 2,  # We can't predict which indices will be selected, but we know how many
         "expected_desc": "Attending to 2 randomly selected targets"},

        # Predefined mask
        {"name": "Predefined mask",
         "cfg": AttrDict({"embedding_type": "theta",
                          "task": AttrDict({"predefined_masks": [[True, False, True, False]], "mask_type": "predefined", "n_target_data": 0, "n_target_theta": 4, "mask_index": 0})}),
         "expected_mask": torch.tensor([True, False, True, False]),
         "expected_desc": "Using predefined mask #0"},

        # Split mask (data)
        {"name": "Split mask (data)",
         "cfg": AttrDict({"embedding_type": "mix",
                          "task": AttrDict({"mask_type": "split", "n_target_data": 3, "n_target_theta": 2, "attend_to": "data",})}),
         "expected_mask": torch.tensor([True, True, True, False, False]),
         "expected_desc": "Attending to all data targets"},

        # Split mask (theta)
        {"name": "Split mask (theta)",
         "cfg": AttrDict({"embedding_type": "mix",
                          "task": AttrDict({"mask_type": "split", "n_target_data": 3, "n_target_theta": 2, "attend_to": "theta",})}),
         "expected_mask": torch.tensor([False, False, False, True, True]),
         "expected_desc": "Attending to all theta targets"},
    ]

    # Run tests
    for test in test_configs:
        print(f"\nTesting: {test['name']}")
        mask = create_target_mask(test["cfg"])
        desc = get_masking_description(test["cfg"])

        print(f"Generated mask: {mask}")
        print(f"Description: {desc}")

        # Check mask shape
        expected_shape = test["cfg"].task.n_target_data + test["cfg"].task.n_target_theta
        assert mask.shape[0] == expected_shape, f"Mask shape mismatch. Expected {expected_shape}, got {mask.shape[0]}"

        # Check mask correctness
        if "expected_mask" in test:
            assert torch.all(
                mask == test["expected_mask"]), f"Mask content mismatch. Expected {test['expected_mask']}, got {mask}"
        elif "expected_count" in test:
            assert torch.sum(mask) == test[
                "expected_count"], f"Mask count mismatch. Expected {test['expected_count']}, got {torch.sum(mask)}"

        # Check description
        assert desc == test["expected_desc"], f"Description mismatch. Expected '{test['expected_desc']}', got '{desc}'"

    print("\nAll mask creation tests passed!")


def test_select_targets_by_mask():
    """Test the select_targets_by_mask function."""
    # Create test data
    batch_size = 2
    n_target = 5
    feature_dim = 3
    target_results = torch.randn(batch_size, n_target, feature_dim)

    # Test with different masks
    test_masks = [
        {"name": "All targets", "mask": torch.ones(n_target, dtype=torch.bool)},
        {"name": "No targets", "mask": torch.zeros(n_target, dtype=torch.bool)},
        {"name": "First two targets", "mask": torch.tensor([True, True, False, False, False])},
        {"name": "Alternate targets", "mask": torch.tensor([True, False, True, False, True])}
    ]

    for test in test_masks:
        print(f"\nTesting selection with: {test['name']}")
        mask = test["mask"]
        selected = select_targets_by_mask(target_results, mask)

        # Print shapes
        print(f"Original shape: {target_results.shape}")
        print(f"Selected shape: {selected.shape}")

        # Check selected shape
        expected_count = torch.sum(mask).item()
        assert selected.shape == (batch_size, expected_count, feature_dim), \
            f"Selected shape mismatch. Expected {(batch_size, expected_count, feature_dim)}, got {selected.shape}"

        # Check selected content
        selected_indices = torch.where(mask)[0]
        for i, idx in enumerate(selected_indices):
            assert torch.all(selected[:, i] == target_results[:, idx]), \
                f"Selected content mismatch at index {i}"

    print("\nAll selection tests passed!")



if __name__ == "__main__":
    # Run all tests
    print("===== Testing mask creation =====")
    test_create_target_mask()

    print("\n===== Testing target selection =====")
    test_select_targets_by_mask()

