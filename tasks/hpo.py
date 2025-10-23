import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from attrdictionary import AttrDict
import os
import json
from tasks.base_task import Task


class HPOBHandler:
    """HPO-B dataset handler for loading and preprocessing data"""
    
    def __init__(self, root_dir="HPOB/", mode="v3-test", surrogates_dir="saved-surrogates/"):
        """
        Constructor for the HPOBHandler.
        Inputs:
            * root_dir: path to directory with the benchmark data.
            * mode: mode name indicating how to load the data. Options:
                - v1: Loads HPO-B-v1
                - v2: Loads HPO-B-v2
                - v3: Loads HPO-B-v3
                - v3-test: Loads only the meta-test split from HPO-B-v3
                - v3-train-augmented: Loads all splits from HPO-B-v3, but augmenting the meta-train data with the less frequent search-spaces.
            * surrogates_dir: path to directory with surrogates models.
        """
        print("Loading HPO-B handler")
        self.mode = mode
        self.surrogates_dir = surrogates_dir
        self.seeds = ["test0", "test1", "test2", "test3", "test4"]

        if self.mode == "v3-test":
            self.load_data(root_dir, only_test=True)
        elif self.mode == "v3-train-augmented":
            self.load_data(root_dir, only_test=False, augmented_train=True)
        elif self.mode in ["v1", "v2", "v3"]:
            self.load_data(root_dir, version=self.mode, only_test=False)
        else:
            raise ValueError("Provide a valid mode")

        surrogates_file = surrogates_dir+"summary-stats.json"
        if os.path.isfile(surrogates_file):
            with open(surrogates_file) as f:
                self.surrogates_stats = json.load(f)

    def load_data(self, rootdir="", version="v3", only_test=True, augmented_train=False):
        """
        Loads data with some specifications.
        Inputs:
            * root_dir: path to directory with the benchmark data.
            * version: name indicating what HPOB version to use. Options: v1, v2, v3).
            * Only test: Whether to load only testing data (valid only for version v3).  Options: True/False
            * augmented_train: Whether to load the augmented train data (valid only for version v3). Options: True/False
        """
        print("Loading data...")
        meta_train_augmented_path = os.path.join(rootdir, "meta-train-dataset-augmented.json")
        meta_train_path = os.path.join(rootdir, "meta-train-dataset.json")
        meta_test_path = os.path.join(rootdir,"meta-test-dataset.json")
        meta_validation_path = os.path.join(rootdir, "meta-validation-dataset.json")
        bo_initializations_path = os.path.join(rootdir, "bo-initializations.json")

        with open(meta_test_path, "rb") as f:
            self.meta_test_data = json.load(f)

        with open(bo_initializations_path, "rb") as f:
            self.bo_initializations = json.load(f)

        if not only_test:
            if augmented_train or version=="v1":
                with open(meta_train_augmented_path, "rb") as f:
                    self.meta_train_data = json.load(f)
            else:
                with open(meta_train_path, "rb") as f:
                    self.meta_train_data = json.load(f)
            with open(meta_validation_path, "rb") as f:
                self.meta_validation_data = json.load(f)

        if version != "v3":
            temp_data = {}
            for search_space in self.meta_train_data.keys():
                temp_data[search_space] = {}

                for dataset in self.meta_train_data[search_space].keys():
                    temp_data[search_space][dataset] =  self.meta_train_data[search_space][dataset]

                if search_space in self.meta_test_data.keys():
                    for dataset in self.meta_test_data[search_space].keys():
                        temp_data[search_space][dataset] = self.meta_test_data[search_space][dataset]

                    for dataset in self.meta_validation_data[search_space].keys():
                        temp_data[search_space][dataset] = self.meta_validation_data[search_space][dataset]

            self.meta_train_data = None
            self.meta_validation_data = None
            self.meta_test_data = temp_data

        self.search_space_dims = {}

        for search_space in self.meta_test_data.keys():
            dataset = list(self.meta_test_data[search_space].keys())[0]
            X = self.meta_test_data[search_space][dataset]["X"][0]
            self.search_space_dims[search_space] = len(X)

    def normalize(self, y, y_min=None, y_max=None):
        """Normalize target values"""
        if y_min is None:
            return (y-np.min(y))/(np.max(y)-np.min(y))
        else:
            return(y-y_min)/(y_max-y_min)

    def get_search_spaces(self):
        return list(self.meta_test_data.keys())

    def get_datasets(self, search_space):
        return list(self.meta_test_data[search_space].keys())

    def get_seeds(self):
        return self.seeds

    def get_search_space_dim(self, search_space):
        return self.search_space_dims[search_space]


class HPOB:
    """HPO-B dataset loader with specific meta-dataset configuration"""
    
    def __init__(self, meta_dataset="glmnet", data_path=None):
        self.datasets_list = {"ranger": "7609", "glmnet": "5860", "svm": "5891", "rpart": "5859", "xgboost": "5971"}
        self.meta_dataset = meta_dataset
        self.path = data_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
        self.data = self.get_data(meta_dataset)
        self.dataset_ids = list(self.data.keys())
        self.n_dataset = len(self.dataset_ids)
        self.min_data_size = len(self.data[self.dataset_ids[0]]['X'])
        self.dim_x = len(self.data[self.dataset_ids[0]]['X'][0])

    def sample(self, batch_size=16, n_context=None, n_query=None, n_target=10, min_n_context=5, max_n_context=10):
        """Sample a batch of HPO data"""
        batch = AttrDict()
        n_context = n_context or torch.randint(low=min_n_context, high=max_n_context, size=[1]).item()
        n_query = n_query or self.min_data_size - n_context - n_target
        assert n_target is not None, "n_target should be specified."

        batch.context_x = torch.zeros([batch_size, n_context, self.dim_x])
        batch.context_y = torch.zeros([batch_size, n_context, 1])
        batch.query_x = torch.zeros([batch_size, n_query, self.dim_x])
        batch.query_y = torch.zeros([batch_size, n_query, 1])
        batch.target_x = torch.zeros([batch_size, n_target, self.dim_x])
        batch.target_y = torch.zeros([batch_size, n_target, 1])

        for i in range(batch_size):
            dataset_id = np.random.choice(self.dataset_ids)
            dataset = self.data[dataset_id]
            X = torch.tensor(dataset['X'], dtype=torch.float32)
            y = torch.tensor(dataset['y'], dtype=torch.float32)
            n_data = X.shape[0]
            indices = torch.randperm(n_data)
            context_indices = indices[:n_context]
            query_indices = indices[n_context:n_context+n_query]
            target_indices = indices[n_context+n_query:n_context+n_query+n_target]

            batch.context_x[i] = X[context_indices]
            batch.context_y[i] = y[context_indices]
            batch.query_x[i] = X[query_indices]
            batch.query_y[i] = y[query_indices]
            batch.target_x[i] = X[target_indices]
            batch.target_y[i] = y[target_indices]

        return batch

    def get_test_set(self):
        """Load test set data"""
        with open(f'{self.path}/HPOB/{self.meta_dataset}_test.json', 'r') as f:
            data = json.load(f)
        return data

    def get_bo_initializations(self):
        """Load BO initialization configurations"""
        with open(f'{self.path}/HPOB/bo-initializations.json', 'r') as f:
            data = json.load(f)
        return data

    def get_available_test_set_id(self, data):
        """Get available test dataset IDs"""
        return list(data.keys())

    def sample_test_set(self, n_context=None, n_query=None, n_target=None):
        """Sample test set data with specific initialization"""
        seeds = ["test0", "test1", "test2", "test3", "test4"]
        all_bo_initializations = self.get_bo_initializations()
        all_dataset_ids = self.get_available_test_set_id(self.get_test_set())
        batch_size = 5 * len(all_dataset_ids)

        batch = AttrDict()
        assert n_context is not None, "n_context should be specified."
        n_query = n_query
        n_target = n_target

        data = self.get_test_set()

        batch_context_x = []
        batch_context_y = []
        batch_query_x = []
        batch_query_y = []
        batch_target_x = []
        batch_target_y = []

        for dataset_id in all_dataset_ids:
            dataset = data[dataset_id]
            for seed in seeds:
                init_ids = all_bo_initializations[self.datasets_list[self.meta_dataset]][dataset_id][seed]

                X = torch.tensor(dataset['X'], dtype=torch.float32)
                y = torch.tensor(dataset['y'], dtype=torch.float32)
                x_context = X[init_ids]
                y_context = y[init_ids]

                mask = torch.ones(X.shape[0], dtype=torch.bool)
                mask[init_ids] = False

                X_rest = X[mask]
                y_rest = y[mask]

                x_query = X_rest[:n_query]
                y_query = y_rest[:n_query]
                x_target = X_rest[n_query:n_query+n_target]
                y_target = y_rest[n_query:n_query+n_target]

                batch_context_x.append(x_context)
                batch_context_y.append(y_context)
                batch_query_x.append(x_query)
                batch_query_y.append(y_query)
                batch_target_x.append(x_target)
                batch_target_y.append(y_target)

        batch.context_x = torch.stack(batch_context_x, dim=0)
        batch.context_y = torch.stack(batch_context_y, dim=0)
        batch.query_x = torch.stack(batch_query_x, dim=0)
        batch.query_y = torch.stack(batch_query_y, dim=0)
        batch.target_x = torch.stack(batch_target_x, dim=0)
        batch.target_y = torch.stack(batch_target_y, dim=0)

        return batch

    def get_data(self, meta_dataset):
        """Load meta-dataset from JSON file"""
        with open(f'{self.path}/HPOB/{meta_dataset}.json', 'r') as f:
            data = json.load(f)
        return data


class HPOTask(Task):
    """Hyperparameter Optimization Task for amortized inference using HPO-B dataset"""

    def __init__(
            self,
            name: str = "HPO",
            meta_dataset: str = "glmnet",  # HPO-B meta-dataset: glmnet, ranger, svm, rpart, xgboost
            embedding_type: str = "data",  # Only data mode for HPO tasks
            n_context_init: int = 5,  # number of initial context points
            n_query_init: int = 10,  # number of initial query points
            n_target_data: int = 5,  # number of target points
            min_n_context: int = 5,  # minimum number of context points
            max_n_context: int = 10,  # maximum number of context points
            data_path: str = None,  # path to data directory
            normalize_y: bool = False,  # whether to normalize target values
            dim_x: int = None,  # allow override from config
            dim_y: int = None,  # allow override from config
            **kwargs
    ) -> None:
        try:
            # Initialize HPOB data loader
            self.hpob = HPOB(meta_dataset=meta_dataset, data_path=data_path)
            
            # Set dimensions based on actual dataset, but allow config override
            actual_dim_x = self.hpob.dim_x
            actual_dim_y = 1  # HPO tasks have 1D output (performance metric)
            
            # Use actual dimensions from dataset, warn if config values differ
            if dim_x is not None and dim_x != actual_dim_x:
                print(f"Warning: Config dim_x ({dim_x}) differs from dataset dim_x ({actual_dim_x}). Using dataset dimension.")
            if dim_y is not None and dim_y != actual_dim_y:
                print(f"Warning: Config dim_y ({dim_y}) differs from dataset dim_y ({actual_dim_y}). Using dataset dimension.")
                
            dim_x = actual_dim_x
            dim_y = actual_dim_y
            
        except Exception as e:
            # If data loading fails, provide helpful error message
            print(f"Error loading HPO-B dataset '{meta_dataset}': {e}")
            print(f"Please ensure data files are available in the expected location.")
            print(f"Expected file: data/HPOB/{meta_dataset}.json")
            
            # Use provided dimensions or reasonable defaults
            dim_x = dim_x or 5  # reasonable default
            dim_y = dim_y or 1
            
            # Set a dummy HPOB object to prevent further errors
            self.hpob = None
        
        super().__init__(
            dim_x=dim_x, 
            dim_y=dim_y, 
            mode="data",  # HPO tasks only support data mode
            **kwargs
        )

        self.name = name
        self.meta_dataset = meta_dataset
        self.embedding_type = embedding_type
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.n_target_data = n_target_data
        self.min_n_context = min_n_context
        self.max_n_context = max_n_context
        self.normalize_y = normalize_y
        
        # HPO tasks don't have theta parameters
        self.n_target_theta = 0

        if self.embedding_type != "data":
            raise ValueError("HPO tasks only support 'data' embedding type")

    def to_design_space(self, xi):
        """Convert normalized design to actual input domain (identity for HPO)"""
        return xi

    def normalise_outcomes(self, y):
        """Normalize outcomes if needed"""
        if self.normalize_y:
            # Normalize to [0, 1] range
            y_min = y.min(dim=1, keepdim=True)[0]
            y_max = y.max(dim=1, keepdim=True)[0]
            y_range = y_max - y_min
            # Avoid division by zero
            y_range = torch.where(y_range == 0, torch.ones_like(y_range), y_range)
            return (y - y_min) / y_range
        return y

    def forward(self, xi, theta=None):
        """
        For HPO task, this method is not used as we load real data rather than generating it.
        Included for compatibility with base class interface.
        """
        raise NotImplementedError("HPO task uses real data, not generated data")

    def sample_batch(self, batch_size):
        """Sample a batch of HPO data in the format expected by the training framework"""
        if self.hpob is None:
            raise RuntimeError("HPO-B data not loaded. Cannot sample batch. Please check data files.")
            
        # Use HPOB to sample data
        hpob_batch = self.hpob.sample(
            batch_size=batch_size,
            n_context=self.n_context_init,
            n_query=self.n_query_init,
            n_target=self.n_target_data,
            min_n_context=self.min_n_context,
            max_n_context=self.max_n_context
        )

        # Convert to format expected by training framework
        batch = AttrDict()
        
        # Copy data from HPOB batch
        batch.context_x = hpob_batch.context_x
        batch.context_y = hpob_batch.context_y
        batch.query_x = hpob_batch.query_x
        batch.query_y = hpob_batch.query_y
        batch.target_x = hpob_batch.target_x
        batch.target_y = hpob_batch.target_y

        # Normalize outcomes if requested
        if self.normalize_y:
            # Concatenate all y values for normalization
            all_y = torch.cat([batch.context_y, batch.query_y, batch.target_y], dim=1)
            all_y_norm = self.normalise_outcomes(all_y)
            
            # Split back into components
            n_context = batch.context_y.shape[1]
            n_query = batch.query_y.shape[1]
            
            batch.context_y = all_y_norm[:, :n_context]
            batch.query_y = all_y_norm[:, n_context:n_context+n_query]
            batch.target_y = all_y_norm[:, n_context+n_query:]

        # Set additional attributes for compatibility
        batch.target_theta = None  # HPO tasks don't have theta
        batch.target_all = batch.target_y  # For data mode, target_all is target_y
        batch.n_target_theta = self.n_target_theta

        return batch

    def __str__(self) -> str:
        info = {
            'name': self.name,
            'meta_dataset': self.meta_dataset,
            'dim_x': self.dim_x,
            'dim_y': self.dim_y,
            'embedding_type': self.embedding_type,
            'n_context_init': self.n_context_init,
            'n_query_init': self.n_query_init,
            'n_target_data': self.n_target_data,
            'normalize_y': self.normalize_y
        }
        return f"HPOTask({', '.join('{}={}'.format(key, val) for key, val in info.items())})"


if __name__ == "__main__":
    # Test the HPO task
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description='HPOTask demonstration')
    parser.add_argument('--meta_dataset', type=str, default='ranger',
                      choices=['glmnet', 'ranger', 'svm', 'rpart', 'xgboost'],
                      help='HPO-B meta-dataset to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for demonstration')
    args = parser.parse_args()

    try:
        # Create the HPO task
        task = HPOTask(
            meta_dataset=args.meta_dataset,
            n_context_init=5,
            n_query_init=100,
            n_target_data=100,
            normalize_y=False
        )

        print(f"Created HPO task: {task}")
        print(f"Input dimension: {task.dim_x}")
        print(f"Number of datasets: {task.hpob.n_dataset}")

        # Sample a batch
        batch = task.sample_batch(args.batch_size)
        
        print(f"\nBatch shapes:")
        print(f"Context X: {batch.context_x.shape}")
        print(f"Context Y: {batch.context_y.shape}")
        print(f"Query X: {batch.query_x.shape}")
        print(f"Query Y: {batch.query_y.shape}")
        print(f"Target X: {batch.target_x.shape}")
        print(f"Target Y: {batch.target_y.shape}")
        
        print(f"\nSample statistics:")
        print(f"Context Y range: [{batch.context_y.min():.3f}, {batch.context_y.max():.3f}]")
        print(f"Query Y range: [{batch.query_y.min():.3f}, {batch.query_y.max():.3f}]")
        print(f"Target Y range: [{batch.target_y.min():.3f}, {batch.target_y.max():.3f}]")

    except FileNotFoundError as e:
        print(f"Error: Could not find HPO-B data files. Please ensure data is available at the expected path.")
        print(f"Expected files: data/HPOB/{args.meta_dataset}.json")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"Error: {e}")
