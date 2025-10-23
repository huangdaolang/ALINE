# Amortized Active Learning and Inference Engine (ALINE)

This repository contains the official implementation of the NeurIPS 2025 paper: "[ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition](https://arxiv.org/abs/2506.07259)".

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/huangdaolang/ALINE.git
    cd ALINE
    ```

2.  Create a conda environment and install dependencies:
    ```bash
    conda create -n aline python=3.12
    conda activate aline
    pip install -r requirements.txt
    ```
    

## Directory
- `config`: hydra configuration
- `distributions`: customised distribution classes
- `loss`: loss functions
- `model`: ALINE architecture
- `tasks`: simulating task data
- `utils`: helper functions
- `notebooks`: Jupyter notebooks for evaluation and analysis

## Train

### Active Learning (Gaussian Process)
1D
```shell
python train_aline.py task=al_mix max_epoch=200000 burning_epoch=20000 task.dim_x=1 task.n_target_theta=2 task.n_query_init=200 task.lengthscale_lower=0.1 task.lengthscale_upper=2.0 task.design_scale=5 gamma=1 file_name='aline_al_1d.pth' checkpoint_name='ckpt_al_1d.tar' min_T=30 T=30
```

2D
```shell
python train_aline.py task=al_mix max_epoch=200000 burning_epoch=20000 task.dim_x=2 task.n_target_theta=3 task.n_query_init=200 task.lengthscale_lower=0.1 task.lengthscale_upper=2.0 task.design_scale=5 gamma=1 file_name='aline_al_2d.pth' checkpoint_name='ckpt_al_2d.tar' min_T=50 T=50
```

### Location Finding
```shell
python train_aline.py task=location_finding task.theta_dist=uniform task.n_target_theta=2 task.K=1 lr=1e-3 T=30 task.n_query_init=200 max_epoch=100000 burning_epoch=20000 eval=bed eval.batch_size=1000 eval.L_final=1000000 eval.batch_size_final=200 eval.n_query_final=2000 eval.T_final=35
```

### CES
```shell
python train_aline.py task=ces lr=1e-3 T=10 task.n_query_init=100 max_epoch=200000 burning_epoch=20000 eval=bed eval.batch_size=500 eval.T_final=20 eval.L_final=10000000 eval.batch_size_final=20 eval.n_query_final=2000 eval.T_final=15
```

### Psychometric model
```shell
python train_aline.py task=psychometric task.mask_type=["predefined"] min_T=30 T=30 gamma=0.99 max_epoch=100000 burning_epoch=10000 checkpoint_name='ckpt_psychometric.tar' file_name='aline_psychometric.pth'
```

## Citation
If you find this work useful in your research, please consider citing our paper:
```
@inproceedings{huang2025aline,
  title={ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition},
  author={Huang, Daolang and Wen, Xinyi and Bharti, Ayush and Kaski, Samuel and Acerbi, Luigi},
  booktitle={Thirty-ninth Conference on Neural Information Processing Systems},
  year={2025}
}
```

## License

This project is licensed under the MIT License.