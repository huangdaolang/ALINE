# Amortized Active Learning and Inference Engine (ALINE)

## Directory
- `config`: hydra configuration
- `distributions`: customised distribution classes
- `loss`: loss functions
- `model`: ALINE architecture
- `task`: simulating task data
- `utils`: helper functions

## Installation
```shell
conda create -n aline python=3.12
conda activate aline
pip install -r requirements.txt
```


## Train


### Active Learning
1D
```shell
python run_al.py task=gp_mix max_epoch=200000 burning_epoch=20000 task.dim_x=1 task.n_target_theta=2 task.n_query_init=200 task.lengthscale_lower=0.1 task.lengthscale_upper=2.0 task.design_scale=5 gamma=1 file_name='aline_al_1d.pth' checkpoint_name='ckpt_al_1d.tar' min_T=30 T=30
````

2D
```shell
python run_al.py task=gp_mix max_epoch=200000 burning_epoch=20000 task.dim_x=2 task.n_target_theta=3 task.n_query_init=200 task.lengthscale_lower=0.1 task.lengthscale_upper=2.0 task.design_scale=5 gamma=1 file_name='aline_al_2d.pth' checkpoint_name='ckpt_al_2d.tar' min_T=50 T=50
```

### Location Finding
```shell
python run_bed.py task=location_finding task.theta_dist=uniform task.n_target_theta=2 task.K=1 lr=1e-3 T=30 task.n_query_init=200 eval_batch_size=1000 max_epoch=100000 burning_epoch=20000 L_final=1000000 eval_batch_size_final=200 n_query_final=2000 T_final=30
```

### CES
```shell
python run_bed.py task=ces lr=1e-3 T=10 task.n_query_init=100 eval_batch_size=500 T_final=20 max_epoch=200000 burning_epoch=40000 L_final=10000000 eval_batch_size_final=20 n_query_final=2000 T_final=10
```

### Psychometric model
```shell
python run_psychometric.py task=psychometric task.mask_type=["predefined"] min_T=30 T=30 gamma=0.99 max_epoch=100000 burning_epoch=10000 checkpoint_name='ckpt_psychometric.tar' file_name='aline_psychometric.pth'
```
