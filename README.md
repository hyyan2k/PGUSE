# PGUSE: A Composite Predictive-Generative Approach to Monaural Universal Speech Enhancement

This is the official implementation of [PGUSE](https://arxiv.org/abs/2505.24576v1) model.



## Installation

1. Prepare a virtual environment with python, pytorch, and pytorch_lightning. (We use python==3.10.14, pytorch==2.0.0, and pytorch_lightning==2.0.7, but other versions probably also work.)
2. Install the package dependencies via `pip install -r requirements.txt`.



## Training

Before training, please check `config.yaml` to set hyperparameters, including **devices**, **logdir**, **dataset path**, ...

Then you can train the model by:

```
python train.py --config ./config/config.yaml
```



## Testing

First, specify the `ckpt_path` in `config/config.yaml`, and then run:

```
python test.py --config ./config/config.yaml --save_enhanced <path-to-savedir>
```



## Reference

[sgmse](https://github.com/sp-uhh/sgmse)

[sgmse-bbed](https://github.com/sp-uhh/sgmse-bbed)

[LiSenNet](https://github.com/hyyan2k/LiSenNet)

