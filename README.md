# COMPASS
What Went Wrong? Closing the Sim-to-Real Gap via Differentiable Causal Discovery

Conference on Robot Learning (CoRL), 2023

[[Webpage]](https://sites.google.com/view/sim2real-compass) | [[Arxiv]](https://arxiv.org/abs/2306.15864)

Please raise an issue or reach out to the authors if you need help with running the code.

## Prepare Conda Env
#### Note: Our simulation environments are modified from [robosuite](https://github.com/ARISE-Initiative/robosuite).
```Shell
conda env create -f environment.yml
pip install -e .
```

## How to run
```Shell
cd ./scripts/
conda activate compass

## You could start with our first air hockey environment and random action
python3 main.py --exp_name pusher

## Running code for drop environment and random action
python3 main.py --exp_name drop

## Running code for air hockey environment with SAC/PPO Agent training in the loop
python3 main.py --use_sac_agent 
python3 main.py --use_ppo_agent
```
There are a few parameters that you can tune from arguments, please feel free to modify them and discovery how COMPASS performs.

## Logs will be saved using tensorboard
```Shell
## Both the causal discovery and agent training will be saved under logdir folder
tensorboard --logdir= /Sim2Real-Compass/scripts/logdir/
```
## Acknowledgement 
[Robosuite: A Modular Simulation Framework and Benchmark for Robot Learning](https://robosuite.ai/) \\ 
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

## Citation
```
@misc{huang2023went,
      title={What Went Wrong? Closing the Sim-to-Real Gap via Differentiable Causal Discovery}, 
      author={Peide Huang and Xilun Zhang and Ziang Cao and Shiqi Liu and Mengdi Xu and Wenhao Ding and Jonathan Francis and Bingqing Chen and Ding Zhao},
      year={2023},
      eprint={2306.15864},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
