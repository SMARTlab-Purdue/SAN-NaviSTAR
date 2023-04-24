# SAN-NaviSTAR
This repository contains the source code for our paper: "NaviSTAR: Socially Aware Robot Navigation with Hybrid Spatio-Temporal Graph Transformer and Preference Learning", submitted to IROS-2023.
For more details, please refer to [our project website](https://sites.google.com/view/san-navistar).


## Abstract
Developing robotic technologies for use in human society requires ensuring the safety of robots’ navigation behaviors while adhering to pedestrians’ expectations and social norms. However, maintaining real-time communication between robots and pedestrians to avoid collisions can be challenging.To address these challenges, we propose a novel socially-aware navigation benchmark called NaviSTAR, which utilizes a hybrid Spatio-Temporal grAph tRansformer (STAR) to understand interactions in human-rich environments fusing potential crowd multi-modal information. We leverage off-policy reinforcement learning algorithm with preference learning to train a policy and a reward function network with supervisor guidance.Additionally, we design a social score function to evaluate the overall performance of social navigation. To compare, we train and test our algorithm and other state-of-the-art methods in both simulator and real-world scenarios independently. Our results show that NaviSTAR outperforms previous methods with outstanding performance.



## Overview Architecture for NaviSTAR
<div align=center>
<img src="/figures/Architecture.jpg" width="800" />
</div>  
The NaviSTAR is composed of two parts: 1) Spatial Temporal Graph Transforemer Block, and 2) Multi-Modal Transformer Block. And NaviSTAR utilizes a spatial-temporal graph transformer block and a multi-modal transformer block to abstract environmental dynamics and human-robot interactions into an ST-graph for safe path planning in crowd-flled environments. The spatial transformer is designed to capture hybrid spatial interactions and generate spatial attention maps, while the temporal transformer presents long-term temporal dependencies and creates temporal attention maps. The multi-modal transformer is deployed to adapt to the uncertainty of multi-modality crowd movements, aggregating all heterogeneous spatial and temporal features. Finally, the planner generates the next timestep action by a decoder.



## Set Up
1. Install the required python package
```
pip install -r requirements.txt
```

2. Install [Human-in-Loop RL](https://github.com/rll-research/BPref) environment

3. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library

4. Install Environment and Navigation into pip
```
pip install -e .
```


## Run the code

1. Train a policy with preference learning. 
```
python train_NaviSTAR.py 
```

2. Test a policy.
```
python test_NaviSTAR.py
```

(The code was tested in Ubuntu 18.04 with Python 3.6.)

## Real-world Experiment

We also conducted real-world tests with 8 participants(1 female and 7 males, all aged over 18). The robot and participants followed the same behavior policy, starting point,and end goal in each test, and the planner of the robot was unknown and randomized for participants.

<div align=center>
<img src="/figures/perception.png" width="800" />
</div>  

## Simulation tests

<div align=center>
<img src="/figures/results.png" width="800" /> 
</div>  

## Citation
If you find the code or the paper useful for your research, please cite our paper:
```
@article{wang2023navistar,
  title={NaviSTAR: Socially Aware Robot Navigation with Hybrid Spatio-Temporal Graph Transformer and Preference Learning},
  author={Wang, Weizheng and Wang, Ruiqi and Mao, Le and Min, Byung-Cheol},
  journal={arXiv preprint arXiv:2304.05979},
  year={2023}
}
```

## Acknowledgement

Contributors:  
[Weizheng Wang](https://github.com/WzWang-Robot/FAPL); [Ruiqi Wang](https://github.com/R7-Robot?tab=repositories); Le Mao; Byung-Cheol Min.

Part of the code is based on the following repositories:  

[SARL](https://github.com/vita-epfl/CrowdNav); [FAPL](https://github.com/SMARTlab-Purdue/SAN-FAPL); and [B_Pref](https://github.com/rll-research/B_Pref).





