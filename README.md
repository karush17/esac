# Evolve To Control
[Karush Suri](https://karush17.github.io/), Xiao Qi Shi, Yuri A. Lawryshyn Konstantinos N. Plataniotis

<p align="center"><img src="/images/schematic.gif"  height="400" width="650"/></p>

This is the original implementation of Evolution-based Soft Actor-Critic (ESAC) introduced in ["Evolve To Control: Evolution-based Soft Actor-Critic for Scalable Reinforcement Learning"](https://arxiv.org/). ESAC combines Evolution Strategies (ES) with Soft Actor-Critic (SAC) for state-of-the-performance equivalent to SAC and scalability comparable to ES. ESAC abstracts exploration from exploitation by exploring policies in weight space using evolutions and optimizing gradient-based knowledge using the SAC framework. ESAC makes use of a novel soft winner selection function and carries out genetic crossovers in hindsight. ESAC also introduces the novel Automatic Mutation Tuning (AMT) which maximizes the mutation rate of ES in a small clipped region and provides significant hyperparameter robustness.  

Find out more-
* [arXiv](https://arxiv.org/)
* [Blog Post](https://karush17.github.io/esac-web/blog.html)
* [Project Website](https://karush17.github.io/esac-web/)
* [Videos](https://karush17.github.io/esac-web/videos.html)

<!-- If you find our algorithm helpful then please cite the following-  

```

``` -->

# Requirements

ESAC implementation requires the following packages and libraries- 
```
argparse==1.1
gym==0.15.4
numpy
torch==1.4.0
mujoco_py==1.50.1.59
```
These can be installed using the terminal command `pip install -r requirements.txt`.

Environment tasks require [MuJoCo](http://www.mujoco.org/index.html) which is activated using a [license key](https://www.roboti.us/license.html) (free for students). A complete tutorial on setting up MuJoCo can be found [here](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Installing-Mujoco-py-on-Linux).  

However, if you wish to proceed without the MuJoCo library, then you can run ESAC on the LunarLanderContinuous-v2 task from [OpenAI Gym's suite](https://github.com/openai/gym).  

# Usage

To run ESAC on your local machine, clone the repository using the following- 
```
git clone https://github.com/karush17/esac.git
cd esac
```

Run ESAC using the following-  
```
python main.py --env ENV_NAME --seed SEED
```

For instance, ESAC can be run on the HalfCheetah-v2 environment using the following-
```
python main.py --env HalfCheetah-v2 --seed 0
```

To change the number of steps use the `--num_steps` flag-
```
python main.py --env Ant-v2 --num_steps 2e6
```

To change the number of winners and SAC agents in the ES population use the `--elite_rate` and `--grad_models` flags respectively-
```
python main.py --env Swimmer-v2 --elite_rate 0.4 --grad_models 1
```

To change the number of episodes played by the SAC agent during the gradient update interval use the `--sac_episodes` flag-
```
python main.py --env Hopper-v2 --num_steps 2e6 --sac_episodes 10
```

In order to have a look at the full flags list and their default values use the `--help` flag-
```
python main.py --help
```

# Reproducibility

It is recommended to use a virtual environment for reproducing the results in order to avoid version conflict. A virtual environment can be created and activated using the following-
```
virtualenv myenv
source myenv/bin/activate
```

Results from the paper can be reproduced using the following commands-

1. HalfCheetah-v2
```
python main.py --env HalfCheetah-v2
```

2. Humanoid-v2
```
python main.py --env Humanoid-v2 --num_steps 5e6
```

3. Ant-v2
```
python main.py --env Ant-v2 --sac_episodes 10 -lr_es 0.001
```

4. Walker2d-v2
```
python main.py --env Walker2d-v2 --sac_episodes 10
```

5. Swimmer-v2
```
python main.py --env Swimmer-v2 --sac_episodes 1 --grad_models 1
```

6. Hopper-v2
```
python main.py --env Hopper-v2 --num_steps 2e6 --sac_episodes 10
```

7. Reacher-v2
```
python main.py --env Reacher-v2 --num_steps 2e6
```

8. LunarLanderContinuous-v2
```
python main.py --env LunarLanderContinuous-v2
```

9. InvertedDoublePendulum-v2
```
python main.py --env InvertedDoublePendulum-v2
```

# Credits
ESAC was developed by [Karush Suri](https://karush17.github.io/) from the University of Toronto under the supervision of Xiao Qi Shi from RBC Captial Markets, Yuri A. Lawryshyn and Konstantinos N. Plataniotis from the Center for Management of Technology and Entrepreneurship (CMTE) and the Multimedia Laboratory at the University of Toronto respectively. Special thanks to our sponsors RBC Capital Markets and RBC Innovation Lab and to the staff of CMTE. 


