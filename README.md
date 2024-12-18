# RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning

This file contains the instructions for reproducing RLDG as presented in the [paper](https://generalist-distillation.github.io/). 

## 1. Task setup
We focus this guide on reproducing the Connector Insertion experiment with a Franka arm.

1. Download and 3D print the [wrist camera mount](./static/files/wrist_mount.step) and the [D-type connector housing](./static/files/D-type_holder.step).
2. We provide the [purchase links](https://docs.google.com/spreadsheets/d/1JqI_bZoseVatrwyDrtJxsr7Ay13MIeRVuGdN5knIEHc/edit?usp=sharing) to the connectors we used in our experiment. You may choose to purchase what you need.
3. We used the robot controller and infra from the HIL-SERL project. Please refer to the [HIL-SERL GitHub](https://github.com/rail-berkeley/hil-serl) for more details.

> For instructions setting up the FMB Insertion and FMB Assembly tasks, please refer to the [FMB project page](https://functional-manipulation-benchmark.github.io).

## 2. Train RL Policy using HIL-SERL
To train the RL policies, we used the standard HIL-SERL recipe. We include the experiment specific configuration files at `rldg/hil-serl/connector_insert/`. To use this:
1. Install the [HIL-SERL repo](https://github.com/rail-berkeley/hil-serl) according to the instructions in the repo.
2. Copy the `rldg/hil-serl/connector_insert/` directory to the `hil-serl/examples/experiments/` directory in your HIL-SERL installation. Also add this experiment config to the `hil-serl/examples/experiments/mappings.py` file.
3. Follow the [HIL-SERL RAM Insertion](https://github.com/rail-berkeley/hil-serl/blob/main/docs/franka_walkthrough.md#1-ram-insertion) instructions to record 20 demos and train a HIL-SERL policy on connector insertion.
4. Once the policy is trained, we can roll out the policy to collect demonstrations. We provide the scripts to roll out the policy at `rldg/hil-serl/rollout_rl.py`. To run this, copy the file into `hil-serl/examples/` and run the `./rollout.sh` script inside the `connector_insert` directory.

> For our experiment, we trained 3 separate policies with USB A, VGA, and Ethernet connectors. 


## 3. Collecting Data
Coming soon

## 3. Fine-tuning OpenVLA and Octo
Coming soon
