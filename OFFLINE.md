# Offline RL Experiments
Dataset path: ```/fs/cml-projects/taco_rl/offline_data/${DATASET_TYPE}```
DATASET_TYPE: ```[hopper_hop_mediumreplay, hopper_hop_replay, walker_run_mediumreplay, walker_run_replay, cheetah_run_mediumreplay, cheetah_run_replay] ```
To be added: ```[quadruped_run_mediumreplay, quadruped_run_replay, hopper_hop_expert, walker_run_expert, cheetah_run_expert] ```


#### BC
To run BC:
```
CUDA_VISIBLE_DEVICES=0 python -W ignore train_offline.py target=bc.BCAgent \
task=TASK batch_size=256 seed=${SEED} exp_name=bc_${DATASET_TYPE}_${SEED} \
offline_data_dir=/fs/cml-projects/taco_rl/offline_data/${DATASET_TYPE}
```
#### TD3+BC
To run TD3/DrQ-v2+BC:
```
CUDA_VISIBLE_DEVICES=0 python -W ignore train_offline.py target=drqv2_offline.DrQV2Agent \
task=TASK batch_size=256 seed=${SEED} drqv2=true \
exp_name=drqbc_${DATASET_TYPE}_${SEED} \
offline_data_dir=/fs/cml-projects/taco_rl/offline_data/${DATASET_TYPE}
```

To run TACO+BC:
```
CUDA_VISIBLE_DEVICES=0 python -W ignore train_offline.py target=drqv2_offline.DrQV2Agent \
task=TASK batch_size=1024 seed=${SEED} drqv2=false \
multistep=1 reward=true exp_name=tacobc_${DATASET_TYPE}_${SEED} \
offline_data_dir=/fs/cml-projects/taco_rl/offline_data/${DATASET_TYPE}
```
(Note: Change multistep=1 to 3 for hopper hop!)

#### CQL
To run CQL:
```
CUDA_VISIBLE_DEVICES=0 python -W ignore train_offline.py target=cql.CQLAgent \
task=TASK batch_size=256 seed=${SEED} drqv2=true temp=5.0 \
exp_name=cql_${DATASET_TYPE}_${SEED} \
offline_data_dir=/fs/cml-projects/taco_rl/offline_data/${DATASET_TYPE}
```
To run TACO-CQL:
```
CUDA_VISIBLE_DEVICES=0 python -W ignore train_offline.py target=cql.CQLAgent \
task=TASK batch_size=256 seed=${SEED} drqv2=false temp=5.0 \
multistep=1 reward=true exp_name=tacocql_${DATASET_TYPE}_${SEED} \
offline_data_dir=/fs/cml-projects/taco_rl/offline_data/${DATASET_TYPE}
```
TEMP in {0.5,1,5,10} (Larger for dataset of better quality)


#### IQL
To run IQL:
```
CUDA_VISIBLE_DEVICES=0 python -W ignore train_offline.py target=iql.IQLAgent \
task=TASK batch_size=256 seed=${SEED} drqv2=true \
temperature=3.0 exp_name=iql_${DATASET_TYPE}_${SEED} \
offline_data_dir=/fs/cml-projects/taco_rl/offline_data/${DATASET_TYPE}
```
TEMP in {0.5,3,10}

