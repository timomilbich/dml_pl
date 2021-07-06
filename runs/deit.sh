export GPU_TRAINING=7,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### DML runs using DEIT-S backbone

# ... margin loss
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=deit' 'lightning.trainer.max_epochs=60' \
#      --savename baseline_marginloss_deitS384f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
#python main.py 'lightning.logger.params.group=deit' 'lightning.trainer.max_epochs=60' \
#        --savename baseline_marginloss_deitS128_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=deit' 'lightning.trainer.max_epochs=60' \
#      --savename baseline_marginloss_deitS512_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=deit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.deit.Network' 'model.params.config.Architecture.params.arch=deit_small_patch16_224_normalize' \
               'data.params.train.params.arch=deit_small_patch16_224_normalize' 'data.params.validation.params.arch=deit_small_patch16_224_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               --savename baseline_marginloss_deit_small_patch16_224_384f_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml


# ... multisimilarity loss
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=deit' 'lightning.trainer.max_epochs=60' \
#              --savename baseline_multisimilarity_deitS384f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python main.py 'lightning.logger.params.group=deit' 'lightning.trainer.max_epochs=60' \
#              --savename baseline_multisimilarity_deitS128_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=deit' 'lightning.trainer.max_epochs=60' \
#              --savename baseline_multisimilarity_deitS512_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=deit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.deit.Network' 'model.params.config.Architecture.params.arch=deit_small_patch16_224_normalize' \
               'data.params.train.params.arch=deit_small_patch16_224_normalize' 'data.params.validation.params.arch=deit_small_patch16_224_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               --savename baseline_multisimilarity_deit_small_patch16_224_384f_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml