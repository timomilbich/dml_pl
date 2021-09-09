export GPU_TRAINING=5,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"
export BETTER_EXCEPTIONS=1

### BASELINE CHECKS
# ... margin loss
#python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#              --savename baseline_marginloss_cub200_CHECK --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

## ... multisimilarity loss
#python main.py 'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
#              --savename baseline_multisimilarity_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
#               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
#              --savename baseline_multisimilarity_cub200_CHECK --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
#               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
#               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
#               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
#              --savename baseline_multisimilarity_sop_CHECK --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml


############### DEBUG MULTISIMILARITY LOSS ##########################
## learnable threshold
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               --savename margin_msloss_vitS16_beta0.5_0.0005_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.7' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               --savename margin_msloss_vitS16_beta0.7_0.0005_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.001' \
#               --savename margin_msloss_vitS16_beta0.5_0.001_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=25' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'model.params.config.Loss.params.loss_multisimilarity_beta=0.7' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
               'model.params.config.Loss.params.loss_multisimilarity_margin=0.3' \
               --savename margin_msloss_vitS16_beta0.7_0.0005_margin0.3_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=25' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'model.params.config.Loss.params.loss_multisimilarity_beta=0.9' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
               'model.params.config.Loss.params.loss_multisimilarity_margin=0.3' \
               --savename margin_msloss_vitS16_beta0.9_0.0005_margin0.3_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml