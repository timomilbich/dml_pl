export GPU_TRAINING=7,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### BASELINE CHECKS
## ... margin loss
#python main.py 'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#            --savename baseline_marginloss_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
#
## ... multisimilarity loss
#python main.py 'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#            --savename baseline_multisimilarity_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
## ... proxyanchor
#python main.py 'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#            --savename baseline_proxyanchor_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               --savename baseline_multisimilarity_vitS16_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename baseline_multisimilarity_vitS16_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
#               --savename baseline_multisimilarity_vitS16_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml


############### DEBUG MULTISIMILARITY LOSS ##########################
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               --savename debug_msloss_r50d384_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               --savename debug_msloss_vitS16_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml

## learnable threshold
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.7' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               --savename margin_msloss_r50d384_beta0.7_0.0005_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.001' \
#               --savename margin_msloss_r50d384_beta0.5_0.001_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_margin=0.3' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_margin0.3_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' \
               --savename margin_msloss_r50d384_beta0.5_0.0005_distance_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml

               python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_const=True' \
               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' \
               --savename margin_msloss_r50d384_beta0.5_fixed_distance_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml