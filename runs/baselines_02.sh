export GPU_TRAINING=0,
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

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_distance_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_const=True' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' \
#               --savename margin_msloss_r50d384_beta0.5_fixed_distance_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' 'model.params.config.Loss.params.loss_multisimilarity_sampling_upper_cutoff=0.8' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_distance_cut0.8_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' 'model.params.config.Loss.params.loss_multisimilarity_sampling_upper_cutoff=0.7' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_distance_cut0.7_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' 'model.params.config.Loss.params.loss_multisimilarity_sampling_upper_cutoff=0.6' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_distance_cut0.6_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' 'model.params.config.Loss.params.loss_multisimilarity_sampling_upper_cutoff=0.9' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_distance_cut0.9_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.25' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.25_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.1' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.25' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.1_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.1' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.1_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_pos_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.25' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.25_pos_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.75' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.75_pos_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' 'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_distance_pos_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=max_min' 'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_maxmin_pos_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_const=True' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' \
#               --savename margin_msloss_r50d384_beta0.5_fixed_padsEma_alpha0.5_pos_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#######################3
########################

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.4' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.4_pos_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=False' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_CHECK_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_pos_CHECK_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_pos_low_uniform_init_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_pos_low_normal0.6_init_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.6' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.1'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_pos_low_normal0.6_0.1_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.6' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.2'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_pos_low_normal0.6_0.2_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.6' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_pos_low_normal0.6_0.3_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml


#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.7' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_pos_low_normal0.7_0.3_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.7' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_normalize_update=False'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_pos_low_normal0.7_0.3_nonorm_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.6' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=1.0'\
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_normalize_update=False'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_pos_low_normal0.6_1.0_nonorm_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml


#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=False' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.6' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=1.0'\
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_normalize_update=False'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_low_normal0.6_1.0_nonorm_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=False' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.6' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_normalize_update=False'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_low_normal0.6_0.3_nonorm_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'model.type_optim=adamW' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.6' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_alpha0.5_pos_low_normal0.6_0.3_adamW_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.6' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_nbins=10'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_10_alpha0.5_pos_low_normal0.6_0.3_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=150' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.6' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_nbins=40'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_40_alpha0.5_pos_low_normal0.6_0.3_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=150' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=False' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.6' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_nbins=10'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_10_alpha0.5_low_normal0.6_0.3_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=150' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.6' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_nbins=15'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_15_alpha0.5_pos_low_normal0.6_0.3_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.6' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_nbins=10'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_10_alpha0.5_pos_low_normal0.6_0.3_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

