export GPU_TRAINING=3,
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
python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
              --savename baseline_multisimilarity_cub200_291121 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

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

#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=25' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.7' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_margin=0.3' \
#               --savename margin_msloss_vitS16_beta0.7_0.0005_margin0.3_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=25' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.9' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_margin=0.3' \
#               --savename margin_msloss_vitS16_beta0.9_0.0005_margin0.3_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' \
#               --savename margin_msloss_vitS16_beta0.5_0.0005_distance_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss.yaml

###################### MARGIN MULTISIMILARITY LOSS - CARS196
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename debug_msloss_r50d384_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=max_min' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_maxmin_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_distance_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=150' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' \
#               --savename margin_msloss_r50d384_beta0.5_0.0005_distance_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=120' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' 'data.params.batch_size=112' \
#               --savename debug_msloss_orig_r50d384_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=120' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' 'data.params.batch_size=112' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta_const=True' 'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=max_min' \
#               --savename debug_msloss_r50d384_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=150' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.6' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' \
#               --savename margin_msloss_r50d384_beta0.6_0.0005_distance_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=150' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.4' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=distance' \
#               --savename margin_msloss_r50d384_beta0.4_0.0005_distance_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml



#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=150' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.6' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_nbins=25'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_25_alpha0.5_pos_low_normal0.6_0.3_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=150' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.5' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_nbins=25'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_25_alpha0.5_pos_low_normal0.5_0.3_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml


#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=150' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.25' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.5' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_nbins=25'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_25_alpha0.25_pos_low_normal0.5_0.3_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=150' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.75' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.5' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_nbins=25'\
#               --savename margin_msloss_r50d384_beta0.5_0.0005_padsEma_25_alpha0.75_pos_low_normal0.5_0.3_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=150' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.4' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.5' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_nbins=25'\
#               --savename margin_msloss_r50d384_beta0.4_0.0005_padsEma_25_alpha0.5_pos_low_normal0.5_0.3_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=20000' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta=0.4' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=pads_ema' 'model.params.config.Loss.params.loss_multisimilarity_sampling_ema_alpha=0.5' \
#               'model.params.config.Loss.params.loss_multisimilarity_sampling_incl_pos=True' 'model.params.config.Loss.params.loss_multisimilarity_init_distr=normal_low' \
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_mu=0.5' 'model.params.config.Loss.params.loss_multisimilarity_init_distr_sigma=0.3'\
#               'model.params.config.Loss.params.loss_multisimilarity_init_distr_nbins=25'\
#               --savename margin_msloss_r50d384_beta0.4_0.0005_padsEma_25_alpha0.5_pos_low_normal0.5_0.3_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml
