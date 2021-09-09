export GPU_TRAINING=9,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### DML runs using ViT backbone

####################
# ... margin loss  #
####################
## nolayernorm
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize_nolayernorm' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename baseline_marginloss_vit_s_p16_224_384fNoLN_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

############################
# ... multisimilarity loss #
############################
## nolayernorm
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize_nolayernorm' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename baseline_msloss_vit_s_p16_224_384fNoLN_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

## baselines large dims
#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
#               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename msloss_bninception_frozen_normalize_512_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
#               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename msloss_bninception_frozen_normalize_384_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=768' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.bninception.Network' 'model.params.config.Architecture.params.arch=bninception_frozen_normalize' \
#               'data.params.train.params.arch=bninception_frozen_normalize' 'data.params.validation.params.arch=bninception_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename msloss_bninception_frozen_normalize_768_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
###
#python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename msloss_resnet50_frozen_normalize_128_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename msloss_resnet50_frozen_normalize_384_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=512' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename msloss_resnet50_frozen_normalize_512_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=768' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename msloss_resnet50_frozen_normalize_768_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml



# optimizer comparison (ref= adam, lr=1e-5, wd=4e-4)

python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=optimizer_comparison' 'lightning.trainer.max_epochs=30' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               --savename msloss_vitS16_baseline_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=optimizer_comparison' 'lightning.trainer.max_epochs=30' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.type_optim=adamW' 'model.base_learning_rate=0.00001' 'model.weight_decay=0.0004' \
#               --savename msloss_vitS16_adamW_lr_1e5_wd_4e4_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=optimizer_comparison' 'lightning.trainer.max_epochs=30' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.type_optim=adamW' 'model.base_learning_rate=0.00001' 'model.weight_decay=0.0001' \
#               --savename msloss_vitS16_adamW_lr_1e5_wd_1e4_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=optimizer_comparison' 'lightning.trainer.max_epochs=30' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.type_optim=adamW' 'model.base_learning_rate=0.00001' 'model.weight_decay=0.005' \
#               --savename msloss_vitS16_adamW_lr_1e5_wd_5e3_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#####
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=optimizer_comparison' 'lightning.trainer.max_epochs=30' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.type_optim=adamW' 'model.base_learning_rate=0.00005' 'model.weight_decay=0.0004' \
#               --savename msloss_vitS16_adamW_lr_5e5_wd_4e4_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=optimizer_comparison' 'lightning.trainer.max_epochs=30' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.type_optim=adamW' 'model.base_learning_rate=0.0001' 'model.weight_decay=0.0004' \
#               --savename msloss_vitS16_adamW_lr_1e4_wd_4e4_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=optimizer_comparison' 'lightning.trainer.max_epochs=30' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.type_optim=adamW' 'model.base_learning_rate=0.000005' 'model.weight_decay=0.0004' \
#               --savename msloss_vitS16_adamW_lr_5e6_wd_4e4_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=optimizer_comparison' 'lightning.trainer.max_epochs=30' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.type_optim=adamW' 'model.base_learning_rate=0.000001' 'model.weight_decay=0.0004' \
#               --savename msloss_vitS16_adamW_lr_1e6_wd_4e4_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml




#########################
## ... proxyanchor loss #
#########################
## nolayernorm
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize_nolayernorm' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' 'model.params.config.Loss.params.embed_dim=384' \
#               --savename baseline_proxy_vit_s_p16_224_384fNoLN_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml

# search lr

#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=search_lr' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.embed_dim=384' 'model.base_learning_rate=5e-5' \
#               --savename proxyanchor_vit_s_p16_224_384f_cars196_5e-5_0.0004 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=search_lr' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.embed_dim=384' 'model.base_learning_rate=1e-4' \
#               --savename proxyanchor_vit_s_p16_224_384f_cars196_1e-4_0.0004 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml

