export GPU_TRAINING=0,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### DML runs using ViT backbone

# ... multisimilarity loss
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_base_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_base_patch16_224_normalize' 'data.params.validation.params.arch=vit_base_patch16_224_normalize' \
#               --savename baseline_multisimilarity_vit_base_patch16_224_768f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_base_patch16_224_in21k_normalize' \
#               'data.params.train.params.arch=vit_base_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_base_patch16_224_in21k_normalize' \
#               --savename baseline_multisimilarity_vit_base_patch16_224_in21k_768f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
#               --savename baseline_multisimilarity_vit_small_patch16_224_384f_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
#               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
#               --savename baseline_multisimilarity_vit_small_patch16_224_in21k_384f_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml


##### vary LR
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=25' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'model.base_learning_rate=1e-6' 'lightning.logger.params.group=search_lr'\
#               --savename msloss_vit_s_p16_224_384f_cub200_lr_1e-6 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=25' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'model.base_learning_rate=5e-6' 'lightning.logger.params.group=search_lr'\
#               --savename msloss_vit_s_p16_224_384f_cub200_lr_5e-6 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=25' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'model.base_learning_rate=5e-5' 'lightning.logger.params.group=search_lr'\
#               --savename msloss_vit_s_p16_224_384f_cub200_lr_5e-5 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=25' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'model.base_learning_rate=1e-4' 'lightning.logger.params.group=search_lr'\
#               --savename msloss_vit_s_p16_224_384f_cub200_lr_1e-4 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml


####### search schedule
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=75' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'lightning.logger.params.group=search_schedule'\
#               'model.base_learning_rate=5e-6' 'model.weight_decay=0.0004' 'model.gamma=0.3' 'model.tau=[15,30]' \
#               --savename msloss_vit_s_p16_224_384f_cub200_5e-6_0.0004_0.3_15-30 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=75' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'lightning.logger.params.group=search_schedule'\
#               'model.base_learning_rate=5e-6' 'model.weight_decay=0.0004' 'model.gamma=0.3' 'model.tau=[10,20]' \
#               --savename msloss_vit_s_p16_224_384f_cub200_5e-6_0.0004_0.3_10-20 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=75' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'lightning.logger.params.group=search_schedule'\
#               'model.base_learning_rate=5e-6' 'model.weight_decay=0.0004' 'model.gamma=0.3' 'model.tau=[5,15]' \
#               --savename msloss_vit_s_p16_224_384f_cub200_5e-6_0.0004_0.3_5-15 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

###### search schedule
python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=75' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' 'lightning.logger.params.group=search_schedule'\
               'model.base_learning_rate=5e-6' 'model.weight_decay=0.0004' 'model.gamma=0.3' 'model.tau=[20,50]' \
               --savename msloss_vit_s_p16_224_384f_cars196_5e-6_0.0004_0.3_20-50 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=75' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' 'lightning.logger.params.group=search_schedule'\
               'model.base_learning_rate=5e-6' 'model.weight_decay=0.0004' 'model.gamma=0.3' 'model.tau=[40,60]' \
               --savename msloss_vit_s_p16_224_384f_cars196_5e-6_0.0004_0.3_40-60 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=75' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' 'lightning.logger.params.group=search_schedule'\
               'model.base_learning_rate=5e-6' 'model.weight_decay=0.0004' 'model.gamma=0.3' 'model.tau=[30,65]' \
               --savename msloss_vit_s_p16_224_384f_cars196_5e-6_0.0004_0.3_30-60 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml




# ... margin loss
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_base_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_base_patch16_224_normalize' 'data.params.validation.params.arch=vit_base_patch16_224_normalize' \
#               --savename baseline_marginloss_vit_base_patch16_224_768f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_base_patch16_224_in21k_normalize' \
#               'data.params.train.params.arch=vit_base_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_base_patch16_224_in21k_normalize' \
#               --savename baseline_marginloss_vit_base_patch16_224_in21k_768f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               --savename baseline_marginloss_vit_small_patch16_224_384f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
#               --savename baseline_marginloss_vit_small_patch16_224_in21k_384f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml