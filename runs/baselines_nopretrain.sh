export GPU_TRAINING=4,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

###########
# ... ViT #
###########
python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=baselines_nopretrain' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'model.params.config.Architecture.params.pretraining=None'\
               --savename marginloss_vit_s_p16_224_384f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=baselines_nopretrain' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' 'model.params.config.Architecture.params.pretraining=None'\
               --savename marginloss_vit_s_p16_224_384f_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

############
# ... DeiT #
############
python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=baselines_nopretrain' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.deit.Network' 'model.params.config.Architecture.params.arch=deit_small_patch16_224_normalize' \
               'data.params.train.params.arch=deit_small_patch16_224_normalize' 'data.params.validation.params.arch=deit_small_patch16_224_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'model.params.config.Architecture.params.pretraining=None'\
               'model.base_learning_rate=1e-4' \
               --savename marginloss_deit_s_p16_224_384f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=baselines_nopretrain' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.deit.Network' 'model.params.config.Architecture.params.arch=deit_small_patch16_224_normalize' \
               'data.params.train.params.arch=deit_small_patch16_224_normalize' 'data.params.validation.params.arch=deit_small_patch16_224_normalize' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' 'model.params.config.Architecture.params.pretraining=None'\
               'model.base_learning_rate=1e-4' \
               --savename marginloss_deit_s_p16_224_384f_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

################
# ... ResNet50 #
################
python main.py 'lightning.logger.params.group=baselines_nopretrain' 'lightning.trainer.max_epochs=100' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'model.params.config.Architecture.params.embed_dim=128' \
              --savename marginloss_resnet50_128_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
python main.py 'lightning.logger.params.group=baselines_nopretrain' 'lightning.trainer.max_epochs=100' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' 'model.params.config.Architecture.params.embed_dim=128' \
              --savename marginloss_resnet50_128_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
