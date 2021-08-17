export GPU_TRAINING=9,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### DML runs using ViT backbone

############################
# ... multisimilarity loss #
############################

# baselines no pretraining
python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=baselines_nopretrain' 'lightning.trainer.max_epochs=60' \
               'model.params.config.Architecture.target=architectures.resnet18.Network' 'model.params.config.Architecture.params.arch=resnet18_normalize' \
               'model.params.config.Architecture.params.pretraining=none' 'model.base_learning_rate=1e-1' \
               'data.params.train.params.arch=resnet18_normalize' 'data.params.validation.params.arch=resnet18_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               --savename msloss_resnet50_frozen_normalize_128_cub200_nopretrain --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=baselines_nopretrain' 'lightning.trainer.max_epochs=100' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'model.params.config.Architecture.params.pretraining=none' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename msloss_resnet50_frozen_normalize_128_cars196_nopretrain --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

#########################
## ... proxyanchor loss #
#########################

