export GPU_TRAINING=7,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### DML runs using ViT backbone

#########################
## ... proxyanchor loss #
#########################

# search lr

python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=search_lr' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
               'model.params.config.Loss.params.embed_dim=384' 'model.base_learning_rate=5e-5' \
               --savename proxyanchor_vit_s_p16_224_384f_cars196_5e-5_0.0004 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml
python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=search_lr' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
               'model.params.config.Loss.params.embed_dim=384' 'model.base_learning_rate=1e-4' \
               --savename proxyanchor_vit_s_p16_224_384f_cars196_1e-4_0.0004 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml
