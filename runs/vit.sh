export GPU_TRAINING=0,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### DML runs using ViT backbone

# ... margin loss
python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_base_patch16_224_normalize' \
               'data.params.train.params.arch=vit_base_patch16_224_normalize' 'data.params.validation.params.arch=vit_base_patch16_224_normalize' \
               --savename baseline_marginloss_vit_base_patch16_224_768f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_base_patch16_224_in21k_normalize' \
               'data.params.train.params.arch=vit_base_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_base_patch16_224_in21k_normalize' \
               --savename baseline_marginloss_vit_base_patch16_224_in21k_768f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml


# ... multisimilarity loss
python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_base_patch16_224_normalize' \
               'data.params.train.params.arch=vit_base_patch16_224_normalize' 'data.params.validation.params.arch=vit_base_patch16_224_normalize' \
               --savename baseline_multisimilarity_vit_base_patch16_224_768f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_base_patch16_224_in21k' \
               'data.params.train.params.arch=vit_base_patch16_224_in21k' 'data.params.validation.params.arch=vit_base_patch16_224_in21k' \
               --savename baseline_multisimilarity_vit_base_patch16_224_in21k_768f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml