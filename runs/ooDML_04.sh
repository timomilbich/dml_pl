export GPU_TRAINING=1,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### DML runs based on ooDML splits

############################
# ... multisimilarity loss #
############################

# baseline (Wide ResNet 50 2) - CUB200
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.wide_resnet.Network' 'model.params.config.Architecture.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.params.arch=wide_resnet50_2_frozen_normalize' 'data.params.validation.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'data.params.train.params.ooDML_split_id=1' 'data.params.validation.params.ooDML_split_id=1' \
               --savename msloss_wide_resnet50_2_frozen_normalize_384_cub200_s1 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.wide_resnet.Network' 'model.params.config.Architecture.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.params.arch=wide_resnet50_2_frozen_normalize' 'data.params.validation.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'data.params.train.params.ooDML_split_id=2' 'data.params.validation.params.ooDML_split_id=2' \
               --savename msloss_wide_resnet50_2_frozen_normalize_384_cub200_s2 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.wide_resnet.Network' 'model.params.config.Architecture.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.params.arch=wide_resnet50_2_frozen_normalize' 'data.params.validation.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'data.params.train.params.ooDML_split_id=3' 'data.params.validation.params.ooDML_split_id=3' \
               --savename msloss_wide_resnet50_2_frozen_normalize_384_cub200_s3 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.wide_resnet.Network' 'model.params.config.Architecture.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.params.arch=wide_resnet50_2_frozen_normalize' 'data.params.validation.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'data.params.train.params.ooDML_split_id=4' 'data.params.validation.params.ooDML_split_id=4' \
               --savename msloss_wide_resnet50_2_frozen_normalize_384_cub200_s4 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.wide_resnet.Network' 'model.params.config.Architecture.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.params.arch=wide_resnet50_2_frozen_normalize' 'data.params.validation.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'data.params.train.params.ooDML_split_id=5' 'data.params.validation.params.ooDML_split_id=5' \
               --savename msloss_wide_resnet50_2_frozen_normalize_384_cub200_s5 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.wide_resnet.Network' 'model.params.config.Architecture.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.params.arch=wide_resnet50_2_frozen_normalize' 'data.params.validation.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'data.params.train.params.ooDML_split_id=6' 'data.params.validation.params.ooDML_split_id=6' \
               --savename msloss_wide_resnet50_2_frozen_normalize_384_cub200_s6 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.wide_resnet.Network' 'model.params.config.Architecture.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.params.arch=wide_resnet50_2_frozen_normalize' 'data.params.validation.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'data.params.train.params.ooDML_split_id=7' 'data.params.validation.params.ooDML_split_id=7' \
               --savename msloss_wide_resnet50_2_frozen_normalize_384_cub200_s7 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.wide_resnet.Network' 'model.params.config.Architecture.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.params.arch=wide_resnet50_2_frozen_normalize' 'data.params.validation.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'data.params.train.params.ooDML_split_id=8' 'data.params.validation.params.ooDML_split_id=8' \
               --savename msloss_wide_resnet50_2_frozen_normalize_384_cub200_s8 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.wide_resnet.Network' 'model.params.config.Architecture.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.params.arch=wide_resnet50_2_frozen_normalize' 'data.params.validation.params.arch=wide_resnet50_2_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'data.params.train.params.ooDML_split_id=9' 'data.params.validation.params.ooDML_split_id=9' \
               --savename msloss_wide_resnet50_2_frozen_normalize_384_cub200_s9 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml