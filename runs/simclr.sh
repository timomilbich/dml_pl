export GPU_TRAINING=0,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### BASELINE CHECKS
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=simclr' 'lightning.trainer.max_epochs=60' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'data.params.batch_size=112' \
               'model.params.config.Loss.params.temperature=0.07' \
               --savename debug_simclr_r50d384_t0.07_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/simclrloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=simclr' 'lightning.trainer.max_epochs=60' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'data.params.batch_size=112' \
               'model.params.config.Loss.params.temperature=0.15' \
               --savename debug_simclr_r50d384_t0.15_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/simclrloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=simclr' 'lightning.trainer.max_epochs=60' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'data.params.batch_size=112' \
               'model.params.config.Loss.params.temperature=0.25' \
               --savename debug_simclr_r50d384_t0.25_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/simclrloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=simclr' 'lightning.trainer.max_epochs=60' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'data.params.batch_size=112' \
               'model.params.config.Loss.params.temperature=0.03' \
               --savename debug_simclr_r50d384_t0.03_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/simclrloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=simclr' 'lightning.trainer.max_epochs=60' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'data.params.batch_size=112' \
               'model.params.config.Loss.params.temperature=0.01' \
               --savename debug_simclr_r50d384_t0.01_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/simclrloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=simclr' 'lightning.trainer.max_epochs=200' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'data.params.batch_size=112' \
               'model.params.config.Loss.params.temperature=0.01' \
               --savename debug_simclr_r50d384_t0.01_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/simclrloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=simclr' 'lightning.trainer.max_epochs=200' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'data.params.batch_size=112' \
               'model.params.config.Loss.params.temperature=0.05' \
               --savename debug_simclr_r50d384_t0.05_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/simclrloss.yaml
