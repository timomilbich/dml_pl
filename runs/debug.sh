export GPU_TRAINING=3
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### DEBUGGING
#python main.py --savename debub_deitS_marginloss_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml --debug

#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=75' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50' \
#               'data.params.train.params.arch=resnet50' 'data.params.validation.params.arch=resnet50' 'model.params.config.Loss.params.embed_dim=128'\
#               'model.params.config.Loss.params.n_warmup_iterations=0' 'model.params.config.Loss.params.embed_dim=128' 'model.params.config.Architecture.params.pretraining=None' \
#               'model.base_learning_rate=0.0005' 'model.weight_decay=0.0001' 'model.params.config.Loss.params.pred_dim=128' \
#               --savename debug_simsiam --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/simsiamloss.yaml --debug

#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=debug' 'lightning.trainer.max_epochs=75' \
#               'model.params.config.Architecture.target=architectures.vit_dino.Network' 'model.params.config.Architecture.params.arch=vit_small_patch8_224_dino_normalize' \
#               'data.params.train.params.arch=vit_small_patch8_224_dino' 'data.params.validation.params.arch=vit_small_patch8_224_dino'\
#               --savename debug_vits8_dino --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml --debug

#python main.py 'lightning.logger.params.group=debug' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=30' \
#               'model.params.config.Architecture.target=architectures.vit_dino.Network' 'model.params.config.Architecture.params.arch=vit_small_patch8_224_dino_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'data.params.train.params.ooDML_split_id=1' 'data.params.validation.params.ooDML_split_id=1' \
#               --savename debug_multi_gpu --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml


## ... multisimilarity loss
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=debug' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=36'\
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               'model.params.config.Architecture.target=architectures.perceiver.Network' 'model.params.config.Architecture.params.arch=perceiver_normalize' \
#               'data.params.train.params.arch=perceiver_normalize' 'data.params.validation.params.arch=perceiver_normalize' \
#              --savename baseline_perceiver_384_multisimilarity_cub200_130122 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=debug' 'lightning.trainer.max_epochs=1000' 'data.params.batch_size=112'\
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_normalize' \
#               'data.params.train.params.arch=resnet50_normalize' 'data.params.validation.params.arch=perceiver_normalize' 'model.params.config.Architecture.params.pretraining=None' \
#              --savename baseline_resnet50_scratch_128_multisimilarity_cars196_231221 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

## ... cross entropy loss
python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=debug' 'lightning.trainer.max_epochs=1000' 'data.params.batch_size=112'\
               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_normalize' \
               'data.params.train.params.arch=resnet50_normalize' 'data.params.validation.params.arch=resnet50_normalize' 'data.params.validation.params.train=true' 'model.params.config.Architecture.params.pretraining=None' \
               'model.params.config.Loss.params.embed_dim=384' 'model.params.config.Loss.params.loss_softmax_lr=0.005'\
              --savename baseline_resnet50_scratch_384_crossentropy_cub200_130122 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/cross_entropy.yaml