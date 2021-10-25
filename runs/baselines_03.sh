export GPU_TRAINING=5,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### BASELINE CHECKS
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=120' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' 'data.params.batch_size=56' \
#               'model.params.config.Loss.params.loss_multisimilarity_beta_const=True' 'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=max_min' \
#               --savename debug_msloss_r50d384_gpu2_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=120' \
#               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' 'data.params.batch_size=112' \
#               --savename debug_msloss_orig_r50d384_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=100' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'model.params.config.Loss.params.loss_multisimilarity_beta=0.5' 'model.params.config.Loss.params.loss_multisimilarity_beta_lr=0.0005' \
               'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=max_min' \
               --savename margin_msloss_r50d384_beta0.5_0.0005_maxmin_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=margin_multisimilarity' 'lightning.trainer.max_epochs=120' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' 'data.params.batch_size=112' \
               'model.params.config.Loss.params.loss_multisimilarity_beta_const=True' 'model.params.config.Loss.params.loss_multisimilarity_sampling_mode=max_min' \
               --savename debug_msloss_r50d384_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/margin_multisimloss2.yaml