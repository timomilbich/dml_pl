export GPU_TRAINING=4,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"
export BETTER_EXCEPTIONS=1

### DML runs based on ooDML splits

############################
# ... multisimilarity loss #
############################

# baseline (ViT-S-8-224-Dino) - SOP
#python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=40' 'data.params.batch_size=30' \
#               'model.params.config.Architecture.target=architectures.resnet50_dino.Network' 'model.params.config.Architecture.params.arch=resnet50_dino_normalize' \
#               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
#               'data.params.train.params.ooDML_split_id=1' 'data.params.validation.params.ooDML_split_id=1' \
#               --savename check_resnet50_dino_gpu2_so --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

python main.py 'model.params.config.Architecture.params.embed_dim=384' 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=40' 'data.params.batch_size=30' \
               'model.params.config.Architecture.target=architectures.resnet50_dino.Network' 'model.params.config.Architecture.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.params.arch=resnet50_dino_frozen_normalize' 'data.params.validation.params.arch=resnet50_dino_frozen_normalize' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'data.params.train.params.ooDML_split_id=1' 'data.params.validation.params.ooDML_split_id=1' \
               --savename check_resnet50_dino_gpu2_sop3 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
