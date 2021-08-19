export GPU_TRAINING=0,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### DML runs based on ooDML splits

############################
# ... multisimilarity loss #
############################

# baseline (ViT-S-16-224-imagenet21k) - CARS196
python main.py 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=60' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
               'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.ooDML_split_id=1' 'data.params.validation.params.ooDML_split_id=1' \
               --savename msloss_vitSin21k_cars196_s1 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=60' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
               'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.ooDML_split_id=2' 'data.params.validation.params.ooDML_split_id=2' \
               --savename msloss_vitSin21k_cars196_s2 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=60' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
               'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.ooDML_split_id=3' 'data.params.validation.params.ooDML_split_id=3' \
               --savename msloss_vitSin21k_cars196_s3 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=60' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
               'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.ooDML_split_id=4' 'data.params.validation.params.ooDML_split_id=4' \
               --savename msloss_vitSin21k_cars196_s4 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=60' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
               'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.ooDML_split_id=5' 'data.params.validation.params.ooDML_split_id=5' \
               --savename msloss_vitSin21k_cars196_s5 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=60' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
               'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.ooDML_split_id=6' 'data.params.validation.params.ooDML_split_id=6' \
               --savename msloss_vitSin21k_cars196_s6 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py  'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=60' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
               'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.ooDML_split_id=7' 'data.params.validation.params.ooDML_split_id=7' \
               --savename msloss_vitSin21k_cars196_s7 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=60' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
               'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.ooDML_split_id=8' 'data.params.validation.params.ooDML_split_id=8' \
               --savename msloss_vitSin21k_cars196_s8 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
python main.py 'lightning.logger.params.group=ooDML_msloss' 'lightning.trainer.max_epochs=60' \
               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
               'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
               'data.params.train.params.ooDML_split_id=9' 'data.params.validation.params.ooDML_split_id=9' \
               --savename msloss_vitSin21k_cars196_s9 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml