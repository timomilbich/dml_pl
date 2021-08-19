export GPU_TRAINING=0,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### DML runs using ViT backbone

####################
# ... margin loss  #
####################
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename baseline_marginloss_vit_small_patch16_224_384f_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename baseline_marginloss_vit_small_patch16_224_in21k_384f_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
#               --savename baseline_marginloss_vit_small_patch16_224_384f_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=60' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
#               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
#               --savename baseline_marginloss_vit_small_patch16_224_in21k_384f_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

# add embedding layer
#python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               --savename baseline_marginloss_vit_s_p16_224_128e_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize_nolayernorm' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               --savename baseline_marginloss_vit_s_p16_224_128eNoLN_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml\

## freeze layer norm parameter
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize_nolayernorm_frozen' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               --savename baseline_marginloss_vit_s_p16_224_384fNoLNfrozen_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

############################
# ... multisimilarity loss #
############################

#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename baseline_multisimilarity_vit_small_patch16_224_384f_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename baseline_multisimilarity_vit_small_patch16_224_in21k_384f_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

###### vary weight decay
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=25' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'model.weight_decay=1e-5' 'lightning.logger.params.group=search_wd'\
#               --savename msloss_vit_s_p16_224_384f_cub200_wd_1e-5 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=25' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'model.weight_decay=1e-4' 'lightning.logger.params.group=search_wd'\
#               --savename msloss_vit_s_p16_224_384f_cub200_wd_1e-4 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=25' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'model.weight_decay=1e-3' 'lightning.logger.params.group=search_wd'\
#               --savename msloss_vit_s_p16_224_384f_cub200_wd_1e-3 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

# add embedding layer
#python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               --savename baseline_msloss_vit_s_p16_224_128e_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize_nolayernorm' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               --savename baseline_msloss_vit_s_p16_224_128eNoLN_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

# test ensemble strategies
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=ensembling_SA' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize_add_ens' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               --savename add_ens_msloss_vit_s_p16_224_384f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=ensembling_SA' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize_pool_ens' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               --savename pool_ens_msloss_vit_s_p16_224_384f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=ensembling_SA' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize_att_ens' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' \
#               --savename att_ens_msloss_vit_s_p16_224_384f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=ensembling_SA' 'lightning.trainer.max_epochs=60' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize_att_ens' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               --savename att_ens_msloss_vit_s_p16_224_384f_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

#########################
## ... proxyanchor loss #
#########################

#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=25' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'model.params.config.Loss.params.embed_dim=384' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               --savename baseline_proxyanchor_vit_small_patch16_224_384f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.embed_dim=384' \
#               --savename baseline_proxyanchor_vit_small_patch16_224_384f_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=25' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
#               'model.params.config.Loss.params.embed_dim=384' \
#               --savename baseline_proxyanchor_vit_small_patch16_224_in21k_384f_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_in21k_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_in21k_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_in21k_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.embed_dim=384' \
#               --savename baseline_proxyanchor_vit_small_patch16_224_in21k_384f_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml

# search schedule

#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=search_schedule' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.params.config.Loss.params.embed_dim=384' \
#               --savename baseline_proxyanchor_vit_s_p16_224_384f_cars196_1e-5_0.0004 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=search_schedule' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.base_learning_rate=1e-5' 'model.weight_decay=0.0004' 'model.gamma=0.3' 'model.tau=[50,70]' \
#               --savename proxyanchor_vit_s_p16_224_384f_cars196_1e-5_0.0004_0.3_20-50 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=search_schedule' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.base_learning_rate=1e-5' 'model.weight_decay=0.0004' 'model.gamma=0.3' 'model.tau=[40,60]' \
#               --savename proxyanchor_vit_s_p16_224_384f_cars196_1e-5_0.0004_0.3_40-60 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml
#python -W ignore main.py 'model.params.config.Architecture.params.embed_dim=-1' 'lightning.logger.params.group=search_schedule' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
#               'model.base_learning_rate=1e-5' 'model.weight_decay=0.0004' 'model.gamma=0.3' 'model.tau=[30,65]' \
#               --savename proxyanchor_vit_s_p16_224_384f_cars196_1e-5_0.0004_0.3_30-60 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml

# add embedding layer
#python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'model.params.config.Loss.params.embed_dim=128' \
#               --savename baseline_proxy_vit_s_p16_224_128e_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml
#python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=vit' 'lightning.trainer.max_epochs=100' 'data.params.batch_size=112' \
#               'model.params.config.Architecture.target=architectures.vit.Network' 'model.params.config.Architecture.params.arch=vit_small_patch16_224_normalize_nolayernorm' \
#               'data.params.train.params.arch=vit_small_patch16_224_normalize' 'data.params.validation.params.arch=vit_small_patch16_224_normalize' \
#               'data.params.train.target=data.CUB200.DATA' 'data.params.validation.target=data.CUB200.DATA' 'model.params.config.Loss.params.embed_dim=128' \
#               --savename baseline_proxy_vit_s_p16_224_128eNoLN_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml