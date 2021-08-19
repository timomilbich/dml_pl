export GPU_TRAINING=7,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### BASELINE CHECKS
# ... margin loss
python main.py 'model.params.config.Architecture.params.embed_dim=128' 'lightning.logger.params.group=baselines' 'lightning.trainer.max_epochs=100' \
               'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
               'model.params.config.Architecture.target=architectures.resnet50.Network' 'model.params.config.Architecture.params.arch=resnet50_frozen_normalize' \
               'data.params.train.params.arch=resnet50_frozen_normalize' 'data.params.validation.params.arch=resnet50_frozen_normalize' \
              --savename baseline_marginloss_sop_CHECK --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

## ... multisimilarity loss
#python main.py 'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
#              --savename baseline_multisimilarity_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
#
## ... proxyanchor
#python main.py 'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
#              --savename baseline_proxyanchor_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml



