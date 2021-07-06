export GPU_TRAINING=7,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### BASELINE CHECKS
# ... margin loss
python main.py 'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
              --savename baseline_marginloss_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

# ... multisimilarity loss
python main.py 'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
              --savename baseline_multisimilarity_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

# ... proxyanchor
python main.py 'data.params.train.target=data.SOP.DATA' 'data.params.validation.target=data.SOP.DATA' \
              --savename baseline_proxyanchor_sop --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml
