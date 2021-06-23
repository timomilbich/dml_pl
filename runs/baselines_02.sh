export GPU_TRAINING=2,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH='/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models'
echo "EXP_PATH: ${EXP_PATH}"

### BASELINE CHECKS
# ... margin loss
python main.py 'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
            --savename baseline_marginloss_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

# ... multisimilarity loss
python main.py 'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
            --savename baseline_multisimilarity_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml

# ... proxyanchor
python main.py 'data.params.train.target=data.CARS196.DATA' 'data.params.validation.target=data.CARS196.DATA' \
            --savename baseline_proxyanchor_cars196 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/proxyanchorloss.yaml
