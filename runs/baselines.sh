export GPU_TRAINING=2,
echo "GPUs: ${GPU_TRAINING}"
export EXP_PATH=/export/data/tmilbich/PycharmProjects/dml_pl/experiments/training_models
echo "EXP_PATH: ${EXP_PATH}"

### BASELINE CHECKS
# ... margin loss
python main.py --savename baseline_marginloss_cub200 --gpus ${GPU_TRAINING} --base configs/marginloss.yaml

# ... multisimilarity loss
#python main.py --savename baseline_multisimilarity_cub200 --exp_path ${EXP_PATH} --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml