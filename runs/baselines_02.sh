export GPU_TRAINING=5,
echo "gpus to use: ${GPU_TRAINING}"
export SAVE_PATH='/export/data2/tmilbich/PycharmProjects/Assessing_transfer_learning/Training_Results'
echo "SAVE_PATH is ${SAVE_PATH}"

### BASELINE CHECKS
# ... multisimilarity loss
python main.py --savename baseline_multisimilarity_cub200 --gpus ${GPU_TRAINING} --base configs/multisimloss.yaml
