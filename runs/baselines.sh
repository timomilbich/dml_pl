export GPU_TRAINING=3,
echo "gpus to use: ${GPU_TRAINING}"
export SAVE_PATH='/export/data2/tmilbich/PycharmProjects/Assessing_transfer_learning/Training_Results'
echo "SAVE_PATH is ${SAVE_PATH}"


python main.py --savename baseline_marginloss_cub200 --gpus ${GPU_TRAINING} --base configs/cub200.yaml

