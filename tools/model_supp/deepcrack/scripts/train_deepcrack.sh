GPU_IDS=0

DATAROOT=./datasets/DeepCrack
MODEL=deepcrack
DATASET_MODE=deepcrack

BATCH_SIZE=1
NUM_CLASSES=1
LOAD_WIDTH_AIGLE=320
LOAD_HEIGHT_AIGLE=480
LOAD_WIDTH_CFD=480
LOAD_HEIGHT_CFD=320
LOAD_WIDTH_DEEPCRACK=256
LOAD_HEIGHT_DEEPCRACK=256

NORM=batch
NITER=400
NITER_DECAY=300

# Trains DeepCrack detector using ./datasets/aigle_dil1 as the training dataset
# Outputs the result under checkpoints/aigle_deepcrack_dil1
# Since we are training with Aigle, set the --load_width
# and --load_height parameters to LOAD_WIDTH_AIGLE and LOAD_HEIGHT_AIGLE
python3 train.py --dataroot ./datasets/aigle_dil1 --name aigle_deepcrack_dil1 \
--model ${MODEL} --dataset_mode ${DATASET_MODE} --gpu_ids ${GPU_IDS} \
--niter ${NITER} --niter_decay ${NITER_DECAY} --batch_size ${BATCH_SIZE} \
--num_classes ${NUM_CLASSES} --norm ${NORM} --lr_decay_iters 175 --lr_policy step \
--load_width ${LOAD_WIDTH_AIGLE} --load_height ${LOAD_HEIGHT_AIGLE} --no_flip 0 --display_id 0 \
--use_augment 1 --lr 0.001 --weight 3e-1

# Trains DeepCrack detector using ./datasets/cfd_dil1 as the training dataset
# Outputs the result under checkpoints/cfd_deepcrack_dil1
# Since we are training with CFD, set the --load_width
# and --load_height parameters to LOAD_WIDTH_CFD and LOAD_HEIGHT_CFD
python3 train.py --dataroot ./datasets/cfd_dil1 --name cfd_deepcrack_dil1 \
--model ${MODEL} --dataset_mode ${DATASET_MODE} --gpu_ids ${GPU_IDS} \
--niter ${NITER} --niter_decay ${NITER_DECAY} --batch_size ${BATCH_SIZE} \
--num_classes ${NUM_CLASSES} --norm ${NORM} --lr_decay_iters 175 --lr_policy step \
--load_width ${LOAD_WIDTH_CFD} --load_height ${LOAD_HEIGHT_CFD} --no_flip 0 --display_id 0 \
--use_augment 1 --lr 0.001 --weight 3e-1

# Trains DeepCrack detector using ./datasets/deepcrack_dil1 as the training dataset
# Outputs the result under checkpoints/dc_deepcrack_dil1
# Since we are training with DeepCrack Dataset, set the --load_width
# and --load_height parameters to LOAD_WIDTH_DEEPCRACK and LOAD_HEIGHT_DEEPCRACK
python3 train.py --dataroot ./datasets/deepcrack_dil1 --name dc_deepcrack_dil1 \
--model ${MODEL} --dataset_mode ${DATASET_MODE} --gpu_ids ${GPU_IDS} \
--niter ${NITER} --niter_decay ${NITER_DECAY} --batch_size ${BATCH_SIZE} \
--num_classes ${NUM_CLASSES} --norm ${NORM} --lr_decay_iters 175 --lr_policy step \
--load_width ${LOAD_WIDTH_DEEPCRACK} --load_height ${LOAD_HEIGHT_DEEPCRACK} --no_flip 0 --display_id 0 \
--use_augment 1 --lr 0.001 --weight 3e-1





# evaluate
./scripts/test_eval.sh
