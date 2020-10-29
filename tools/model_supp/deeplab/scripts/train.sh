
# Trains DeepLab detector using datasets/data/tfrecords/aigle_dil1 as the training dataset
# Outputs the result under outputs/aigle_deeplab_dil1
# Since we are training with Aigle Dataset, set the --dataset as aigle
python3 train.py --logtostderr --train_split=train --model_variant=xception_65 \
--atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 \
--decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=2 \
--fine_tune_batch_norm=true \
--tf_initial_checkpoint="./datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/model.ckpt" \
--train_logdir="./outputs/aigle_deeplab_dil1/train" \
--dataset_dir="./datasets/data/tfrecords/aigle_dil1" \
--dataset=aigle --training_number_of_steps=200000 \
--class_weight=100 --base_learning_rate 0.005

# Trains DeepLab detector using datasets/data/tfrecords/cfd_dil1 as the training dataset
# Outputs the result under outputs/cfd_deeplab_dil1
# Since we are training with CFD Dataset, set the --dataset as cfd
python3 train.py --logtostderr --train_split=train --model_variant=xception_65 \
--atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 \
--decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=2 \
--fine_tune_batch_norm=true \
--tf_initial_checkpoint="./datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/model.ckpt" \
--train_logdir="./outputs/cfd_deeplab_dil1/train" \
--dataset_dir="./datasets/data/tfrecords/cfd_dil1" \
--dataset=CFD --training_number_of_steps=200000 \
--class_weight=100 --base_learning_rate 0.005

# Trains DeepLab detector using datasets/data/tfrecords/deepcrack_dil1 as the training dataset
# Outputs the result under outputs/dc_deeplab_dil1
# Since we are training with DeepCrack Dataset, set the --dataset as deepcrack
python3 train.py --logtostderr --train_split=train --model_variant=xception_65 \
--atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 \
--decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=2 \
--fine_tune_batch_norm=true \
--tf_initial_checkpoint="./datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/model.ckpt" \
--train_logdir="./outputs/dc_deeplab_dil1/train" \
--dataset_dir="./datasets/data/tfrecords/deepcrack_dil1" \
--dataset=deepcrack --training_number_of_steps=200000 \
--class_weight=100 --base_learning_rate 0.005

./scripts/inference.sh