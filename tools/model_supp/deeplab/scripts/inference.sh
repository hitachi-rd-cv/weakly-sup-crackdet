
for NAME in aigle_deeplab_dil1; do
    python3 export_model.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --crop_size=545 \
    --crop_size=993 \
    --dataset=aigle \
    --checkpoint_path="./outputs/"$NAME"/train/model.ckpt-200000" \
    --export_path="./outputs/"$NAME"/train/trained.pb" \
    --num_classes=2

    mkdir -p "outputs/"$NAME"/sample_imgs/test_output"

    python3 ./scripts/inference.py \
    --graph_dir_str="outputs/"$NAME"/train/trained.pb" \
    --img_dir="datasets/data/aigle_detailed/test_img" \
    --out_dir="outputs/"$NAME"/sample_imgs/test_output"
done

for NAME in cfd_deeplab_dil1; do
    python3 export_model.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --crop_size=545 \
    --crop_size=993 \
    --dataset=cfd \
    --checkpoint_path="./outputs/"$NAME"/train/model.ckpt-200000" \
    --export_path="./outputs/"$NAME"/train/trained.pb" \
    --num_classes=2

    mkdir -p "outputs/"$NAME"/sample_imgs/test_output"

    python3 ./scripts/inference.py \
    --graph_dir_str="outputs/"$NAME"/train/trained.pb" \
    --img_dir="datasets/data/cfd_detailed/test_img" \
    --out_dir="outputs/"$NAME"/sample_imgs/test_output"
done

for NAME in dc_deeplab_dil1; do
    python3 export_model.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --crop_size=545 \
    --crop_size=993 \
    --dataset=deepcrack \
    --checkpoint_path="./outputs/"$NAME"/train/model.ckpt-200000" \
    --export_path="./outputs/"$NAME"/train/trained.pb" \
    --num_classes=2

    mkdir -p "outputs/"$NAME"/sample_imgs/test_output"

    python3 ./scripts/inference.py \
    --graph_dir_str="outputs/"$NAME"/train/trained.pb" \
    --img_dir="datasets/data/deepcrack_detailed/test_img" \
    --out_dir="outputs/"$NAME"/sample_imgs/test_output"
done
