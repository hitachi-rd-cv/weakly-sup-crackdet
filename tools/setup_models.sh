# DeepSegmentor
git clone http://github.com/yhlleo/DeepSegmentor models/deepcrack
cd models/deepcrack
git checkout 50440b52ddaf49cf54c2415e6b40646a7601c219
cd ../..

# DeepLab
git clone http://github.com/tensorflow/models models/deeplab
cd models/deeplab
git checkout 0a161121852ee5f34b939279d54b5d3e231ca501
cd ../..

PRETRAIN_DIR=models/deeplab/research/deeplab/datasets/pascal_voc_seg/init_models
mkdir -p ${PRETRAIN_DIR}

# format the models
python3 tools/setup_models.py --deepcrack --deeplab
