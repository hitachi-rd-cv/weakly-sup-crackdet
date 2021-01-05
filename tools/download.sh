# Aigle
mkdir data/aigle_github
mkdir data/aigle_github/original
mkdir data/aigle_github/gt
curl https://www.irit.fr/~Sylvie.Chambon/AigleRN_GT.html -o data/aigle_github/tmp.html
grep -oh 'IMAGES/AIGLE_RN/.*png' data/aigle_github/tmp.html > data/aigle_github/tmp.txt
awk '$0="https://www.irit.fr/~Sylvie.Chambon/TITS/"$0' data/aigle_github/tmp.txt | wget -P data/aigle_github/original -i-
grep -oh 'GROUND_TRUTH/AIGLE_RN/.*png' data/aigle_github/tmp.html > data/aigle_github/tmp.txt
awk '$0="https://www.irit.fr/~Sylvie.Chambon/TITS/"$0' data/aigle_github/tmp.txt | wget -P data/aigle_github/gt -i-

# # Crack Forest Dataset
# git clone https://github.com/cuilimeng/CrackForest-dataset data/cfd_github

# # DeepCrack Dataset
# git clone https://github.com/yhlleo/DeepCrack.git data/deepcrack_github
unzip data/deepcrack_github/dataset/DeepCrack.zip -d data/deepcrack_github/dataset

# format the images
python3 tools/download.py

# # clean up
# # rm -rf data/*_github
