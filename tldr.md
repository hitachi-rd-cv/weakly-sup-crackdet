# Tl;dr

If README is too long to read, here is the stripped down version that just lists the commands to execute to train DeepCrack and DeepLab models with Dilation-1 Annotation for Aigle, CFD, and DCD.



## Preliminaries

1. Clone the repo

   ```bash
   git clone https://github.com/hitachi-rd-cv/weakly-sup-crackdet.git
   ```

2. Download the RGB images and the Precise Annotations

   ```bash
   ./tools/download.sh
   ```

   - This script downloads images from different websites, and thus it may fail depending on the internet connections. Refer below for backup steps:
   - Fails on the ```curl``` call to https://www.irit.fr/~Sylvie.Chambon/AigleRN_GT.html
     - Go to the website, copy the html and place it under ```data/aigle_github/tmp.html```
   - Fails on the ```git clone``` call to CrackForest-dataset repo
     - download the zip file from the website and extract it under ```data/cfd_github```
   - Fails on the ```git clone``` call to DeepCrack repo
     - download the ```dataset/DeepCrack.zip``` file to ```data/deepcrack_github/dataset/DeepCrack.zip```

3. Download the low quality annotations from Zenodo repo, and copy them to the "data" directory

   ```bash
   cp -r ${weakly_sup_crackdet_dataset}/data/* ${git_repo_dir}/data
   ```

4. Download the pretrained Xception backbone network from the Zenodo repo and copy it to the main directory.

   ```bash
   cp -r ${pascal_voc_seg}/ tools/model_supp/deeplab/datasets/pascal_voc_seg/
   ```

5. Copy the RGB images from the `data/*_detailed` directories to the low quality annotations

   ```bash
   python tools/data_gen.py --fill
   ```

   - :warning: Make sure that all folders under ```data``` directory start with ```aigle_```, ```cfd_```, or ```deepcrack_```

6. Set up the repos for the crack detectors

   ```bash
   ./tools/setup_models.sh
   ```



## Training DeepCrack

1. Prepare the dataset

   ```bash
   python tools/data_gen.py --deepcrack --anno_type detailed rough rougher 1 2 3 4 --dataset_name all
   ```

2. Move to the DeepCrack directory

   ```bash
   cd models/deepcrack
   ```

3. Train the model. The script will train the model with synthetic annotation, dilation=1. Change the contents of the script accordingly to train with other annotations. The script also runs ```scripts/test_eval.sh``` at the end to format the training output.

   ```bash
   scripts/train_deepcrack.sh
   ```

   - :warning: The ```--name``` option for ```train.py``` specifies the output name, and it should be formatted as follows: ```${DATASET_NAME}_deepcrack_${ANNOTATION_TYPE}```.
   - After running ```scripts/test_eval.sh```, you should be able to see the outputs under ```checkpoints/${DATASET_NAME}_deepcrack_${ANNOTATION_TYPE}/sample_imgs/test_output```




## Training DeepLab V3+

1. Prepare the dataset

   ```bash
   python tools/data_gen.py --deeplab --anno_type detailed rough rougher 1 2 3 4 --dataset_name all
   ```

2. Move to the research directory of the DeepLab directory and set the environmental variable as per requested by the DeepLab repo.

   ```bash
   cd models/deeplab/research
   export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
   ```

3. Move to the DeepLab directory

   ```bash
   cd deeplab
   ```

4. Train the model. The script will train the model with synthetic annotation, dilation=1. Change the contents of the script accordingly to train with other annotations. The script also runs ```scripts/inference.sh``` at the end to format the training output.

   ```bash
   scripts/train.sh
   ```

   - :warning: The ```--train_logdir``` option for ```train.py``` specifies the output name, and it should be formatted as follows: ```${DATASET_NAME}/deeplab/${ANNOTATION_TYPE}```.



## Evaluation

1. Move the trained results from the crack detection repos to the main evaluation directory.

   ```bash
   # move to the main directory
   cd ${git_repo_dir}
   
   # create the eval/results directory
   mkdir eval/results
   
   # for DeepCrack
   mv models/deepcrack/checkpoints/* eval/results/
   
   # for DeepLab
   mv models/deeplab/research/deeplab/outputs/* eval/results/
   ```

2. Evaluate

   ```bash
   python eval/micro_eval.py
   ```

3. Aggregate the results. It outputs a csv file named `results.csv`

   ```bash
   python eval/gen_table.py
   ```
   
   - Contents of ```results.csv```
     - ```f1_0_test```: F-score of the output without the Micro Branch is applied
     - ```f1_0_cv_none```: F-score of the output with the Micro Branch is applied
