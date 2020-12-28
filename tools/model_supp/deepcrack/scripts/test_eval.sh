

for MODEL in aigle_deepcrack_dil1 ; do
    scripts/test_deepcrack.sh 0 $MODEL ./datasets/aigle_detailed ./checkpoints/
    python3 tools/guided_filter.py --results_dir checkpoints --model_name $MODEL
    python3 scripts/output_format.py --log_dir checkpoints/$MODEL
done

for MODEL in cfd_deepcrack_dil1 ; do
    scripts/test_deepcrack.sh 0 $MODEL ./datasets/cfd_detailed ./checkpoints/
    python3 tools/guided_filter.py --results_dir checkpoints --model_name $MODEL
    python3 scripts/output_format.py --log_dir checkpoints/$MODEL
done

for MODEL in dc_deepcrack_dil1 ; do
    scripts/test_deepcrack.sh 0 $MODEL ./datasets/deepcrack_detailed ./checkpoints/
    python3 tools/guided_filter.py --results_dir checkpoints --model_name $MODEL
    python3 scripts/output_format.py --log_dir checkpoints/$MODEL
done
