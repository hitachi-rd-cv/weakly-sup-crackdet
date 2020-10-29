
import os
import re

import feature_extractor

from eval import calculatePR, getMaxF1Score
from utils import getAbsPaths, makeDir, convertFnameDomains


dataset_cands = ['aigle', 'cfd', 'dc']


def applyCV2Dir(dirname, infdir_name='sample_imgs/test_output',
                procdir_name='sample_imgs/cv_output',
                use_clahe=True, overwrite_file=False,
                data_reconstruct=False):
    log_names = os.listdir(dirname)
    for log_name_short in log_names:
        print('-' * 20)
        print(log_name_short)
        log_name = os.path.join(dirname, log_name_short)
        infdir = os.path.join(log_name, infdir_name)
        procdir = os.path.join(log_name, procdir_name)
        if not os.path.isdir(infdir):
            continue
        if (not overwrite_file) and os.path.isdir(procdir):
            continue
        print('processing %s...' % infdir)
        makeDir(procdir)

        dset_name = None
        name_split = os.path.basename(log_name).split('_')
        for cand in dataset_cands:
            dataset_str = name_split[0]
            dset_name = cand if (cand in dataset_str) else dset_name
        for cand in dataset_cands:
            dataset_str = name_split[-1]
            dset_name = cand if (cand in dataset_str) else dset_name

        indir = os.path.join('data/%s_detailed/original' % dset_name)
        fnames = [os.path.join(indir, fname.replace('_output', '')) for fname in os.listdir(infdir)]
        for fname in fnames:
            feature_extractor.main_dirinput(
                fname, procdir, infdir, use_clahe=use_clahe)


def evalDir(dirname, num_ste=100, infdir_name='sample_imgs/test_output',
            procdir_name='sample_imgs/cv_output', pix_tols=[0],
            overwrite_file=False):
    log_names = getAbsPaths(dirname)
    for log_name in log_names:
        preddir = os.path.join(log_name, infdir_name)
        procdir = os.path.join(log_name, procdir_name)
        if not os.path.isdir(preddir) or not os.path.isdir(procdir):
            continue
        if len(os.listdir(procdir)) == 0:
            continue

        dset_name = None
        for cand in dataset_cands:
            dataset_str = os.path.basename(log_name).split('_')[0]
            dset_name = cand if (cand in dataset_str) else dset_name
        for cand in dataset_cands:
            dataset_str = os.path.basename(log_name).split('_')[-1]
            dset_name = cand if (cand in dataset_str) else dset_name
        anno_dir = os.path.join('data/%s_detailed/gt' % dset_name)

        for infdir in [procdir, preddir]:
        # for infdir in [procdir]:
            for pix_tol in pix_tols:
                ofname = 'eval_%s_dil%d.txt' % (infdir.split('/')[-1], pix_tol)
                evalfname = os.path.join(log_name, ofname)
                if os.path.isfile(evalfname) and not overwrite_file:
                    continue

                print('evaluating %s...' % preddir)
                pred_thr = [float(i) / num_ste for i in range(num_ste)]
                eval_list = calculatePR(
                    anno_dir, infdir, pixel_tolerance=pix_tol,
                    postprocess_optimizer=None, adjust_thresh_per_img=False,
                    pred_threshes=pred_thr, pred_dir_sub=preddir)
                _, rec, _, prec = eval_list
                f1 = [2 * (r * p) / (r + p + 0.000001) for r, p in zip(rec, prec)]
                bf, br, bp, bindx = getMaxF1Score(rec, prec)

                with open(evalfname, 'w') as f:
                    f.write('threshold,fscore,recall,precision\n')
                    for th, fs, r, p in zip(pred_thr, f1, rec, prec):
                        f.write('%0.2f,%0.4f,%0.4f,%0.4f\n' % (th, fs, r, p))
                    f.write('\nbest:\n')
                    f.write('%0.2f,%0.4f,%0.4f,%0.4f\n' % (pred_thr[bindx], bf, br, bp))


if __name__ == '__main__':
    dirname = 'eval/results'
    infdir_name = 'sample_imgs/test_output'
    procdir_name = 'sample_imgs/cv_output'
    procdir_none_name = 'sample_imgs/cv_output_none'
    infdir_gf_name = 'sample_imgs/test_output_gf'
    procdir_gf_name = 'sample_imgs/cv_output_gf'

    overwrite_file = False

    print('applying cv models...')
    applyCV2Dir(dirname, infdir_name, procdir_none_name, use_clahe=False,
                overwrite_file=overwrite_file, data_reconstruct=False)

    print('evaluating the models')
    evalDir(dirname, 100, infdir_name, procdir_none_name, pix_tols=[0],
            overwrite_file=overwrite_file)
