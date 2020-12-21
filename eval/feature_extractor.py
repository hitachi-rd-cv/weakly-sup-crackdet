import os
import cv2

import numpy as np

from utils import rescaleTo1, rescaleTo255, min0max1scale, \
    convertFnameDomains, makeDir, getCorrectImgExt


def intensityScore2(src_img, front_mask=None, use_clahe=True):
    in_img_1ch = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed_img = clahe.apply(in_img_1ch)
    else:
        processed_img = in_img_1ch

    bw_img = rescaleTo1(processed_img)
    if front_mask is None:
        score_map = 1 - min0max1scale(bw_img)
    else:
        bw_roi = bw_img[front_mask]
        score_map = 1 - min0max1scale(bw_img, bw_roi)

    return np.expand_dims(score_map, axis=-1)


def main_dirinput(fname, outdir, rough_inf_dir, mask_thresh=0.6,
                  use_clahe=True):
    _, img_name = os.path.dirname(fname), os.path.basename(fname)
    output_fname = convertFnameDomains('original', 'output', img_name)
    pred_fname = os.path.join(rough_inf_dir, output_fname)
    out_fname = os.path.join(outdir, output_fname)

    main(fname, out_fname, pred_fname, mask_thresh, use_clahe)


def main(fname, out_fname, pred_fname, mask_thresh=0.6,
         use_clahe=True):
    in_img = cv2.imread(getCorrectImgExt(fname.replace('dc', 'deepcrack')))
    in_img_1ch = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)

    pred_img = cv2.imread(pred_fname, cv2.IMREAD_GRAYSCALE)

    intensity_map = intensityScore2(in_img, None, use_clahe=use_clahe)
    final_map = intensity_map * rescaleTo1(pred_img, threeD=True)
    out_img = rescaleTo255(final_map)

    if out_fname is None:
        return out_img
    else:
        cv2.imwrite(out_fname, out_img)

