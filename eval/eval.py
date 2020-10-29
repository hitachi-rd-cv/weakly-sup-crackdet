
import os
import cv2

import numpy as np

from utils import rescaleTo255, convertFnameDomains

from multiprocessing import Pool

POOL_WORKER_COUNT = 10
kernel3 = np.ones((3, 3), np.uint8)


# evaluation functions
def getMaxF1Score(recs, precs):
    recs = np.asarray(recs)
    precs = np.asarray(precs)

    f1_score = 2 * (recs * precs) / (recs + precs + 0.000001)
    indx = np.argmax(f1_score)

    return np.max(f1_score), recs[indx], precs[indx], indx


def evalImg(gt_img, pred_mask, pixel_tolerance=1):
    eps = 0.000001
    annotated_mask = (gt_img[:, :, 1] == 255) | (gt_img[:, :, 2] == 255)
    gt_mask = (gt_img[:, :, 1] == 255)
    gt_mask_not = np.logical_not(gt_mask) & annotated_mask
    pred_mask = pred_mask & annotated_mask
    pred_mask_not = np.logical_not(pred_mask) & annotated_mask

    num_gt_pix = float(np.sum(gt_mask)) + eps
    num_pred_pix = float(np.sum(pred_mask)) + eps
    num_elements = float(np.sum(annotated_mask))

    true_positive = np.sum(gt_mask & pred_mask)
    true_negative = np.sum(gt_mask_not & pred_mask_not)
    false_positive = np.sum(gt_mask_not & pred_mask)

    accuracy = (true_positive + true_negative) / num_elements
    precision = true_positive / num_pred_pix
    recall = true_positive / num_gt_pix

    spe_den = num_elements - num_gt_pix
    minus_specificity = false_positive / spe_den

    gt_img = rescaleTo255(gt_mask)
    pred_img = rescaleTo255(pred_mask)

    gt_dil = cv2.dilate(gt_img, kernel3, iterations=pixel_tolerance) > 0
    gt_neighbor = gt_dil & annotated_mask
    pred_dil = cv2.dilate(pred_img, kernel3, iterations=pixel_tolerance) > 0
    pred_neighbor = pred_dil & annotated_mask

    precision_dilated_tp = np.sum(gt_neighbor & pred_mask)
    # recall_dilated_tp = np.sum(gt_mask & pred_neighbor)

    gt_neighbor_not = np.logical_not(gt_neighbor) & annotated_mask
    crack_hits = np.sum(gt_mask & pred_neighbor)
    non_crack_hits = np.sum(gt_neighbor_not & pred_mask_not)
    accuracy_dilated = ((non_crack_hits + crack_hits)
                        / float(num_gt_pix + np.sum(gt_neighbor_not) + eps))

    precision_dilated = precision_dilated_tp / num_pred_pix
    recall_dilated = np.sum(gt_mask & pred_neighbor) / num_gt_pix
    minus_specificity_dilated = np.sum(gt_neighbor_not & pred_mask) / spe_den

    return accuracy, accuracy_dilated, precision, precision_dilated, \
        recall, recall_dilated, minus_specificity, minus_specificity_dilated


def specialMask(pred_img, thresh, refine_img):
    ori_mask = (pred_img > (255 * thresh))

    if refine_img is None:
        return ori_mask

    _, refine_mask = cv2.threshold(refine_img, 255 * thresh, 255, cv2.THRESH_BINARY)
    h, w = pred_img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    for x in range(w):
        for y in range(h):
            if ori_mask[y, x] and not mask[y+1, x+1]:
                cv2.floodFill(refine_mask, mask, (x, y), 255,
                              flags=(8 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY))
    mask = mask[1:-1, 1:-1]
    mask = (mask == 255)

    return mask


def evalBulk(ano_imgs, pred_imgs, pixel_tolerance=1,
             pred_thresh=0.5,
             postprocess_optimizer=None, pred_fnames=None, pred_imgs_sub=list()):
    """
    Evaluates images

    Args:
            :ano_imgs (img list):	List of all annotation images
            :pred_imgs (img list):	List of all prediction images
            :pixel_tolerance (int):	Number of pixels a prediction needs to be within the ground truth for it to be deemed correct prediction
            :pred_thresh (float):	Threshold value used to binarize the inference image.
            :postprocess_optimizer (str):	Defines what values to optimize when doing postprocess optimization. Available: ``pr``, ``pr2``, ``pr3``, ``pr6``, ``prminus``. ``None`` for no postprocess optimizing.

    Returns:
            :stat_scores_mean (numpy array):	Mean value of stats for every image evaluated. Below is a list of stats stored in each index.\n
                                    0: accuracy\n
                                    1: accuracy_dilated\n
                                    2: precision\n
                                    3: precision_dilated\n
                                    4: recall\n
                                    5: recall_dilated\n
                                    6: minus_specificity (=1-specificity)\n
                                    7: minus_specificity_dilated\n
                                    8: crack_coverage_accuracy\n
                                    dilated versions use ``pixel_tolerance``\n
    """
    stat_scores = []
    for annotated_img, pred_img, pred_img_sub in zip(ano_imgs, pred_imgs, pred_imgs_sub):
        if postprocess_optimizer is None:
            pred_mask = specialMask(pred_img, pred_thresh, pred_img_sub)

            stat_score = evalImg(
                annotated_img, pred_mask, pixel_tolerance=pixel_tolerance)

        stat_scores.append(list(stat_score))

    stat_scores = np.stack(stat_scores)
    return stat_scores


def cal_prf_metrics(gt_list, pred_list, pred_thresh, pred_imgs_sub):
    stat_scores = []

    for pred, gt, pred_img_sub in zip(pred_list, gt_list, pred_imgs_sub):
        if len(gt.shape) == 3:
            gt = gt[:, :, 1]
        if len(pred.shape) == 3:
            pred = pred[:, :, 0]

        gt_img = (gt / 255).astype('uint8')
        pred_img = specialMask(pred, pred_thresh, None).astype('uint8')

        # calculate each image
        tp, fp, fn = get_statistics(pred_img, gt_img)

        p_acc = 1.0 if (tp == 0 and fp == 0) else tp / (tp + fp + 1e-10)
        r_acc = tp / (tp + fn + 1e-10)

        stat_scores.append([0, 0, p_acc, p_acc, r_acc, r_acc, 0, 0, 0])


    stat_scores = np.stack(stat_scores)
    return stat_scores


def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return [tp, fp, fn]


def evalBulkWrapper(data_pack):
    indx, ano_imgs, pred_imgs, pixel_tolerance, pred_thresh, \
    postprocess_optimizer, pred_fnames, pred_imgs_sub = data_pack

    if pixel_tolerance == 0:
        score = cal_prf_metrics(ano_imgs, pred_imgs, pred_thresh,
                                pred_imgs_sub)
    else:
        score = evalBulk(
            ano_imgs, pred_imgs, pixel_tolerance, pred_thresh,
            postprocess_optimizer, pred_fnames, pred_imgs_sub)

    return indx, score


def eval(gt_dir, pred_dir, pixel_tolerance=1,
         postprocess_optimizer=None, pred_fnames=None,
         pred_threshes=None, adjust_thresh_per_img=False, pred_dir_sub=None):
    if pred_threshes is None:
        pred_threshes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                         0.8, 0.9, 0.95, 0.99, 0.995]

    # pre-load all images
    ano_imgs, pred_imgs, pred_imgs_sub = list(), list(), list()
    pred_fnames = [fn for fn in os.listdir(pred_dir) if ('heatmap' not in fn)]
    pred_fnames.sort()
    for pred_fname in pred_fnames:
        annotated_fname = convertFnameDomains('output', 'gt',
                                              os.path.basename(pred_fname))
        ano_imgs.append(cv2.imread(os.path.join(gt_dir, annotated_fname)))
        pred_imgs.append(cv2.imread(os.path.join(pred_dir, pred_fname),
                                    cv2.IMREAD_GRAYSCALE))

        if pred_dir_sub is None:
            pred_img_sub = None
        else:
            pred_img_sub = cv2.imread(os.path.join(pred_dir_sub, pred_fname),
                                      cv2.IMREAD_GRAYSCALE)
        pred_imgs_sub.append(pred_img_sub)

    mapList = []
    stat_scores = [None] * len(pred_threshes)
    for indx, pred_thresh in enumerate(pred_threshes):
        mapList.append((indx, ano_imgs, pred_imgs, pixel_tolerance, pred_thresh,
                        postprocess_optimizer, pred_fnames, pred_imgs_sub))

    p = Pool(POOL_WORKER_COUNT)
    score_vals = map(evalBulkWrapper, mapList)
    p.close()
    p.join()

    for score_indx, score in score_vals:
        stat_scores[score_indx] = score.tolist()
    score_3d = np.stack(stat_scores)

    if adjust_thresh_per_img:
        p, r = score_3d[:, :, 3], score_3d[:, :, 5]
        f = 2 * p * r / (p + r + 0.000001)
        findxs = np.argmax(f, axis=0)
        score_reduced = [score_3d[val, i, :] for i, val in enumerate(findxs)]
        score_reduced = np.stack(score_reduced)

        tt = [pred_threshes[indx] for indx in findxs]
        names = os.listdir(gt_dir)
        for ttt, name in zip(tt, names):
            print(ttt, name)

        score_reduced = np.mean(score_reduced, axis=0, keepdims=True)
    else:
        score_reduced = np.mean(score_3d, axis=1)

    return score_reduced


def stat_extractor(stat_keys, stat_scores):
    EVAL_KEYS = ['accuracy', 'accuracy_dilated',
                 'precision', 'precision_dilated',
                 'recall', 'recall_dilated', 'minus_specificity',
                 'minus_specificity_dilated', 'crack_coverage_accuracy']
    stat_ids = [EVAL_KEYS.index(sk) for sk in stat_keys]

    result_list = []
    for indx in stat_ids:
        result_list.append(list(stat_scores[:, indx]))

    return result_list


def calculatePR(gt_dir, pred_dir, pixel_tolerance=1,
                postprocess_optimizer=None, pred_fnames=None,
                pred_threshes=None, adjust_thresh_per_img=False,
                pred_dir_sub=None):
    """
    Evaluates images and outputs precision & recall

    Args:
            :gt_dir (str):  Directory where ground truth files are (``*_annotated.png``)
            :pred_dir (str):    Directory where predicted files are (``*_output.png``)
            :pixel_tolerance (int): Number of pixels a prediction needs to be within the ground truth for it to be deemed correct prediction
            :postprocess_optimizer (str):   Defines what values to optimize when doing postprocess optimization. Available: ``pr``, ``pr2``, ``pr3``, ``pr6``, ``prminus``. ``None`` for no postprocess optimizing.
            :pred_fnames (string list):   Names of the files to evaluate
            :pred_threshes (float list):   Threshold value used to binarize the inference image
            :adjust_thresh_per_img (bool):   True if binarization threshold should be adjusted for each image
            :pred_dir_sub:  used for refining the thresholded prediction

    Returns:
            :result_list (2D list of float): A list containing 4 lists. List #1: recall, List #2: recall_dilated, List #3: precision, List #4: precision_dilated. Each list contains the evaluation result for various threshold value. Use ``list2d2str()`` to view as a string
    """
    stat_keys = ['recall', 'recall_dilated', 'precision', 'precision_dilated']

    stat_scores = eval(
        gt_dir, pred_dir, pixel_tolerance,
        postprocess_optimizer, pred_fnames, pred_threshes,
        adjust_thresh_per_img, pred_dir_sub)
    result_list = stat_extractor(stat_keys, stat_scores)

    return result_list
