import os
import cv2
import scipy
import shutil
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from sklearn.metrics import precision_score, recall_score

import albumentations as albu
import albumentations.augmentations.transforms as trans
from albumentations import Compose

from utils import convertFnameDomains, makeDir


def genDataset(dirname, out_dirname, procfn, **kwargs):
    for dname in ['original', 'data_split.json']:
        try:
            shutil.copytree(os.path.join(dirname, dname),
                            os.path.join(out_dirname, dname))
        except:
            shutil.copy(os.path.join(dirname, dname),
                        os.path.join(out_dirname, dname))

    oridir = os.path.join(dirname, 'original')
    gtdir = os.path.join(dirname, 'gt')
    gtoutdir = os.path.join(out_dirname, 'gt')
    makeDir([gtoutdir])

    fnames = [fname for fname in os.listdir(oridir)]
    fnames.sort()
    for img_name in fnames:
        gt_fname = convertFnameDomains('original', 'ground_truth', img_name)
        pred_fname = os.path.join(gtdir, gt_fname)
        out_fname = os.path.join(gtoutdir, gt_fname)

        macro_img = cv2.imread(pred_fname)
        if macro_img is None:
            continue

        bin_img = procfn(macro_img, img_name, kwargs)
        anno_fname = convertFnameDomains('original', 'annotated', img_name)
        cv2.imwrite(out_fname, bin_img)


def procDistort(macro_img, img_name, k_dicts):
    upper_limit, lower_limit = k_dicts['upper_lim'], k_dicts['lower_lim']

    # apply dilation
    dil_amount = k_dicts['dil_amount']

    dil_kernel = np.ones((3, 3), np.uint8)
    dil_img = cv2.dilate(macro_img, dil_kernel, iterations=dil_amount)

    # apply random transformations
    alpha = 10
    count = 0
    alpha_lower, alpha_upper = 10, 10000
    while True:
        if count == 5:
            alpha += 10 if (rec > upper_limit) else -10
            count = 0
        elif ('random' in k_dicts) and (k_dicts['random']):
            if alpha_lower >= alpha_upper:
                alpha_lower, alpha_upper = 10, 10000
            alpha = np.random.randint(alpha_lower, alpha_upper)
        else:
            count += 1
        # print(count, alpha, alpha_lower, alpha_upper)

        aug_list = list()
        aug_list.append(
            trans.ElasticTransform(
                alpha=alpha, sigma=12, alpha_affine=0.2,
                interpolation=cv2.INTER_NEAREST, p=1))

        com = Compose(aug_list)
        aug_img = com(image=dil_img)['image']

        gt = (macro_img[:, :, 1]).flatten() / 255
        pred = (aug_img[:, :, 1]).flatten() / 255
        if sum(gt) == 0:
            rec = 0
            break

        rec = recall_score(gt, pred)
        if rec > upper_limit:
            alpha_lower = alpha
        elif rec < lower_limit:
            alpha_upper = alpha
        else:
            break

    bin_img = aug_img

    prec = precision_score(gt, pred)
    print('%s, precision:%0.3f, recall:%0.3f, alpha:%0.3f' \
          % (img_name, prec, rec, alpha))

    return bin_img


def main(dset_name, dil_amount):
    original_dir = 'data/%s_detailed/' % dset_name
    dataset_name = 'data/%s_dil%d' % (dset_name, dil_amount)
    genDataset(
        original_dir, dataset_name, procfn=procDistort, dil_amount=dil_amount,
        random=True, upper_lim=0.975, lower_lim=0.925)
