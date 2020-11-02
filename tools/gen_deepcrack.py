
import os
import cv2
import json
import math

import numpy as np


def getPatchedList(src_img, patch_h, patch_w, num_patches_h, num_patches_w):
    if len(src_img.shape) == 4:
        src_h, src_w = src_img.shape[1:3]
    elif len(src_img.shape) == 3:
        src_h, src_w = src_img.shape[:2]

    patch_h, patch_w = int(patch_h), int(patch_w)

    h_stride = patch_h if (num_patches_h == 1) else int(
        np.floor((src_h - float(patch_h)) / (num_patches_h - 1)))
    w_stride = patch_w if (num_patches_w == 1) else int(
        np.floor((src_w - float(patch_w)) / (num_patches_w - 1)))

    imgs = []
    for y in range(num_patches_h):
        y_start = h_stride * y
        for x in range(num_patches_w):
            x_start = w_stride * x

            if len(src_img.shape) == 4:
                img_region = src_img[:, y_start:(y_start + patch_h),
                                     x_start:(x_start + patch_w), :]
            elif len(src_img.shape) == 3:
                img_region = src_img[y_start:(y_start + patch_h),
                                     x_start:(x_start + patch_w), :]

            imgs.append(img_region)

    return imgs, h_stride, w_stride


def makeDir(dirname):
    # create a folder
    if type(dirname) == str:
        dirname = [dirname]

    for d in dirname:
        try:
            os.makedirs(d)
        except:
            pass


def main(ori_logdir, out_logdir, target_sz):
    in_imgdir = os.path.join(ori_logdir, 'original')
    gt_imgdir = os.path.join(ori_logdir, 'gt')
    data_split = os.path.join(ori_logdir, 'data_split.json')

    with open(data_split, 'r') as f:
        data_split_d = json.load(f)

    img_outdir = os.path.join(out_logdir, 'train_img')
    lbl_outdir = os.path.join(out_logdir, 'train_lab')
    testimg_outdir = os.path.join(out_logdir, 'test_img')
    testlbl_outdir = os.path.join(out_logdir, 'test_lab')
    makeDir([out_logdir, img_outdir, lbl_outdir,
             testimg_outdir, testlbl_outdir])

    dir_map = [(in_imgdir, img_outdir, data_split_d['train']['input'], False, True),
               (gt_imgdir, lbl_outdir, data_split_d['train']['gt'], True, True),
               (in_imgdir, testimg_outdir, data_split_d['test']['input'], False, False),
               (gt_imgdir, testlbl_outdir, data_split_d['test']['gt'], True, False)]

    for fromdir, todir, namelist, is_gt, is_train in dir_map:
        for imgname in namelist:
            from_name = os.path.join(fromdir, imgname)

            to_fname = imgname.replace('_gt', '')
            to_fname = os.path.splitext(to_fname)[0]
            to_ext = '.png' if is_gt else '.jpg'

            img = cv2.imread(from_name)[:, :, 1:2]
            img = np.repeat(img, 3, axis=-1)

            if is_train:
                if target_sz is None:
                    div_imgs = [img]
                else:
                    h, w = img.shape[:2]
                    num_p_h = int(math.ceil(float(h) / target_sz[0]))
                    num_p_w = int(math.ceil(float(w) / target_sz[1]))
                    div_imgs, h_stride, w_stride \
                        = getPatchedList(img, target_sz[0], target_sz[1], num_p_h, num_p_w)

                for indx, div_img in enumerate(div_imgs):
                    to_name_indx = '' if (indx == 0) else ('_%d' % indx)
                    to_name = os.path.join(todir, to_fname + to_name_indx + to_ext)
                    cv2.imwrite(to_name, div_img)
            else:
                to_name = os.path.join(todir, to_fname + to_ext)
                cv2.imwrite(to_name, img)

            # shutil.copy(from_name, to_name)
