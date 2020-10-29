import os
import cv2

import numpy as np


def mkdir(dname):
    try:
        os.mkdir(dname)
    except:
        pass


def populate(from_dname, to_dname, proc_fn, prefix='', suffix='',
             extension='.png'):
    for f_fname in os.listdir(from_dname):
        f_extname = os.path.splitext(f_fname)[1]
        if f_extname.lower() not in ['.png', '.jpg', '.jpeg', '.seg']:
            continue

        abs_from_fname = os.path.join(from_dname, f_fname)
        basename = os.path.splitext(f_fname)[0]
        to_fname = os.path.join(
            to_dname, prefix + basename + suffix + extension)

        img = proc_fn(abs_from_fname)
        cv2.imwrite(to_fname, img)


def procCFDGT(in_fname):
    w, h = None, None
    for row in open(in_fname, 'r'):
        if 'width' in row:
            w = int(row.replace('width', ''))
        if 'height' in row:
            h = int(row.replace('height', ''))
            gt_img = np.zeros((h, w, 1), np.uint8)

        try:
            anno, y, x_start, x_end = row.strip().split(' ')
            anno, y, x_start, x_end = int(anno), int(
                y), int(x_start), int(x_end)
        except:
            continue

        if anno == 0:
            continue

        elif anno == 1:
            gt_img[y, x_start:x_end, :] = 255
    return gt_img


def processAigle():
    from_dname = 'data/aigle_github'
    to_dname = 'data/aigle_detailed'
    mkdir(to_dname)

    invert_fn = lambda imgname: 255 - cv2.imread(imgname)

    t_img_dname = os.path.join(to_dname, 'original')
    mkdir(t_img_dname)
    f_img_dname = os.path.join(from_dname, 'original')
    populate(f_img_dname, t_img_dname, cv2.imread)

    t_img_dname = os.path.join(to_dname, 'gt')
    mkdir(t_img_dname)
    f_img_dname = os.path.join(from_dname, 'gt')
    populate(f_img_dname, t_img_dname, invert_fn, prefix='Im_', suffix='or_gt')


def processCFD():
    from_dname = 'data/cfd_github'
    to_dname = 'data/cfd_detailed'
    mkdir(to_dname)

    t_img_dname = os.path.join(to_dname, 'original')
    mkdir(t_img_dname)
    f_img_dname = os.path.join(from_dname, 'image')
    populate(f_img_dname, t_img_dname, cv2.imread, extension='.jpg')

    t_img_dname = os.path.join(to_dname, 'gt')
    mkdir(t_img_dname)
    f_img_dname = os.path.join(from_dname, 'seg')
    populate(f_img_dname, t_img_dname, procCFDGT, suffix='_gt')


def processDCD():
    from_dname = 'data/deepcrack_github/dataset'
    to_dname = 'data/deepcrack_detailed'
    mkdir(to_dname)

    for pre, dn in zip(['', '_'], ['test', 'train']):
        t_img_dname = os.path.join(to_dname, 'original')
        mkdir(t_img_dname)
        f_img_dname = os.path.join(from_dname, dn+'_img')
        populate(
            f_img_dname, t_img_dname, cv2.imread, prefix=pre, extension='.jpg')

        t_img_dname = os.path.join(to_dname, 'gt')
        mkdir(t_img_dname)
        f_img_dname = os.path.join(from_dname, dn+'_lab')
        populate(f_img_dname, t_img_dname, cv2.imread, prefix=pre, suffix='_gt')


def check():
    for dname in ['aigle', 'cfd', 'deepcrack']:
        gt_dirname = 'data/%s_detailed/gt' % dname
        ori_dirname = 'data/%s_detailed/original' % dname
        gts = os.listdir(gt_dirname)
        oris = os.listdir(ori_dirname)
        gt_comp = [os.path.splitext(gname.replace('_gt', ''))[0] for gname in gts]
        ori_comp = [os.path.splitext(oname.replace('_gt', ''))[0] for oname in oris]

        for gname, ggname in zip(gt_comp, gts):
            if gname not in ori_comp:
                os.remove(os.path.join(gt_dirname, ggname))
        for oname, ooname in zip(ori_comp, oris):
            if oname not in gt_comp:
                os.remove(os.path.join(ori_dirname, ooname))


if __name__ == "__main__":
    print('Aigle')
    processAigle()
    print('CFD')
    processCFD()
    print('DCD')
    processDCD()
    check()
