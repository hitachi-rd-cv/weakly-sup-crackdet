import os
import shutil
import argparse

from utils import makeDir

import gen_deeplab
import gen_deepcrack
import synthesize_ws_dataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--synth_dil_anno', action='store_true')
    parser.add_argument('--deepcrack', action='store_true')
    parser.add_argument('--deeplab', action='store_true')
    parser.add_argument('--fill', action='store_true')

    parser.add_argument('--dataset_name', type=str, nargs='*', default=list(),
                        help='name of the dataset')
    parser.add_argument('--anno_type', type=str, nargs='*', default=list(),
                        help='Dilation values. Larger value implies lower quality annotation.')

    return parser.parse_args()


def convDeepcrack(img_dir, dataset_name):
    target_sz = {'aigle': (462, 311), 'cfd': None, 'deepcrack': None}
    deepcrack_dir = os.path.join(img_dir, 'deepcrack')
    makeDir(deepcrack_dir)
    gen_deepcrack.main(img_dir, deepcrack_dir, target_sz[dataset_name])


def dispatchDeepcrack(dirname, out_dc_dirname):
    ori_dc_dirname = os.path.join(dirname, 'deepcrack')
    out_dirname = os.path.join(out_dc_dirname, 'datasets', os.path.basename(dirname))
    shutil.move(ori_dc_dirname, out_dirname)


def convDeeplab(img_dir, dataset_name):
    deeplab_dir = os.path.join(img_dir, 'deeplab')
    deeplab_tf_dir = os.path.join(img_dir, 'deeplab_tfrecords')
    makeDir([deeplab_dir, deeplab_tf_dir])
    gen_deeplab.main(img_dir, deeplab_dir, deeplab_tf_dir)


def dispatchDeeplab(dirname, out_dl_dirname):
    ori_dl_dirname = os.path.join(dirname, 'deeplab')
    out_dirname = os.path.join(
        out_dl_dirname, 'datasets/data', os.path.basename(dirname))
    shutil.move(ori_dl_dirname, out_dirname)

    ori_dl_dirname = os.path.join(dirname, 'deeplab_tfrecords')
    out_dirname = os.path.join(
        out_dl_dirname, 'datasets/data/tfrecords', os.path.basename(dirname))
    shutil.move(ori_dl_dirname, out_dirname)


def fillDataset(dirname):
    for dname in os.listdir(dirname):
        abs_dname = os.path.join(dirname, dname)

        # check if the directory needs to be filled
        # by checking if the directory contains the 'original' folder
        dset_name = dname.split('_')[0]
        if ('original' in os.listdir(abs_dname)) or (dset_name not in ['aigle', 'cfd', 'deepcrack']) or ('github' in abs_dname):
            continue

        from_dname = os.path.join(dirname, dset_name + '_detailed')
        for dname in ['original', 'data_split.json']:
            fromname = os.path.join(from_dname, dname)
            toname = os.path.join(abs_dname, dname)
            if os.path.isfile(fromname):
                shutil.copy(fromname, toname)
            else:
                shutil.copytree(fromname, toname)


def isInteger(n):
    try:
        int(n)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    args = get_args()
    dn = args.dataset_name
    dset_names = ['aigle', 'cfd', 'deepcrack'] if ('all' in dn) else dn

    # example usage
    # python data_gen.py --synth_dil_anno --anno_type 1 2 3 4 --dataset_name all
    if args.synth_dil_anno:
        anno_types = [int(val) for val in args.anno_type]
        for dset_name in dset_names:
            for atype in anno_types:
                target_dname = 'data/%s_%s' % (dset_name, atype)
                print('synthesizing ' + target_dname)
                synthesize_ws_dataset.main(dset_name, atype)

    anno_types = [('dil%s' % at if isInteger(at) else at) for at in args.anno_type]

    # example usage
    # python data_gen.py --deepcrack --anno_type 1 2 3 4 --dataset_name all
    if args.deepcrack:
        for dset_name in dset_names:
            for atype in anno_types:
                target_dname = 'data/%s_%s' % (dset_name, atype)
                print('deepcrack ' + target_dname)
                convDeepcrack(target_dname, dset_name)
                dispatchDeepcrack(target_dname, 'models/deepcrack')

    # example usage
    # python data_gen.py --deeplab --anno_type 1 2 3 4 --dataset_name all
    if args.deeplab:
        for dset_name in dset_names:
            for atype in anno_types:
                target_dname = 'data/%s_%s' % (dset_name, atype)
                print('deeplab ' + target_dname)
                convDeeplab(target_dname, dset_name)
                dispatchDeeplab(target_dname, 'models/deeplab/research/deeplab')

    # example usage
    # python data_gen.py --fill
    if args.fill:
        fillDataset('data')

    # os.system('chown -R 1001 .')
