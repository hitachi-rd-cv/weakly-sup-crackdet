
import os
import shutil
import argparse


def makeDir(dirname):
    # create a folder
    if type(dirname) == str:
        dirname = [dirname]

    for d in dirname:
        try:
            os.makedirs(d)
        except:
            pass


def createArgumentParser(flexible=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--log_dir', action='store', type=str,
                        default='train', required=True)

    flexible_keys = []
    if flexible:
        parsed, unknown = parser.parse_known_args()
        for arg in unknown:
            if arg.startswith(("-", "--")):
                flexible_keys.append(arg.replace('-', ''))
                parser.add_argument(arg, type=str, action='store')

    args = parser.parse_args()

    tmp_dict, flex_dict = vars(args), {}
    for k in flexible_keys:
        flex_dict[k] = tmp_dict[k]

    return args, flex_dict


if __name__ == '__main__':
    args, flex_dict = createArgumentParser(flexible=True)
    log_dir = args.log_dir

    out_dir = os.path.join(log_dir, 'sample_imgs/test_output')
    out_gf_dir = os.path.join(log_dir, 'sample_imgs/test_output_gf')
    deepcrack_dir = os.path.join(log_dir, 'test_latest/images')
    makeDir([os.path.join(log_dir, 'sample_imgs'), out_dir, out_gf_dir])

    for fname in os.listdir(deepcrack_dir):
        src_fname = os.path.join(deepcrack_dir, fname)

        if 'fused.png' in fname:
            fuse_fname = fname.replace('_fused.png', '_output.png')
            dst_fname = os.path.join(out_dir, fuse_fname)
            shutil.copy(src_fname, dst_fname)
        elif 'gf.png' in fname:
            fuse_fname = fname.replace('_gf.png', '_output.png')
            dst_fname = os.path.join(out_gf_dir, fuse_fname)
            shutil.copy(src_fname, dst_fname)

    # cleanup
    try:
        shutil.rmtree(os.path.join(log_dir, 'web/images'))
    except:
        pass
