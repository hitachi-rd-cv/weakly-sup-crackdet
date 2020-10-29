import os
import shutil
import argparse

from utils import makeDir


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--deepcrack', action='store_true')
    parser.add_argument('--deeplab', action='store_true')

    return parser.parse_args()


def setupFiles(from_dir, to_dir):
    if os.path.isfile(from_dir):
        shutil.copyfile(from_dir, to_dir)
        return

    makeDir(to_dir)
    for dname in os.listdir(from_dir):
        setupFiles(os.path.join(from_dir, dname), os.path.join(to_dir, dname))


if __name__ == "__main__":
    # example usage
    # python tools/setup_models.py --deepcrack --deeplab
    args = get_args()

    if args.deepcrack:
        setupFiles('tools/model_supp/deepcrack', 'models/deepcrack')

    if args.deeplab:
        setupFiles('tools/model_supp/deeplab', 'models/deeplab/research/deeplab/')
