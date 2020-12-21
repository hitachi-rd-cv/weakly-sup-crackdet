
import numpy as np
import cv2
import os


def convertFnameDomains(from_domain, to_domain, fname):
    """
    Converts filenames in different domains
    (ex. ground truth imagename to original imagename)

    Available domain names:
    "original", "original2", "ground_truth", "annotated", "unknown", "output",
    "output_heatmap"

    Usage Example:
    convertFnameDomains(original, ground_truth, a.jpg)
    -> returns "a_gt.png"
    """
    domain_dict = {'jpg': '.jpg', 'jpeg': '.jpeg',
                   'ground_truth': '_gt.png', 'png': '.png',
                   'original': '.png', 'output': '_output.png',
                   'annotated': '_annotated.png'}

    from_keyword = (domain_dict[from_domain]
                    if from_domain in domain_dict
                    else '_%s.png' % from_domain)

    to_keyword = (domain_dict[to_domain]
                  if to_domain in domain_dict
                  else '_%s.png' % to_domain)

    if from_domain == 'original':
        fname = os.path.splitext(fname)[0] + '.png'
    fname = fname.replace(from_keyword, to_keyword)

    return fname


def rescaleTo255(img, threeD=False):
    scaled = np.clip(img.astype(float) * 255, 0, 255).astype(np.uint8)

    return (np.expand_dims(scaled, -1)
            if (threeD and len(scaled.shape) != 3) else scaled)


def rescaleTo1(img, threeD=False):
    scaled = np.clip(img.astype(float) / 255, 0, 1)

    return (np.expand_dims(scaled, -1)
            if (threeD and len(scaled.shape) != 3) else scaled)


def min0max1scale(img, ref_img=None):
    if (ref_img is None) or (ref_img.size == 0):
        ref_img = img

    img_fl = img.astype(float)
    mi, ma = np.min(ref_img), np.max(ref_img)
    return np.clip((img_fl - mi) / (ma - mi), 0, 1)


def convert2extension(in_dir, out_dir, to_domain, from_domain='unknown'):
    makeDir(out_dir)
    for fname in getAbsPaths(in_dir):
        img = cv2.imread(fname)
        suffix = convertFnameDomains(from_domain, to_domain,
                                     os.path.basename(fname))
        cv2.imwrite(os.path.join(out_dir, suffix), img)


def getAbsPaths(path_name):
    return [os.path.join(path_name, fn) for fn in os.listdir(path_name)]


def makeDir(dirname):
    print(dirname)
    # create a folder
    if type(dirname) == str:
        dirname = [dirname]

    for d in dirname:
        try:
            os.makedirs(d)
        except:
            pass


def countImgCrackPixels(fname, ano_color, verbose=True):
    img = cv2.imread(fname)

    h, w = img.shape[:2]
    num_pixels = h * w
    crack_map = np.all(img == ano_color, axis=-1)
    num_crack_pixels = np.count_nonzero(crack_map)

    density = float(num_crack_pixels) / num_pixels

    if verbose:
        print('%s: %d' % (fname, num_crack_pixels))

    return num_crack_pixels


def calcDensityStats(density_list):
    eps = 1e-5
    density_arr = np.stack(density_list)
    dens_sorted = -np.sort(-density_arr)
    dens_normalized = dens_sorted / float(np.sum(dens_sorted) + eps)
    cum_dens = np.cumsum(dens_normalized)

    cum_list = ['%.3f' % c for c in list(cum_dens)]
    print('\n'.join(cum_list))


def getCorrectImgExt(png_fname):
    if not os.path.isfile(png_fname):
        return png_fname.replace('png', 'jpg')
    return png_fname


if __name__ == "__main__":
    # # convert extension in a directory
    # # annotated images should end with "_annotated.png"
    # convert2extension('deepcrack/tmp',
    #                   'deepcrack/paper_deepcrack_gf',
    #                   'output', from_domain='gf')

    # calculate crack density
    dataset_name = 'deepcrack'
    anno_dir = os.path.join(dataset_name, 'full_annotated_gt')
    fnames = [os.path.join(anno_dir, fname) for fname in os.listdir(anno_dir)]
    ds = list()
    for fname in fnames:
        density = countImgCrackPixels(fname, ano_color=(0, 255, 0),
                                      verbose=False)
        ds.append(density)
    calcDensityStats(ds)
