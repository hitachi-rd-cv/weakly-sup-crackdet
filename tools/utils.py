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
    domain_dict = {'original': '.jpg', 'original2': '.png', 'jpeg': '.jpeg',
                   'ground_truth': '_gt.png',
                   'annotated': '_annotated.png', 'output': '_output.png',
                   'output_heatmap': '_output_heatmap.png'}

    fname = fname[:-len(domain_dict[from_domain])]

    fname = '%s%s' % (fname, domain_dict[to_domain])

    return fname


def makeDir(dirname):
    # create a folder
    if type(dirname) == str:
        dirname = [dirname]

    for d in dirname:
        try:
            os.makedirs(d)
        except:
            pass
