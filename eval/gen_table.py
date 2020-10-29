import os
from datetime import datetime

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np

from collections import defaultdict, OrderedDict


def parseResultsTxt(results_fname, prefix):
    with open(results_fname, 'r') as result_f:
        file_list = result_f.read().strip().split('\n')
        summary_indx = [i for i, s in enumerate(file_list) if ('best:' in s)][-1] + 1
        thresh, f1, rec, prec = map(float, file_list[summary_indx].split(','))

    info = {'thresh': thresh, 'f1': f1, 'rec': rec, 'prec': prec}
    return {'%s_%s' % (k, prefix): v for k, v in info.items()}


def parseLogDir(datadir, result_fnames,
                parse_func=parseResultsTxt, threshtime=None,
                alphabet_sort=True, header_map=None):
    log_dirnames = os.listdir(datadir)

    mod_times = [(logname, os.path.getmtime(os.path.join(datadir, logname)))
                 for logname in log_dirnames]
    if threshtime is not None:
        mod_times = [mt for mt in mod_times
                     if datetime.fromtimestamp(mt[1]) > datetime.strptime(threshtime, '%Y-%m-%d')]

    if alphabet_sort:
        log_dirnames.sort()
    else:
        log_dirnames = [s[0] for s in sorted(mod_times, key=lambda ss: ss[1],
                                             reverse=True)]

    # info_dict = OrderedDefaultDict(lambda: defaultdict(str))
    info_dict = OrderedDict()
    for log_dirname in log_dirnames:
        log_absdirname = os.path.join(datadir, log_dirname)

        for result_fname in result_fnames:
            rslt_absname = os.path.join(log_absdirname, result_fname)
            suffix = result_fname if (header_map is None) else header_map[result_fname]
            if os.path.isfile(rslt_absname):
                if log_dirname not in info_dict:
                    info_dict[log_dirname] = defaultdict(str)
                info_dict[log_dirname].update(parse_func(rslt_absname, suffix))

    return info_dict


def genTable(datadir, result_fnames, header_map):
    info_dict = parseLogDir(
        datadir, result_fnames,
        threshtime='2019-06-17', alphabet_sort=True, header_map=header_map)

    vals = list()
    headers = set()
    for k, v in info_dict.items():
        name_sep = k.split('_')
        row_vals = [name_sep[0], name_sep[1], '_'.join(name_sep[2:])]

        for kk in v.keys():
            headers.add(kk)

        v['1_dataset'] = name_sep[0]
        v['2_model'] = name_sep[1]
        v['3_dataset_modifier'] = '_'.join(name_sep[2:])
        vals.append(v)

    df = pd.DataFrame(vals, index=info_dict.keys())
    df.to_csv('results.csv')


if __name__ == "__main__":
    header_map = {
        'eval_test_output_dil0.txt': '0_test',
        'eval_cv_output_none_dil0.txt': '0_cv_none'}
    result_fnames = list(header_map)

    genTable('eval/results', result_fnames, header_map)
