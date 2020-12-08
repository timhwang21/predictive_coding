import pandas as pd
import glob
import os

CWD = os.path.abspath('.')

GIT_COMMIT_HASH = os.popen('git rev-parse --short HEAD').read().replace('\n', '')
RESULTS_DIR = os.path.join(CWD, 'results', GIT_COMMIT_HASH)
if not os.path.exists(RESULTS_DIR):
    os.system('mkdir -p {}'.format(RESULTS_DIR))

pkl_list = glob.glob(os.path.join(RESULTS_DIR, '*.pkl'))

pkl_all = []

for x in pkl_list:
    pkl_name = os.path.splitext(os.path.split(x)[1])[0]
    
    pkl = pd.read_pickle(x)
    
    pkl['Name'] = pkl_name
    
    if type(pkl) == pd.core.series.Series:
        pkl = pkl.to_frame().T
    pkl_all.append(pkl)
    
pkl_all_concat = pd.concat(pkl_all, ignore_index=True)

pkl_all_concat.to_csv(os.path.join(RESULTS_DIR, 'all_pkl.csv'), index=False)
