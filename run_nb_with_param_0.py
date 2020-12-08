import os
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import pytz
import pandas as pd
import hashlib

CWD = os.path.abspath('.')

GIT_COMMIT_HASH = os.popen('git rev-parse --short HEAD').read().replace('\n', '')
RESULTS_DIR = os.path.join(CWD, 'results', GIT_COMMIT_HASH)
if not os.path.exists(RESULTS_DIR):
    os.system('mkdir -p {}'.format(RESULTS_DIR))

PARAMS = {'USE_MASK': [False],
          'GAUSS_MASK_SIGMA': [1.0],
          'IMAGE_FILTER': [(-1,1)],
          'DOG_KSIZE': [(5,5)],
          'DOG_SIGMA1': [1.3],
          'DOG_SIGMA2': [2.6],
          'INPUT_SCALE': [1.0],
          'ITER_N': [5],
          'EPOCH_N': [10000],
          'CLEAR_SAVED_WEIGHTS': [False],
          'IN_DIR': [os.path.join(CWD, 'data', 'slex_len3_small')],
          'OUT_DIR': [os.path.join(RESULTS_DIR, 'slex_len3_small')],
          'RF1_SIZE': [{'x': 1, 'y': 3}],
          'RF1_OFFSET': [{'x': 1, 'y': 3}],
          'RF1_LAYOUT': [{'x': 1, 'y': 7}],
          'LEVEL1_MODULE_SIZE': [8],
          'LEVEL2_MODULE_SIZE': [32],
          'ALPHA_R': [0.05],
          'ALPHA_U': [0.0005],
          'ALPHA_V': [0.01],
          'ALPHA_DECAY': [1],
          'ALPHA_MIN': [0],
          'TEST_INTERVAL': [100]}

# save parameter grid
GRID = list(ParameterGrid(PARAMS))
GRID_DF = pd.DataFrame(GRID)

if PARAMS['ALPHA_V'] == [None]:
    GRID_DF.ALPHA_V = GRID_DF.ALPHA_U

TIMESTAMP = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d_%H-%M-%S')
GRID_DF['TIMESTAMP'] = TIMESTAMP

GRID_DF_KEEP = GRID_DF.drop(['EPOCH_N', 'OUT_DIR', 'CLEAR_SAVED_WEIGHTS', 'TIMESTAMP'], axis=1)

# save simulations with the same set of parameters to the same directory (with hashing)
GRID_DF['OUT_DIR'] = GRID_DF_KEEP.apply(lambda x: 
                                        os.path.join(RESULTS_DIR, hashlib.sha1(x.to_json().encode()).hexdigest()[:10]),
                                        axis=1)

for i in GRID_DF.index:
    OUT_DIR_SPLIT = os.path.split(GRID_DF.OUT_DIR[i])
    PKL_PATH = os.path.join(OUT_DIR_SPLIT[0], '{}_{:03d}_{}.pkl'.format(TIMESTAMP, i, OUT_DIR_SPLIT[1]))
    GRID_DF.loc[i].to_pickle(PKL_PATH)
    print(PKL_PATH)
