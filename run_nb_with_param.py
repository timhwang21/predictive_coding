import os
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import pytz
import pandas as pd
from IPython.display import display

GIT_COMMIT_HASH = os.popen('git rev-parse --short HEAD').read().replace('\n', '')
if not os.path.exists(GIT_COMMIT_HASH):
    os.system('mkdir {}'.format(GIT_COMMIT_HASH))

IPYNB_FILENAME = 'classification_slex_slim.ipynb'
CONFIG_FILENAME = 'parameters.pkl'

PARAMS = {'USE_MASK': [False],
          'GAUSS_MASK_SIGMA': [1.0],
          'IMAGE_FILTER': [(-1,1)],
          'DOG_KSIZE': [(5,5)],
          'DOG_SIGMA1': [1.3],
          'DOG_SIGMA2': [2.6],
          'INPUT_SCALE': [1.0],
          'ITER_N': [1],
          'EPOCH_N': [1000],
          'CLEAR_SAVED_WEIGHTS': [True],
          'IN_DIR': ['slex_len3_small'],
          'OUT_DIR': ['slex_len3_small_results'],
          'RF1_SIZE': [{'x': 1, 'y': 3}],
          'RF1_OFFSET': [{'x': 1, 'y': 3}],
          'RF1_LAYOUT': [{'x': 1, 'y': 7}],
          'LEVEL1_MODULE_SIZE': [32],
          'LEVEL2_MODULE_SIZE': [128],
          'ALPHA_R': [0.1],
          'ALPHA_U': [0.1],
          'ALPHA_V': [0.1],
          'ALPHA_DECAY': [1],
          'ALPHA_MIN': [0],
          'TEST_INTERVAL': [100]}

GRID = list(ParameterGrid(PARAMS))
GRID_df = pd.DataFrame(GRID)

if PARAMS['ALPHA_V'] == [None]:
    GRID_df.ALPHA_V = GRID_df.ALPHA_U

# save parameter grid
timestamp = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d_%H-%M-%S')
output_path = os.path.join(GIT_COMMIT_HASH, 'ParameterGrid_{}.pkl'.format(timestamp))
GRID_df['timestamp'] = timestamp
GRID_df.to_pickle(output_path)

# run simulation for each set of parameters
for i in GRID_df.index:
    param_df = GRID_df.iloc[i]
    param_df.to_pickle(CONFIG_FILENAME)

    display(param_df)
    
    output_path = os.path.join(GIT_COMMIT_HASH, '{}_{}_{}.pdf'.format(os.path.splitext(IPYNB_FILENAME)[0], timestamp, i))
    
    os.system('jupyter nbconvert {} --execute --to pdf --output {}'.format(IPYNB_FILENAME, output_path))