import os
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import pytz
import pandas as pd
from IPython.display import display

GIT_COMMIT_HASH = os.popen('git rev-parse --short HEAD').read().replace('\n', '')
if not os.path.exists(GIT_COMMIT_HASH):
    os.system('mkdir {}'.format(GIT_COMMIT_HASH))

IPYNB_FILENAME = 'classification_slex_212.ipynb'
CONFIG_FILENAME = 'parameters.pkl'

PARAMS = {'PRIOR': ['kurtotic'],
          'USE_MASK': [False],
          'GAUSS_MASK_SIGMA': [1.0],
          'IMAGE_FILTER': [(-1,1)],
          'DOG_KSIZE': [(5,5)],
          'DOG_SIGMA1': [1.3],
          'DOG_SIGMA2': [2.6],
          'INPUT_SCALE': [1.0],
          'ITER_N': [30],
          'EPOCH_N': [1000],
          'K1': [0.005],
          'K2': [0.01],
          'SS0': [1.0],
          'SS1': [10.0],
          'SS2': [10.0],
          'SS3': [2.0],
          'ALPHA1': [1.0],
          'ALPHA2': [0.05],
          'ALPHA3': [0.05],
          'LAMBDA1': [0.02],
          'LAMBDA2': [0.00001],
          'LAMBDA3': [0.02],
          'CLEAR_SAVED_WEIGHTS': [False],
          'IN_DIR': ['slex_small'],
          'OUT_DIR': ['classification_slex_212_results'],
          'RF1_SIZE': [{'x': 9, 'y': 9}],
          'RF1_OFFSET': [{'x': 6, 'y': 6}],
          'RF1_LAYOUT': [{'x': 5, 'y': 3}]}

PARAMS_LIST = []

# # one grid for each k
# for k in [0.001]:
#     PARAMS_COPY = PARAMS.copy()
#     PARAMS_COPY['K1'] = [k]
#     PARAMS_COPY['K2'] = [k]
#     PARAMS_LIST.append(PARAMS_COPY)

PARAMS_COPY = PARAMS.copy()
PARAMS_LIST.append(PARAMS_COPY)

GRID = list(ParameterGrid(PARAMS_LIST))
GRID_df = pd.DataFrame(GRID)

# save parameter grid
timestamp = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d_%H-%M-%S-%f')
output_path = os.path.join(GIT_COMMIT_HASH, 'ParameterGrid_{}.pkl'.format(timestamp))
GRID_df.to_pickle(output_path)

# run simulation for each set of parameters
for i in GRID_df.index:
    param_df = GRID_df.iloc[i]
    param_df.to_pickle(CONFIG_FILENAME)

    display(param_df)
    
    timestamp = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d_%H-%M-%S-%f')
    output_path = os.path.join(GIT_COMMIT_HASH, '{}_{}.pdf'.format(os.path.splitext(IPYNB_FILENAME)[0], timestamp))
    
    os.system('jupyter nbconvert {} --execute --to pdf --output {}'.format(IPYNB_FILENAME, output_path))