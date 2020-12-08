import sys
import os
import pandas as pd
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

CWD = os.path.abspath('.')

IPYNB_FILENAME = sys.argv[1]
PKL_PATH = sys.argv[2]

param = pd.read_pickle(PKL_PATH)
print(param)

OUT_PATH = '{}.ipynb'.format(os.path.splitext(PKL_PATH)[0])

nb = nbformat.read(IPYNB_FILENAME, as_version=4)
nb['cells'].insert(1, nbformat.v4.new_code_cell('param = {}'.format(str(param.to_dict()))))
ep = ExecutePreprocessor()
ep.preprocess(nb, {'metadata': {'path': CWD}})

nbformat.write(nb, OUT_PATH)
os.system('jupyter nbconvert {} --to pdf'.format(OUT_PATH))
os.system('rm {}'.format(OUT_PATH))
