# Definitions of all project constants
from os.path import join
from pathlib import Path

PROJECT_NAME = 'fea-net'
# DATA_PATH = join('.', 'data', "V2-FE_BULGETEST_SIMULATION_RESULTS.XLSX")
ROOT_PATH = Path.cwd()
DATA_PATH = ROOT_PATH / 'data' / 'new_data' / 'data_clean.csv'
CONFIG_PATH = ROOT_PATH / 'src' / 'model_config.yaml'
MODEL_PATH = ROOT_PATH / 'model'