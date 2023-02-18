# Definitions of all project constants
from os.path import join
from pathlib import Path

PROJECT_NAME = 'fea-net'
# DATA_PATH = join('.', 'data', "V2-FE_BULGETEST_SIMULATION_RESULTS.XLSX")
ROOT_PATH = Path.cwd()
DATA_PATH = ROOT_PATH / 'data' / 'new_data' / 'data_clean.csv'
TEST_SIZE = 0.15
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
HIDDEN_LSTM_DIM = 500
HIDDEN_FC_DIM = 64
