from common import *
from run_graph import *

graph = run({'use_dropout': True,
             'use_max_pool': True,
             'learning_rate': 0.05,
             'layers': [16, 32, 64]}, dataset_28)
