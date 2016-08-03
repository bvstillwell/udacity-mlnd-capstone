from common import *
from run_graph import *

# Override some of the training settings
training_config = {
    'eval_step': 500,
    'valid_step': 500,
    'batch_size': 16,
    'mins': 5,
    'save_model': True
}

default_data_config['image_set'] = dataset_56

graph = run({'use_dropout': True,
             'use_max_pool': True,
             'learning_rate': 0.05,
             'layers': [16, 32, 64]},
            training_config)
