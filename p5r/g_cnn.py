from common import *
from run_graph import *

redirect_print_to_file()


# Override some of the training settings
training_config = {
    'eval_step': 500,
    'valid_step': 500,
    'batch_size': 16,
    'mins': 2,
    'save_model': False,
}

default_data_config['image_set'] = dataset_28

graph = run({'use_dropout': True,
             'use_max_pool': True,
             'layers': [8]},
            training_config)

# print("Higher hidden layer")
# graph = run({'use_dropout': True,
#              'use_max_pool': True,
#              'learning_rate': 0.05,
#              'layers': [16, 32, 64],
#              'num_hidden': 128},
#             training_config)

# graph = run({'use_dropout': True,
#              'use_max_pool': True,
#              'learning_rate': 0.025,
#              'layers': [16, 32, 64]},
#             training_config)


# graph = run({'use_dropout': True,
#              'use_max_pool': True,
#              'learning_rate': 0.1,
#              'layers': [16, 32, 64]},
#             training_config)


# print("Common cnn sizes 16,16")
# graph = run({'use_dropout': True,
#              'use_max_pool': True,
#              'layers': [16, 16]},
#             training_config)

# print("Common cnn sizes 16, 16, 16")
# graph = run({'use_dropout': True,
#              'use_max_pool': True,
#              'layers': [16, 16, 16]},
#             training_config)

# print("Common cnn sizes 8,8,8")
# graph = run({'use_dropout': True,
#              'use_max_pool': True,
#              'layers': [8, 8, 8]},
#             training_config)

# print("Common cnn sizes 8,8,8")
# graph = run({'use_dropout': True,
#              'use_max_pool': True,
#              'layers': [8, 8]},
#             training_config)

