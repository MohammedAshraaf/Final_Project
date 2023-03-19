from datasource import Datasource
from network_x import NetworkX
from mobilenet_fullNetwork_network import MobilenetModel
from data_processor import DataProcessor
from filter_functions import filter_data_by_labels
from functools import partial
from collections import defaultdict
from config import main_configs

# load the datasources. Each datasource represents one combination of  splits, classes, ground truth
datasources = defaultdict(list)
# for split in ['training']:
#     for label_idx, label in enumerate(main_configs['labels']):
#         for truth_value in [1, 0]:
#             filters = [
#                 partial(filter_data_by_labels, filtering_label=label, ground_truth=truth_value)
#             ]
#             ds = Datasource(
#                 path_to_data_folder='../dataset/',
#                 labels=main_configs['labels'],
#                 filters=filters,
#                 dataset_split=split,
#                 datasource_name='{}_{}_{}'.format(split, label_idx, truth_value),
#                 lazy_load=True
#             )
#             print("Finished loading {} with {} images".format(ds.datasource_name, len(ds.df)))
#             datasources[split].append(ds)

for split in ['training', 'validation']:
        ds = Datasource(
            path_to_data_folder='../dataset/',
            labels=main_configs['labels'],
            filters=[],
            dataset_split=split,
            datasource_name='{}'.format(split),
            lazy_load=True
        )
        print("Finished loading {} with {} images".format(ds.datasource_name, len(ds.df)))
        datasources[split].append(ds)

# build data processor for training
training_data_processor = DataProcessor(
    datasources=datasources['training'],
    input_shape=main_configs['input_shape'],
    batch_size=main_configs['batch_size'],
    classes=main_configs['labels'],
    augmentation=True
)

# build data processor for validation
validation_data_processor = DataProcessor(
    datasources=datasources['validation'],
    input_shape=main_configs['input_shape'],
    batch_size=main_configs['batch_size'],
    classes=main_configs['labels'],
    augmentation=False
)

# create training and validation generators
training_data = training_data_processor.fetch_data_generator()
validation_data = validation_data_processor.fetch_all_data()

# create the model
# model = MobilenetModel(
#     classes=main_configs['labels'],
#     input_shape=main_configs['input_shape'],
#     model_saving_path=main_configs['experiment_saving_model'],
#     pretrained_weights='imagenet',
#     learning_rate=main_configs['learning_rate']
# )
#
model = NetworkX(
    classes=main_configs['labels'],
    model_saving_path=main_configs['experiment_saving_model'],
    input_shape=main_configs['input_shape'],
    first_filters=16,
    last_filters=256,
    alpha_contraction=1,
    learning_rate=main_configs['learning_rate'],
    feature_extractor=False,
    trainable=True

)


# train the model
model.train_network(training_data, validation_data, validation_batch_size=128)
