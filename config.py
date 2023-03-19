main_configs = {
    'input_shape': [128, 128, 1],
    'batch_size': 12*2*5,
    'learning_rate': 0.0005,
    'labels': [
        'Anger', 'Disgust', 'Happiness', 'Sadness', 'Surprise',
        'AU01', 'AU02', 'AU04', 'AU09', 'AU15', 'AU25', 'Smile'
    ],
    'experiment_saving_model': '../trained_models/updated_labels/exp11_MNV2_imagenet_randomLabels_negative_allNegative'

}
