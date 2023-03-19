from datasource import Datasource
import numpy as np
import cv2
from augmentation import Augmentation
from tqdm import tqdm

class DataProcessor:
    def __init__(self,
                 datasources: list[Datasource],
                 input_shape: list,
                 batch_size: int,
                 classes: list,
                 augmentation: bool = False,
                 ):
        """
        :param datasources: list of datasources
        :param input_shape: input shape in format [h, w, c]
        :param batch_size: the batch size
        :param classes: list of the classes
        :param augmentation: whether to apply augmentation or not
        """
        self.datasources = datasources
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.classes = classes
        self.no_classes = len(classes)
        self.classes_idx_mapping = {label: i for i, label in enumerate(self.classes)}
        self.augmentation = augmentation

    def fetch_data_generator(self):
        """
        Fetches the data as a generator that works forever
        """
        # store the generators
        generators = {}
        for datasource in self.datasources:
            generators[datasource.datasource_name] = datasource.fetch_data(True)

        # prepare the batch's array
        x = np.empty((self.batch_size, *self.input_shape), dtype=np.float32)
        y = np.empty((self.batch_size, self.no_classes), dtype=np.float32)
        i = 0
        while True:
            for datasource_name in generators:
                data = next(generators[datasource_name], None)
                if data is None:
                    continue
                image, labels, label_idx = data

                x[i, ] = self.__image_preprocessing(np.copy(image))
                y[i, ] = np.copy(labels)
                if 'training' in datasource_name:
                    y[i, ] = np.ones_like(labels) * -1
                    y[i, label_idx] = labels[label_idx]

                i += 1
                if i >= self.batch_size:
                    yield x, y
                    i = 0

    def __image_preprocessing(self, image):
        if self.input_shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, self.input_shape[:2])
        image = np.expand_dims(image, -1)
        if self.augmentation:
            Augmentation.apply_augmentation(image, 0.6)

        return image.astype('float32') / 255.0

    def fetch_all_data(self):
        """
        fetches all the data once as numpy array
        :return: the images and their labels
        """
        x = []
        y = []
        with tqdm(total=56917) as tbar:
            for datasource in self.datasources:
                for image, label, image_id in datasource.fetch_data():
                    x.append(self.__image_preprocessing(np.copy(image)))
                    y.append(label)
                    tbar.update(1)
        return np.array(x), np.array(y, dtype=np.float32)

    def build_labels_connections(self, labels, label_idx):
        updated_labels = np.ones_like(labels) * -1
        updated_labels[label_idx] = labels[label_idx]
        if self.classes[label_idx] == 'Anger' and labels[label_idx] == 1:
            updated_labels[self.classes_idx_mapping['Happiness']] = 0
            updated_labels[self.classes_idx_mapping['Smile']] = 0

            updated_labels[self.classes_idx_mapping['AU04']] = 1
            updated_labels[self.classes_idx_mapping['AU07']] = 1

        if self.classes[label_idx] == 'Disgust' and labels[label_idx] == 1:
            updated_labels[self.classes_idx_mapping['Happiness']] = 0
            updated_labels[self.classes_idx_mapping['Smile']] = 0

            updated_labels[self.classes_idx_mapping['AU09']] = 1

        if self.classes[label_idx] == 'Sadness' and labels[label_idx] == 1:
            updated_labels[self.classes_idx_mapping['Happiness']] = 0
            updated_labels[self.classes_idx_mapping['Smile']] = 0

            updated_labels[self.classes_idx_mapping['AU15']] = 1
            updated_labels[self.classes_idx_mapping['AU01']] = 1

        if self.classes[label_idx] == 'Happiness' and labels[label_idx] == 1:
            updated_labels[self.classes_idx_mapping['Anger']] = 0
            updated_labels[self.classes_idx_mapping['AU04']] = 0
            updated_labels[self.classes_idx_mapping['Disgust']] = 0
            updated_labels[self.classes_idx_mapping['AU09']] = 0
            updated_labels[self.classes_idx_mapping['Sadness']] = 0
            updated_labels[self.classes_idx_mapping['AU15']] = 0

            updated_labels[self.classes_idx_mapping['Smile']] = 1
            updated_labels[self.classes_idx_mapping['AU06']] = 1

        if self.classes[label_idx] == 'Surprise' and labels[label_idx] == 1:
            updated_labels[self.classes_idx_mapping['AU02']] = 1
            updated_labels[self.classes_idx_mapping['AU05']] = 1
            # keep happiness as it's
            updated_labels[self.classes_idx_mapping['Happiness']] = labels[self.classes_idx_mapping['Happiness']]



        if self.classes[label_idx] == 'Anger' and labels[label_idx] == 0:
            updated_labels[self.classes_idx_mapping['AU04']] = 0
            updated_labels[self.classes_idx_mapping['AU07']] = 0

        if self.classes[label_idx] == 'Disgust' and labels[label_idx] == 0:
            updated_labels[self.classes_idx_mapping['AU09']] = 0

        if self.classes[label_idx] == 'Sadness' and labels[label_idx] == 0:
            updated_labels[self.classes_idx_mapping['AU15']] = 0
            updated_labels[self.classes_idx_mapping['AU01']] = 0

        if self.classes[label_idx] == 'Happiness' and labels[label_idx] == 0:
            updated_labels[self.classes_idx_mapping['Smile']] = 0
            updated_labels[self.classes_idx_mapping['AU06']] = 0

        if self.classes[label_idx] == 'Surprise' and labels[label_idx] == 0:
            updated_labels[self.classes_idx_mapping['AU02']] = 0
            updated_labels[self.classes_idx_mapping['AU05']] = 0

        return updated_labels



