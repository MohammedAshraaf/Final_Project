import os
import cv2
import pandas as pd


class Datasource:
    def __init__(
            self,
            path_to_data_folder: str,
            labels: list,
            filters: list,
            dataset_split: str,
            datasource_name: str,
            lazy_load: bool = False,
    ):
        """
        :param path_to_data_folder: path to the main dataset folder
        :param labels list labels to be used for training
        :param filters: list of filters
        :param dataset_split: training, validation, or testing
        :param datasource_name: name to be used for the datasource
        :param lazy_load: whether to only load the images as needed or load everything in memory
        """
        self.dataset = {}
        self.df = None
        self.path_to_data_folder = path_to_data_folder
        self.filters = filters
        self.labels = labels
        self.datasource_name = datasource_name
        self.dataset_split = dataset_split
        self.lazy_load = lazy_load
        self.__load_labels_df()
        self.__filter_dataset()
        if not self.lazy_load:
            self.__load_dataset()

    def fetch_data(self, infinite=False):
        """
        creates a generator of the data in this datasource
        :param infinite: whether to have infinite generator or stop once all data is consumed
        :return: it yields image, label, and the label index of the current label associated with this dataset
        """
        if infinite:
            while True:
                if 'training' in self.datasource_name:
                    df = self.df.sample(frac=1)
                else:
                    df = self.df
                generator = self.__data_generator(df)
                for image, labels, label_idx in generator:
                    yield image, labels, label_idx
        else:
            generator = self.__data_generator(self.df)
            for image, labels, label_idx in generator:
                yield image, labels, label_idx

    def __data_generator(self, df):
        for image_id, row in df.iterrows():
            if self.lazy_load:
                image = self.__get_image(image_id)
            else:
                image = self.dataset[image_id]
            label_idx = -1
            if 'training' in self.datasource_name:
                label_idx = int(self.datasource_name.split('_')[1])
            yield image, row[self.labels], label_idx

    def __load_dataset(self):
        if not os.path.exists(self.path_to_data_folder):
            raise "path: {} does not exist".format(self.path_to_data_folder)

        for image_id, row in self.df.iterrows():
            image = self.__get_image(image_id)
            self.dataset[image_id] = image

    def __filter_dataset(self):
        for f in self.filters:
            self.df = f(self.df)

    def __get_image(self, image_id):
        image_file_path = '{}/images/{}'.format(self.path_to_data_folder, image_id)
        image = cv2.imread(image_file_path)
        return image

    def __load_labels_df(self):
        data_csv_path = '{}/labels/{}_updated.csv'.format(self.path_to_data_folder, self.dataset_split)

        self.df = pd.read_csv(data_csv_path, index_col='image_name')

