import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np


class Augmentation:
    @staticmethod
    def apply_scale_augmentation(image):
        return iaa.Affine(scale=(0.85, 1.15)).augment_image(image)

    @staticmethod
    def apply_rotate_augmentation(image):
        return iaa.Affine(rotate=(-15, 15)).augment_image(image)

    @staticmethod
    def apply_translate_augmentation(image):
        return iaa.Affine(translate_percent={"x": (-0.15, 0.15), "y": (-0.10, 0.10)}).augment_image(image)

    @staticmethod
    def apply_flipping_augmentation(image):
        return iaa.Fliplr(1).augment_image(image)

    @staticmethod
    def apply_contrast_augmentation(image):
        return iaa.LinearContrast((0.8, 1.2)).augment_image(image)

    @staticmethod
    def apply_gussianNoise_augmentation( image):
        return iaa.AdditiveGaussianNoise(scale=(0, 5)).augment_image(image)

    @staticmethod
    def apply_blurriness_augmentation(image):
        return iaa.GaussianBlur((0.5, 2)).augment_image(image)

    @staticmethod
    def apply_sharpness_augmentation(image):
        return iaa.Sharpen(alpha=(0.2, 0.6), lightness=1).augment_image(image)

    @staticmethod
    def apply_brightness_augmentation(image):
        return iaa.Add((-50, 50)).augment_image(image)

    @staticmethod
    def apply_augmentation(image, p=0.65):
        """
        applies series of augmentations on a given image starting with the giving probability
        :param image: the image to be augmented
        :param p: the probability of applying augmentation
        :return: the resulted image, whether the augmentation applied or not
        """
        augmentation_list = [
            Augmentation.apply_brightness_augmentation,
            Augmentation.apply_blurriness_augmentation,
            Augmentation.apply_flipping_augmentation,
            Augmentation.apply_contrast_augmentation,
            Augmentation.apply_gussianNoise_augmentation,
            Augmentation.apply_scale_augmentation,
            Augmentation.apply_rotate_augmentation,
            Augmentation.apply_translate_augmentation,
            Augmentation.apply_sharpness_augmentation
        ]

        np.random.shuffle(augmentation_list)
        x = np.random.random_sample()
        # don't apply augmentation
        if x > p:
            return image
        # apply augmentation
        for augmentation in augmentation_list:
            x = np.random.random_sample()
            # we can apply this augmentation
            if x < p:
                image = augmentation(image)
                # reduce the probability of applying new augmentation
                p -= 0.1

        return image
