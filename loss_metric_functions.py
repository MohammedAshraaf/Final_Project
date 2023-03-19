import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import sklearn
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import interp
EPS = 10e-8
MISSING_VALUE = -1


def customized_crossentropy(y_true, y_pred):
    """
    Applies cross entropy loss function while considering the missing values
    :param y_true: array of the ground truth
    :param y_pred: array of the prediction
    :return: cross entropy loss ignoring the missing labels
    """
    loss = K.binary_crossentropy(y_true, y_pred)

    # Find the indices of missing labels.
    missing_val = tf.constant(MISSING_VALUE, dtype=tf.float32)

    missing_value_condition = tf.not_equal(y_true, missing_val)

    zeros = tf.zeros_like(loss, dtype=loss.dtype)

    # get zeros for the missing values, otherwise, get the correct loss
    updated_loss = tf.where(missing_value_condition, loss, zeros)

    # calculate the mean loss for each class without considering missing labels
    labeled_sum = K.sum(K.cast(missing_value_condition, dtype='float32'), axis=0)
    loss_mean_cls = K.sum(updated_loss, axis=0) / (labeled_sum + EPS)

    zeros_ = tf.zeros_like(labeled_sum, dtype=labeled_sum.dtype)
    ones_ = tf.ones_like(labeled_sum, dtype=labeled_sum.dtype)

    # Calculate mean over all class losses
    labeled_cls = tf.where(tf.equal(labeled_sum, 0), zeros_, ones_)
    total_mean_loss = K.sum(loss_mean_cls) / (K.sum(labeled_cls) + EPS)

    return total_mean_loss


def precision_per_class_numpy(y_true, y_pred, cls):
    """
    Calculates precision for a given class
    :param y_true: ground truth array
    :param y_pred: prediction array
    :param cls: the desired class
    :return: precision
    """
    tp, fp, fn, tn = tp_fp_fn_tn_per_class_numpy(y_true, y_pred, cls)
    return tp / (tp + fp + EPS)


def recall_per_class_numpy(y_true, y_pred, cls):
    """
    Calculates recall for a given class
    :param y_true: ground truth array
    :param y_pred: prediction array
    :param cls: the desired class
    :return: recall
    """
    tp, _, fn, _ = tp_fp_fn_tn_per_class_numpy(y_true, y_pred, cls)
    return tp / (tp + fn + EPS)


def f1_score(y_true, y_pred, cls):
    """
    Calculates F1 score for a given class
    :param y_true: ground truth array
    :param y_pred: prediction array
    :param cls: the desired class
    :return: F1
    """
    precision = precision_per_class_numpy(y_true, y_pred, cls)
    recall = recall_per_class_numpy(y_true, y_pred, cls)
    return 2 * ((precision * recall) / (precision + recall + EPS))


def tp_fp_fn_tn_per_class_numpy(y_true, y_pred, cls):
    """
    Calculates tp, fp, fn, tn for a given class.
    :param y_true: ground truth array
    :param y_pred: prediction array
    :param cls: the desired class
    :return: recall
    """
    # Pick selected class outputs only
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    #calculate overall
    if cls == -1:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        non_missing_idx = y_true != MISSING_VALUE
        y_true = y_true[non_missing_idx]
        y_pred = y_pred[non_missing_idx]
    else:
        non_missing_idx = y_true[:, cls] != MISSING_VALUE
        y_true = y_true[non_missing_idx, cls]
        y_pred = y_pred[non_missing_idx, cls]

    # Threshold prediction probability
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.int32)
    y_true = y_true.astype(np.int32)

    tp = np.sum(y_pred * y_true).astype(np.float32)
    fp = np.sum(y_pred * (1 - y_true)).astype(np.float32)
    fn = np.sum((1 - y_pred) * y_true).astype(np.float32)
    tn = np.sum((1 - y_pred) * (1 - y_true)).astype(np.float32)

    return tp, fp, fn, tn