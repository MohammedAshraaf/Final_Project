def filter_data_by_labels(df, filtering_label, ground_truth):
    """
    filters the data based on the passed label
    :param df: dataframe for the dataset
    :param filtering_label: the label to be used for filtering
    :param ground_truth: the ground truth to filter the label based on
    :return: True if the filtering label = ground truth, False otherwise
    """
    return df[df[filtering_label] == ground_truth]
