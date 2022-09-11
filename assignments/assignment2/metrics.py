def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    accuracy = 0
    pred_len = len(prediction)
    if (pred_len):
        accuracy = sum(prediction == ground_truth) / pred_len
    
    return accuracy
