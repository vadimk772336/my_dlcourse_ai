def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    n = len(prediction)
    tp,tn,fp,fn = 0,0,0,0
    
    for i in range(n):
        if prediction[i] and ground_truth[i]:
            tp += 1
        if prediction[i] == ground_truth[i] == False:
            tn += 1
        if prediction[i] and not ground_truth[i]:
            fp += 1
        if not prediction[i] and ground_truth[i]:
            fn +=1
            
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/n
    f1 = (2*precision*recall)/(precision+recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    accuracy = len(prediction[prediction == ground_truth])/len(prediction)
    return accuracy
