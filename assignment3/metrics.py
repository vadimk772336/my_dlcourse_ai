def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!

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
            
    if (tp+fp):
        precision = tp/(tp+fp)
    if (tp+fn):
        recall = tp/(tp+fn)
    if n:
        accuracy = (tp+tn)/n
    
    if (precision+recall):
        f1 = (2*precision*recall)/(precision+recall)
    
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    
    accuracy = 0
    pred_len = len(prediction)
    
    if (pred_len):
        accuracy = sum(prediction == ground_truth) / pred_len
    
    return accuracy
