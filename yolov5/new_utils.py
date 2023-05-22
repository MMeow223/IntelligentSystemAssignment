import math

def TruthValues(metric):
    confusion_metric = metric

    tp = confusion_metric.diagonal()
    fp = confusion_metric.sum(axis=1) - tp
    fn = confusion_metric.sum(axis=0) - tp
    tn = confusion_metric.sum() - (tp + fp + fn)
        
    return [tp, tn, fp, fn]
    
    
def Accuracy(truth):
    tp = truth[0]
    tn = truth[1]
    fp = truth[2]
    fn = truth[3]
    
    accuracy = []
    for i in range(len(tp)):
        accuracy.append((tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i]))
    
    return accuracy
    
def Percision(truth):
    tp = truth[0]
    tn = truth[1]
    fp = truth[2]
    fn = truth[3]
    
    precision = []
    for i in range(len(tp)):
        precision.append(tp[i] / (tp[i] + fp[i]))
    
    return precision

def Recall(truth):
    tp = truth[0]
    tn = truth[1]
    fp = truth[2]
    fn = truth[3]
    
    recall = []
    for i in range(len(tp)):
        recall.append(tp[i] / (tp[i] + fn[i]))
    
    return recall

def F1Score(truth):
    tp = truth[0]
    tn = truth[1]
    fp = truth[2]
    fn = truth[3]
    
    f1 = []
    for i in range(len(tp)):
        f1.append(2 * tp[i] / (2 * tp[i] + fp[i] + fn[i]))
    
    return f1

def MatthewsCorrelationCoefficient(truth):
    tp = truth[0]
    tn = truth[1]
    fp = truth[2]
    fn = truth[3]
    
    mc = []
    # (TP x TN - FP x FN) / sqrt((TP+FP) x (TP+FN) x (TN+FP) x (TN+FN)).
    
    for i in range(len(tp)):
        mc.append((tp[i] * tn[i] - fp[i] * fn[i]) / math.sqrt((tp[i] + fp[i]) * (tp[i] + fn[i]) * (tn[i] + fp[i]) * (tn[i] + fn[i])))
    
    return mc
    
def CohenKappa(truth):
    tp = truth[0]
    tn = truth[1]
    fp = truth[2]
    fn = truth[3]
    
    ck = []
    
    # alternative formula (also return same result)
    # p0 = (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i])
    # pyes = ((tp[i] + fp[i]) / (tp[i] + tn[i] + fp[i] + fn[i])) * ((tp[i] + fn[i]) / (tp[i] + tn[i] + fp[i] + fn[i]))
    # pno = ((fn[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i])) * ((fp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i]))
    # pe = pyes + pno

    for i in range(len(tp)):
        ck.append((2 * (tp[i] * tn[i] - fn[i] * fp[i])) / ((tp[i] + fp[i]) * (fp[i] + tn[i]) + (tp[i] + fn[i]) * (fn[i] + tn[i])))

    return ck        
   
    
