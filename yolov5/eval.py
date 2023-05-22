from ultralytics import YOLO
from ultralytics.yolo.utils.metrics import Metric, ConfusionMatrix
import math
import pandas as pd


# [average f1 score, [f1 score for each class]]
# def f1_score(metric):
#     score = metric.box.f1
#     average_score =  sum(score) / len(score)
#     return [average_score, score]
    
    
def TruthValues(metric):
    confusion_metric = metric.confusion_matrix.matrix

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
   
    
if __name__ == "__main__":
    file = '11'
    model = YOLO(f'C:/Users/Cleme/yolov3_custom/yolov5/runs/train/exp{file}/weights/best.pt')  # load a custom model
    metric = model.val(data="C:/Users/Cleme/yolov3_custom/yolov5/data/coco128.yaml", save=True)  # no arguments needed, dataset and settings remembered
    
    truth = TruthValues(metric)
    a = Accuracy(truth)
    p = Percision(truth)
    r = Recall(truth)
    f1 = F1Score(truth)
    mc = MatthewsCorrelationCoefficient(truth)
    ck = CohenKappa(truth)

    # a.pop()
    # p.pop()
    # r.pop()
    # f1.pop()
    # mc.pop()
    # ck.pop()


    a.append((sum(a)/len(a)))
    p.append((sum(p)/len(p)))
    r.append((sum(r)/len(r)))
    f1.append((sum(f1)/len(f1)))
    mc.append((sum(mc)/len(mc)))
    ck.append((sum(ck)/len(ck)))

    table = [a, p, r, f1, mc, ck]

    df = pd.DataFrame(table, columns = ['fall','sit','walk','background','average'], index = ['Accuracy', 'Precision', 'Recall', 'F1', 'Matthews Correlation Coefficient', 'Cohen Kappa'])
    # df = pd.DataFrame(table, columns = ['fall','sit','walk','average'], index = ['Accuracy', 'Precision', 'Recall', 'F1', 'Matthews Correlation Coefficient', 'Cohen Kappa'])

    print(pd.DataFrame.transpose(df))
