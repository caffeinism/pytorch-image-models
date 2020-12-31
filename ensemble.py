import numpy as np

def ensemble(pred_lists, targets=None, thresholds=None):    
    if thresholds is not None:
        print('use custom threshold')
        thresholds = thresholds
    elif targets is not None:
        print('do calibration...')
        thresholds = calibrate(pred_lists, targets)
        print('custom_thresholds: ', thresholds)
    else:
        print('use default threshold')
        thresholds = 0.5
        
    preds = sum(pred_lists) / len(pred_lists)
    prediction = preds > thresholds
    
    return preds, prediction

def calibrate(pred_lists, targets):
    preds = sum(pred_lists) / len(pred_lists)
    
    start, end, period = 5, 100, 5
    f1s = []
    for i in range(5, 100, 5):
        threshold = i / 100
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(targets, preds > threshold, average=None)
        f1s.append(f1)
#         print('threshold: {}\n  F1: {}'.format(threshold, f1))

    thresholds = (start + np.argmax(np.array(f1s), axis=0) * period) * 0.01

    return thresholds
