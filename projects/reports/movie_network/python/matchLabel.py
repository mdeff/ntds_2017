import operator
from collections import Counter


def matchLabel(l,genres):
    pred_label = [x for x,y in sorted(Counter(l).items(), key=operator.itemgetter(1), reverse=True)]
    true_label = [x for x,y in sorted(Counter(genres).items(), key=operator.itemgetter(1), reverse=True)]
   
    pred = l.copy()
    for i in range(0,len(true_label)):
        pred[l==pred_label[i]] = true_label[i]
    return pred