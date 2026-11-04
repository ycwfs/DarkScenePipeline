"""CPU-only metric calculation shared by training and reporting."""
import numpy as np
from common import LABELS


def metrics(rows, labels):
    cm=np.zeros((10,10),np.int64)
    for r,p in zip(rows,labels):cm[LABELS.index(r['label']),int(p)]+=1
    support=cm.sum(1);f1=2*np.diag(cm)/np.maximum(support+cm.sum(0),1)
    recall=np.diag(cm)/np.maximum(support,1);positive=support[:9]>0
    return dict(n=len(rows),accuracy=float(np.trace(cm)/max(len(rows),1)),
        macro_f1=float(f1[support>0].mean()) if np.any(support) else 0.,
        other_n=int(support[9]),other_fp=int(support[9]-cm[9,9]),
        other_fpr=float((support[9]-cm[9,9])/max(support[9],1)),
        positive_n=int(sum(support[:9])),positive_tp=int(sum(np.diag(cm)[:9])),
        positive_macro_recall=float(recall[:9][positive].mean()) if positive.any() else 0.,
        classes={c:dict(n=int(support[j]),tp=int(cm[j,j]),recall=float(recall[j]),f1=float(f1[j]))for j,c in enumerate(LABELS)},
        confusion=cm.tolist())
