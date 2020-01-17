import numpy as np
from sklearn import metrics as skmetrics
import warnings
warnings.filterwarnings("ignore")

def lgb_MaF(preds, dtrain):
    Y = np.array(dtrain.get_label(), dtype=np.int32)
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'macro_f1', float(MaF(preds.shape[1], Y_pre, Y)), True

def lgb_precision(preds, dtrain):
    Y = dtrain.get_label()
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'precision', float(Counter(Y==Y_pre)[True]/len(Y)), True

class Metrictor:
    def __init__(self, classNum):
        self.classNum = classNum
        self._reporter_ = {"MaF":self.MaF, "MiF":self.MiF, 
                           "ACC":self.ACC,
                           "MaAUC":self.MaAUC, "MiAUC":self.MiAUC, 
                           "MaMCC":self.MaMCC, "MiMCC":self.MiMCC}
    def __call__(self, report, end='\n'):
        res = {}
        for mtc in report:
            v = self._reporter_[mtc]()
            print(f" {mtc}={v:6.3f}", end=';')
            res[mtc] = v
        print(end=end)
        return res
    def set_data(self, Y_prob_pre, Y):
        self.Y_prob_pre,self.Y = Y_prob_pre,Y
        self.Y_pre = Y_prob_pre.argmax(axis=1)
        self.N = len(Y)
    @staticmethod
    def table_show(resList, report, rowName='CV'):
        lineLen = len(report)*8 + 6
        print("="*(lineLen//2-6) + "FINAL RESULT" + "="*(lineLen//2-6))
        print(f"{'-':^6}" + "".join([f"{i:>8}" for i in report]))
        for i,res in enumerate(resList):
            print(f"{rowName+'_'+str(i+1):^6}" + "".join([f"{res[j]:>8.3f}" for j in report]))
        print(f"{'MEAN':^6}" + "".join([f"{np.mean([res[i] for res in resList]):>8.3f}" for i in report]))
        print("======" + "========"*len(report))
    def each_class_indictor_show(self, id2lab):
        id2lab = np.array(id2lab)
        Yarr = np.zeros((self.N, self.classNum), dtype='int32')
        Yarr[list(range(self.N)),self.Y] = 1
        TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(self.classNum, self.Y_pre, self.Y)
        MCCi = fill_inf((TPi*TNi - FPi*FNi) / np.sqrt( (TPi+FPi)*(TPi+FNi)*(TNi+FPi)*(TNi+FNi) ), np.nan)
        Pi = fill_inf(TPi/(TPi+FPi))
        Ri = fill_inf(TPi/(TPi+FNi))
        Fi = fill_inf(2*Pi*Ri/(Pi+Ri))
        sortedIndex = np.argsort(id2lab)
        classRate = Yarr.sum(axis=0)[sortedIndex] / self.N
        id2lab,MCCi,Pi,Ri,Fi = id2lab[sortedIndex],MCCi[sortedIndex],Pi[sortedIndex],Ri[sortedIndex],Fi[sortedIndex]
        print("-"*28 + "MACRO INDICTOR" + "-"*28)
        print(f"{'':30}{'rate':<8}{'MCCi':<8}{'Pi':<8}{'Ri':<8}{'Fi':<8}")
        for i,c in enumerate(id2lab):
            print(f"{c:30}{classRate[i]:<8.2f}{MCCi[i]:<8.3f}{Pi[i]:<8.3f}{Ri[i]:<8.3f}{Fi[i]:<8.3f}")
        print("-"*70)
    def MaF(self):
        return F1(self.classNum,  self.Y_pre, self.Y, average='macro')
    def MiF(self, showInfo=False):
        return F1(self.classNum,  self.Y_pre, self.Y, average='micro')
    def ACC(self):
        return ACC(self.classNum, self.Y_pre, self.Y)
    def MaMCC(self):
        return MCC(self.classNum, self.Y_pre, self.Y, average='macro')
    def MiMCC(self):
        return MCC(self.classNum, self.Y_pre, self.Y, average='micro')
    def MaAUC(self):
        return AUC(self.classNum, self.Y_prob_pre, self.Y, average='macro')
    def MiAUC(self):
        return AUC(self.classNum, self.Y_prob_pre, self.Y, average='micro')

def _TPiFPiTNiFNi(classNum, Y_pre, Y):
    Yarr, Yarr_pre = np.zeros((len(Y), classNum), dtype='int32'), np.zeros((len(Y), classNum), dtype='int32')
    Yarr[list(range(len(Y))),Y] = 1
    Yarr_pre[list(range(len(Y))),Y_pre] = 1
    isValid = (Yarr.sum(axis=0) + Yarr_pre.sum(axis=0))>0
    Yarr,Yarr_pre = Yarr[:,isValid],Yarr_pre[:,isValid]
    TPi = np.array([Yarr_pre[:,i][Yarr[:,i]==1].sum() for i in range(Yarr.shape[1])], dtype='float32')
    FPi = Yarr_pre.sum(axis=0) - TPi
    TNi = (1^Yarr).sum(axis=0) - FPi
    FNi = Yarr.sum(axis=0) - TPi
    return TPi,FPi,TNi,FNi

def ACC(classNum, Y_pre, Y):
    TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    return TPi.sum() / len(Y)

def AUC(classNum, Y_prob_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    Yarr = np.zeros((len(Y), classNum), dtype='int32')
    Yarr[list(range(len(Y))),Y] = 1
    return skmetrics.roc_auc_score(Yarr, Y_prob_pre, average=average)

def MCC(classNum, Y_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    if average=='micro':
        TP,FP,TN,FN = TPi.sum(),FPi.sum(),TNi.sum(),FNi.sum()
        MiMCC = fill_inf((TP*TN - FP*FN) / np.sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) ), np.nan)
        return MiMCC
    else:
        MCCi = fill_inf( (TPi*TNi - FPi*FNi) / np.sqrt((TPi+FPi)*(TPi+FNi)*(TNi+FPi)*(TNi+FNi)), np.nan )
        return MCCi.mean()

def Precision(classNum, Y_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    if average=='micro':
        MiP = fill_inf(TPi.sum() / (TPi.sum() + FPi.sum()))
        return MiP
    else:
        Pi = fill_inf(TPi/(TPi+FPi))
        return Pi.mean()

def Recall(classNum, Y_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    if average=='micro':
        MiR = fill_inf(TPi.sum() / (TPi.sum() + FNi.sum()))
        return MiR
    else:
        Ri = fill_inf(TPi/(TPi + FNi))
        return Ri.mean()

def F1(classNum, Y_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    if average=='micro':
        MiP,MiR = Precision(classNum, Y_pre, Y, average='micro'),Recall(classNum, Y_pre, Y, average='micro')
        MiF = fill_inf(2*MiP*MiR/(MiP+MiR))
        return MiF
    else:
        TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
        Pi,Ri = TPi/(TPi + FPi),TPi/(TPi + FNi)
        Pi[Pi==np.inf],Ri[Ri==np.inf] = 0.0,0.0
        Fi = fill_inf(2*Pi*Ri/(Pi+Ri))
        return Fi.mean()

from collections import Iterable
def fill_inf(x, v=0.0):
    if isinstance(x, Iterable):
        x[x==np.inf] = v
        x[np.isnan(x)] = v
    else:
        x = v if x==np.inf else x
        x = v if np.isnan(x) else x
    return x
    