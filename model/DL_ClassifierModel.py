import numpy as np
import torch,time,os,pickle
from torch import nn as nn
from .nnLayer import *
from .metrics import *
from collections import Counter,Iterable
from sklearn.model_selection import StratifiedKFold

class BaseClassifier:
    def __init__(self):
        pass
    def calculate_y_logit(self, X, XLen):
        pass
    def cv_train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, 
                 optimType='Adam', lr=0.001, weightDecay=0, kFold=5, isHigherBetter=True, metrics="MaF", report=["ACC", "MaF"], 
                 savePath='model'):
        skf = StratifiedKFold(n_splits=kFold)
        validRes,testRes = [],[]
        tvIdList = dataClass.trainIdList+dataClass.validIdList
        for i,(trainIndices,validIndices) in enumerate(skf.split(tvIdList, dataClass.Lab[tvIdList])):
            print(f'CV_{i+1}:')
            self.reset_parameters()
            dataClass.trainIdList,dataClass.validIdList = [tvIdList[i] for i in trainIndices],[tvIdList[i] for i in validIndices]
            res = self.train(dataClass,trainSize,batchSize,epoch,stopRounds,earlyStop,saveRounds,optimType,lr,weightDecay,
                             isHigherBetter,metrics,report,f"{savePath}_cv{i+1}")
            validRes.append(res)
            if dataClass.testSampleNum>0:
                testRes.append(self.calculate_indicator_by_iterator(dataClass.one_epoch_batch_data_stream(type='test',batchSize=trainSize), dataClass.classNum, report))
        Metrictor.table_show(validRes, report)
        if dataClass.testSampleNum>0:
            Metrictor.table_show(testRes, report)
    def train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, 
              optimType='Adam', lr=0.001, weightDecay=0, isHigherBetter=True, metrics="MaF", report=["ACC", "MiF"], 
              savePath='model'):
        dataClass.describe()
        assert batchSize%trainSize==0
        metrictor = Metrictor(dataClass.classNum)
        self.stepCounter = 0
        self.stepUpdate = batchSize//trainSize
        optimizer = torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device)
        itersPerEpoch = (dataClass.trainSampleNum+trainSize-1)//trainSize
        mtc,bestMtc,stopSteps = 0.0,0.0,0
        if dataClass.validSampleNum>0: validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device)
        st = time.time()
        for e in range(epoch):
            for i in range(itersPerEpoch):
                self.to_train_mode()
                X,Y = next(trainStream)
                loss = self._train_step(X,Y, optimizer)
                if stopRounds>0 and (e*itersPerEpoch+i+1)%stopRounds==0:
                    self.to_eval_mode()
                    print(f"After iters {e*itersPerEpoch+i+1}: [train] loss= {loss:.3f};", end='')
                    if dataClass.validSampleNum>0:
                        X,Y = next(validStream)
                        loss = self.calculate_loss(X,Y)
                        print(f' [valid] loss= {loss:.3f};', end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*trainSize
                    speed = (e*itersPerEpoch+i+1)*trainSize/(time.time()-st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;"%(speed, restNum/speed))
            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
                self.to_eval_mode()
                print(f'========== Epoch:{e+1:5d} ==========')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
                metrictor.set_data(Y_pre, Y)
                print(f'[Total Train]',end='')
                metrictor(report)
                print(f'[Total Valid]',end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
                metrictor.set_data(Y_pre, Y)
                res = metrictor(report)
                mtc = res[metrics]
                print('=================================')
                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                    print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                    bestMtc = mtc
                    self.save("%s.pkl"%savePath, e+1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps>=earlyStop:
                        print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.')
                        break
        self.load("%s.pkl"%savePath)
        os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))
        print(f'============ Result ============')
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Train]',end='')
        metrictor(report)
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Valid]',end='')
        res = metrictor(report)
        metrictor.each_class_indictor_show(dataClass.id2lab)
        print(f'================================')
        return res
    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            stateDict['lab2id'],stateDict['id2lab'] = dataClass.lab2id,dataClass.id2lab
            stateDict['kmers2id'],stateDict['id2kmers'] = dataClass.kmers2id,dataClass.id2kmers
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            if "trainIdList" in parameters:
                dataClass.trainIdList = parameters['trainIdList']
            if "validIdList" in parameters:
                dataClass.validIdList = parameters['validIdList']
            if "testIdList" in parameters:
                dataClass.testIdList = parameters['testIdList']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=1)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=1)
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        return self.criterion(Y_logit, Y)
    def calculate_indicator_by_iterator(self, dataStream, classNum, report):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y)
        return metrictor(report)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.hstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=1), YArr
    def to_train_mode(self):
        for module in self.moduleList:
            module.train()
    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()
    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter<self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y)/self.stepUpdate
        loss.backward()
        if p:
            optimizer.step()
            optimizer.zero_grad()
        return loss*self.stepUpdate

class TextClassifier_SPPCNN(BaseClassifier):
    def __init__(self, classNum, embedding, SPPSize=128, feaSize=512, filterNum=448, contextSizeList=[1,3,5], hiddenList=[], 
                 embDropout=0.3, fcDropout=0.5, 
                 useFocalLoss=False, weight=None, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),dropout=embDropout ).to(device)
        self.textSPP = TextSPP(SPPSize).to(device)
        self.textCNN = TextCNN(feaSize, contextSizeList, filterNum).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding, self.textSPP, self.textCNN, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight)
    def calculate_y_logit(self, X):
        X,_ = X['seqArr'],X['seqLenArr']
        # X: batchSize × seqLen
        X = self.textEmbedding(X) # => batchSize × seqLen × feaSize
        X = X.transpose(1,2)
        X = self.textSPP(X) # => batchSize × feaSize × sppSize
        X = self.textCNN(X) # => batchSize × scaleNum*filterNum
        return self.fcLinear(X) # => batchSize × classNum
