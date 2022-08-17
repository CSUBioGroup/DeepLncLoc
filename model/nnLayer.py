from torch import nn as nn
from torch.nn import functional as F
import torch,time,os
import numpy as np

class TextSPP(nn.Module):
    def __init__(self, size=128, name='textSpp'):
        super(TextSPP, self).__init__()
        self.name = name
        self.spp = nn.AdaptiveAvgPool1d(size)
    def forward(self, x):
        return self.spp(x.cpu()).to(x.device)

class TextSPP2(nn.Module):
    def __init__(self, size=128, name='textSpp2'):
        super(TextSPP2, self).__init__()
        self.name = name
        self.spp1 = nn.AdaptiveMaxPool1d(size)
        self.spp2 = nn.AdaptiveAvgPool1d(size)
    def forward(self, x):
        x1 = self.spp1(x).unsqueeze(dim=3) # => batchSize × feaSize × size × 1
        x2 = self.spp2(x).unsqueeze(dim=3) # => batchSize × feaSize × size × 1
        x3 = -self.spp1(-x).unsqueeze(dim=3) # => batchSize × feaSize × size × 1
        return torch.cat([x1,x2,x3], dim=3) # => batchSize × feaSize × size × 3

class TextEmbedding(nn.Module):
    def __init__(self, embedding, dropout=0.3, freeze=False, name='textEmbedding'):
        super(TextEmbedding, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=freeze)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        # x: batchSize × seqLen
        return self.dropout(self.embedding(x))

class TextCNN(nn.Module):
    def __init__(self, feaSize, contextSizeList, filterNum, name='textCNN'):
        super(TextCNN, self).__init__()
        self.name = name
        moduleList = []
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=feaSize, out_channels=filterNum, kernel_size=contextSizeList[i]),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1)
                    )
                )
        self.conv1dList = nn.ModuleList(moduleList)
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = [conv(x).squeeze(dim=2) for conv in self.conv1dList] # => scaleNum * (batchSize × filterNum)
        return torch.cat(x, dim=1) # => batchSize × scaleNum*filterNum

class TextAYNICNN(nn.Module):
    def __init__(self, featureSize, filterSize, contextSizeList=[1,3,5], name='textAYNICNN'):
        super(TextAYNICNN, self).__init__()
        moduleList = []
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=featureSize, out_channels=filterSize, kernel_size=contextSizeList[i], padding=contextSizeList[i]//2),
                    nn.ReLU(),
                    )
                )
        self.feaConv1dList = nn.ModuleList(moduleList)
        self.attnConv1d = nn.Sequential(
                            nn.Conv1d(in_channels=featureSize+filterSize*len(contextSizeList), out_channels=1, kernel_size=5, padding=2),
                            nn.Softmax(dim=2)
                          )
        self.name = name
    def forward(self, x):
        # x: batchSize × feaSize × seqLen
        fea = torch.cat([conv(x) for conv in self.feaConv1dList],dim=1) # => batchSize × filterSize*contextNum × seqLen
        xfea = torch.cat([x,fea], dim=1) # => batchSize × (filterSize*contextNum+filterSize) × seqLen
        alpha = self.attnConv1d(xfea).transpose(1,2) # => batchSize × seqLen × 1
        return torch.matmul(xfea, alpha).squeeze(dim=2) # => batchSize × (feaSize+filterSize*contextNum)

class TextAttnCNN(nn.Module):
    def __init__(self, feaSize, contextSizeList, filterNum, seqMaxLen, name='textAttnCNN'):
        super(TextAttnCNN, self).__init__()
        self.name = name
        self.seqMaxLen = seqMaxLen
        moduleList = []
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=feaSize, out_channels=filterNum, kernel_size=contextSizeList[i]),
                    nn.ReLU(),
                    SimpleAttention(filterNum, filterNum//4, actFunc=nn.ReLU, name=f'SimpleAttention{i}', transpose=True)
                    )
                )
        self.attnConv1dList = nn.ModuleList(moduleList)
    def forward(self, x):
        x = [attnConv(x) for attnConv in self.attnConv1dList]
        return torch.cat(x, dim=1) # => batchSize × scaleNum*filterNum

class TextBiLSTM(nn.Module):
    def __init__(self, feaSize, hiddenSize, name='textBiLSTM'):
        super(TextBiLSTM, self).__init__()
        self.name = name
        self.biLSTM = nn.LSTM(feaSize, hiddenSize, bidirectional=True, batch_first=True)
    def forward(self, x, xlen=None):
        # x: batchSizeh × seqLen × feaSize
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)

            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)
        output, hn = self.biLSTM(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output[desortedIndices]
        return output # output: batchSize × seqLen × hiddenSize*2

class TextBiGRU(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, name='textBiGRU'):
        super(TextBiGRU, self).__init__()
        self.name = name
        self.biGRU = nn.GRU(feaSize, hiddenSize, bidirectional=True, batch_first=True, num_layers=num_layers, dropout=dropout)
    def forward(self, x, xlen=None):
        # x: batchSizeh × seqLen × feaSize
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)

            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)
        output, hn = self.biGRU(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output[desortedIndices]
        return output # output: batchSize × seqLen × hiddenSize*2

class FastText(nn.Module):
    def __init__(self, feaSize, name='fastText'):
        super(FastText, self).__init__()
        self.name = name
    def forward(self, x, xLen):
        # x: batchSize × seqLen × feaSize; xLen: batchSize
        x = torch.sum(x, dim=1) / xLen.float().view(-1,1)
        return x

class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.1, name='MLP', actFunc=nn.ReLU):
        super(MLP, self).__init__()
        self.name = name
        layers = nn.Sequential()
        for i,os in enumerate(hiddenList):
            layers.add_module(str(i*2), nn.Linear(inSize, os))
            layers.add_module(str(i*2+1), actFunc())
            inSize = os
        self.hiddenLayers = layers
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(inSize, outSize)
    def forward(self, x):
        x = self.hiddenLayers(x)
        return self.out(self.dropout(x))

class SimpleAttention(nn.Module):
    def __init__(self, inSize, outSize, actFunc=nn.Tanh, name='SimpleAttention', transpose=False):
        super(SimpleAttention, self).__init__()
        self.name = name
        self.W = nn.Parameter(torch.randn(size=(outSize,1), dtype=torch.float32))
        self.attnWeight = nn.Sequential(
            nn.Linear(inSize, outSize),
            actFunc()
            )
        self.transpose = transpose
    def forward(self, input):
        if self.transpose:
            input = input.transpose(1,2)
        # input: batchSize × seqLen × inSize
        H = self.attnWeight(input) # => batchSize × seqLen × outSize
        alpha = F.softmax(torch.matmul(H,self.W), dim=1) # => batchSize × seqLen × 1
        return torch.matmul(input.transpose(1,2), alpha).squeeze(2) # => batchSize × inSize

class ConvAttention(nn.Module):
    def __init__(self, feaSize, contextSize, transpose=True, name='convAttention'):
        super(ConvAttention, self).__init__()
        self.name = name
        self.attnConv = nn.Sequential(
            nn.Conv1d(in_channels=feaSize, out_channels=1, kernel_size=contextSize, padding=contextSize//2), 
            nn.Softmax(dim=2)
            )
        self.transpose = transpose
    def forward(self, x):
        if self.transpose:
            x = x.transpose(1,2)
        # x: batchSize × feaSize × seqLen
        alpha = self.attnConv(x) # => batchSize × 1 × seqLen 
        return alpha.transpose(1,2) # => batchSize × seqLen × 1

class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gama=2, weight=None, logit=True):
        super(FocalCrossEntropyLoss, self).__init__()
        self.weight = torch.tensor(weight, dtype=torch.float32) if weight is not None else weight
        self.gama = gama
        self.logit = logit
    def forward(self, Y_pre, Y):
        if self.logit:
            Y_pre = F.softmax(Y_pre, dim=1)
        P = Y_pre[list(range(len(Y))), Y]
        if self.weight is not None:
            w = self.weight[Y]
        else:
            w = torch.tensor([1.0 for i in range(len(Y))])
        w = (w/w.sum()).reshape(-1,1)
        return -((1-P)**self.gama * torch.log(P)).sum()

class HierarchicalSoftmax(nn.Module):
    def __init__(self, inSize, hierarchicalStructure, lab2id, hiddenList1=[], hiddenList2=[], dropout=0.1, name='HierarchicalSoftmax'):
        super(HierarchicalSoftmax, self).__init__()
        self.name = name
        self.dropout = nn.Dropout(p=dropout)
        layers = nn.Sequential()
        for i,os in enumerate(hiddenList1):
            layers.add_module(str(i*2), nn.Linear(inSize, os))
            layers.add_module(str(i*2+1), nn.ReLU())
            inSize = os
        self.hiddenLayers1 = layers
        moduleList = [nn.Linear(inSize, len(hierarchicalStructure))]

        layers = nn.Sequential()
        for i,os in enumerate(hiddenList2):
            layers.add_module(str(i*2), nn.Linear(inSize, os))
            layers.add_module(str(i*2+1), nn.ReLU())
            inSize = os
        self.hiddenLayers2 = layers

        for i in hierarchicalStructure:
            moduleList.append( nn.Linear(inSize, len(i)) )
            for j in range(len(i)):
                i[j] = lab2id[i[j]]
        self.hierarchicalNum = [len(i) for i in hierarchicalStructure]
        self.restoreIndex = np.argsort(sum(hierarchicalStructure,[]))
        self.linearList = nn.ModuleList(moduleList)
    def forward(self, x):
        # x: batchSize × feaSize
        x = self.hiddenLayers1(x)
        x = self.dropout(x)
        y = [F.softmax(linear(x), dim=1) for linear in self.linearList[:1]]
        x = self.hiddenLayers2(x)
        y += [F.softmax(linear(x), dim=1) for linear in self.linearList[1:]]
        y = torch.cat([y[0][:,i-1].unsqueeze(1)*y[i] for i in range(1,len(y))], dim=1) # => batchSize × classNum
        return y[:,self.restoreIndex]
