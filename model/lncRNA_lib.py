from .nnLayer import *
from torch.nn import functional as F

class lncRNALocalizer:
    def __init__(self, weightPath, classNum=5, contextSizeList=[1,3,5], hiddenList=[], map_location=None, device=torch.device("cpu")):
        tmp = weightPath[:-4].split('_')
        params = {p[0]:int(p[1:]) for p in tmp if p[0] in ['s','f','k','d']}
        self.k = params['k']

        stateDict = torch.load(weightPath, map_location=map_location)
        self.lab2id,self.id2lab = stateDict['lab2id'],stateDict['id2lab']
        self.kmers2id,self.id2kmers = stateDict['kmers2id'],stateDict['id2kmers']
        self.textEmbedding = TextEmbedding( torch.zeros((len(self.id2kmers),params['d']), dtype=torch.float) ).to(device)
        self.textSPP = TextSPP(params['s']).to(device)
        self.textCNN = TextCNN(params['d'], contextSizeList, params['f']).to(device)
        self.fcLinear = MLP(len(contextSizeList)*params['f'], classNum, hiddenList).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding, self.textSPP, self.textCNN, self.fcLinear])
        for module in self.moduleList:
            module.load_state_dict(stateDict[module.name])
            module.eval()
        
        self.device = device

    def predict(self, x):
        # x: seqLen
        x = self.__transform__(x) # => 1 × seqLen
        x = self.textEmbedding(x).transpose(1,2) # => 1 × feaSize × seqLen
        x = self.textSPP(x) # => 1 × feaSize × sppSize
        x = self.textCNN(x) # => 1 × scaleNum*filterNum
        x = self.fcLinear(x)[0] # => classNum
        return  {k:v for k,v in zip(self.id2lab,F.softmax(x, dim=0).data.numpy())}
    def __transform__(self, RNA):
        RNA = ''.join([i if i in 'ATCG' else 'O' for i in RNA.replace('U', 'T')])
        kmers = [RNA[i:i+self.k] for i in range(len(RNA)-self.k+1)] + ['<EOS>']
        return torch.tensor( [self.kmers2id[i] for i in kmers if i in self.kmers2id],dtype=torch.long,device=self.device ).view(1,-1)

from collections import Counter 
def vote_predict(localizers, RNA):
    num = len(localizers)
    res = Counter({})
    for localizer in localizers:
        res += Counter(localizer.predict(RNA))
    return {k:res[k]/num for k in res.keys()}
