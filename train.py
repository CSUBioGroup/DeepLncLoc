from model.utils import *
from model.DL_ClassifierModel import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--k', default='3')
parser.add_argument('--d', default='64')
parser.add_argument('--s', default='64')
parser.add_argument('--f', default='128')
parser.add_argument('--metrics', default='MaF')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--savePath', default='out/')
args = parser.parse_args()
if __name__ == '__main__':
    k,d,s,f = int(args.k),int(args.d),int(args.s),int(args.f)
    device,path = args.device,args.savePath
    metrics = args.metrics
    
    report = ["ACC", "MaF", "MiAUC", "MaAUC"]

    dataClass = DataClass('data.txt', 0.2, 0.0, kmers=k)
    dataClass.vectorize("char2vec", feaSize=d, loadCache=True)

    model = TextClassifier_SPPCNN( classNum=5, embedding=dataClass.vector['embedding'], SPPSize=s, feaSize=d, filterNum=f, 
                                   contextSizeList=[1,3,5], hiddenList=[], 
                                   embDropout=0.3, fcDropout=0.5, useFocalLoss=True, weight=None, 
                                   device=device)
    model.cv_train(dataClass, trainSize=1, batchSize=16, stopRounds=-1, earlyStop=10, 
                   epoch=100, lr=0.001, kFold=5, savePath=f'{path}CNN_s{s}_f{f}_k{k}_d{d}', report=report)