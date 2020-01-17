from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
import os,logging,pickle,random,torch
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class DataClass:
    def __init__(self, dataPath, validSize, testSize, kmers=3, check=True):
        validSize *= 1.0/(1.0-testSize)
        # Open files and load data
        print('Loading the raw data...')
        with open(dataPath,'r') as f:
            data = []
            for i in tqdm(f):
                data.append(i.strip().split('\t'))
        self.kmers = kmers
        # Get labels and splited sentences
        RNA,Lab = [[i[1][j:j+kmers] for j in range(len(i[1])-kmers+1)] for i in data],[i[2] for i in data]
        # Get the mapping variables for label and label_id
        print('Getting the mapping variables for label and label id......')
        self.lab2id,self.id2lab = {},[]
        cnt = 0
        for lab in tqdm(Lab):
            if lab not in self.lab2id:
                self.lab2id[lab] = cnt
                self.id2lab.append(lab)
                cnt += 1
        self.classNum = cnt
        # Get the mapping variables for kmers and kmers_id
        print('Getting the mapping variables for kmers and kmers id......')
        self.kmers2id,self.id2kmers = {"<EOS>":0},["<EOS>"]
        kmersCnt = 1
        for rna in tqdm(RNA):
            for kmers in rna:
                if kmers not in self.kmers2id:
                    self.kmers2id[kmers] = kmersCnt
                    self.id2kmers.append(kmers)
                    kmersCnt += 1
        self.kmersNum = kmersCnt
        # Tokenize the sentences and labels
        self.RNADoc = RNA
        self.Lab = np.array( [self.lab2id[i] for i in Lab],dtype='int32' )
        self.RNASeqLen = np.array([len(s)+1 for s in self.RNADoc],dtype='int32')
        self.tokenizedRNASent = np.array([[self.kmers2id[i] for i in s] for s in self.RNADoc])
        self.vector = {}
        print('Start train_valid_test split......')
        while True:
            restIdList,testIdList = train_test_split(range(len(data)), test_size=testSize) if testSize>0.0 else (list(range(len(data))),[])
            trainIdList,validIdList = train_test_split(restIdList, test_size=validSize) if validSize>0.0 else (restIdList,[])
            trainSampleNum,validSampleNum,testSampleNum = len(trainIdList),len(validIdList),len(testIdList)
            totalSampleNum = trainSampleNum + validSampleNum + testSampleNum
            if not check:
                break
            elif (trainSampleNum==0 or len(set(self.Lab[trainIdList]))==self.classNum) and \
                 (validSampleNum==0 or len(set(self.Lab[validIdList]))==self.classNum) and \
                 (testSampleNum==0 or len(set(self.Lab[testIdList]))==self.classNum):
                break
            else:
                continue
        self.trainIdList,self.validIdList,self.testIdList = trainIdList,validIdList,testIdList
        self.trainSampleNum,self.validSampleNum,self.testSampleNum = len(trainIdList),len(validIdList),len(testIdList)
        self.totalSampleNum = self.trainSampleNum+self.validSampleNum+self.testSampleNum
        print('train sample size:',len(self.trainIdList))
        print('valid sample size:',len(self.validIdList))
        print('test sample size:',len(self.testIdList))

    def describe(self):
        print(f'===========DataClass Describe===========')
        print(f'{"CLASS":<16}{"TRAIN":<8}{"VALID":<8}{"TEST":<8}')
        for i,c in enumerate(self.id2lab):
            trainIsC = sum(self.Lab[self.trainIdList]==i)/self.trainSampleNum if self.trainSampleNum>0 else -1.0
            validIsC = sum(self.Lab[self.validIdList]==i)/self.validSampleNum if self.validSampleNum>0 else -1.0
            testIsC  = sum(self.Lab[self.testIdList]==i) /self.testSampleNum  if self.testSampleNum>0  else -1.0
            print(f'{c:<16}{trainIsC:<8.3f}{validIsC:<8.3f}{testIsC:<8.3f}')
        print(f'========================================')
        self.Lab[self.trainIdList]

    def vectorize(self, method="char2vec", feaSize=512, window=5, sg=1, 
                        workers=8, loadCache=True):
        if os.path.exists(f'cache/{method}_k{self.kmers}_d{feaSize}.pkl') and loadCache:
            with open(f'cache/{method}_k{self.kmers}_d{feaSize}.pkl', 'rb') as f:
                if method=='kmers':
                    tmp = pickle.load(f)
                    self.vector['encoder'],self.kmersFea = tmp['encoder'],tmp['kmersFea']
                else:
                    self.vector['embedding'] = pickle.load(f)
            print(f'Loaded cache from cache/{method}_k{self.kmers}_d{feaSize}.pkl.')
            return
        if method == 'char2vec':
            doc = [i+['<EOS>'] for i in self.RNADoc]
            model = Word2Vec(doc, min_count=0, window=window, size=feaSize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.kmersNum, feaSize), dtype=np.float32)
            for i in range(self.kmersNum):
                char2vec[i] = model.wv[self.id2kmers[i]]
            self.vector['embedding'] = char2vec
        elif method == 'glovechar':
            from glove import Glove,Corpus
            doc = [i+['<EOS>'] for i in self.RNADoc]
            corpus = Corpus()
            corpus.fit(doc, window=window)
            glove = Glove(no_components=feaSize)
            glove.fit(corpus.matrix, epochs=10, no_threads=workers, verbose=True)
            glove.add_dictionary(corpus.dictionary)
            gloveVec = np.zeros((self.kmersNum, feaSize), dtype=np.float32)
            for i in range(self.kmersNum):
                gloveVec[i] = glove.word_vectors[glove.dictionary[self.id2kmers[i]]]
            self.vector['embedding'] = gloveVec
        elif method == 'kmers':
            enc = OneHotEncoder(categories='auto')
            enc.fit([[i] for i in self.kmers2id.values()])
            feaSize = len(self.kmers2id)
            kmers = np.zeros((self.totalSampleNum, feaSize))
            bs = 50000
            print('Getting the kmers vector......')
            for i,t in enumerate(tqdm(self.tokenizedRNASent)):
                for k in range((len(t)+bs-1)//bs):
                    kmers[i] += enc.transform(np.array(t[k*10000:(k+1)*10000]).reshape(-1,1)).toarray().sum(axis=0)
            kmers = kmers[:,1:]
            kmers = (kmers-kmers.mean(axis=0))/kmers.std(axis=0)
            self.vector['encoder'] = enc
            self.kmersFea = kmers
        with open(f'cache/{method}_k{self.kmers}_d{feaSize}.pkl', 'wb') as f:
            if method=='kmers':
                pickle.dump({'encoder':self.vector['encoder'], 'kmersFea':self.kmersFea}, f, protocol=4)
            else:
                pickle.dump(self.vector['embedding'], f, protocol=4)

    def vector_merge(self, vecList, mergeVecName='mergeVec'):
        self.vector[mergeVec] = np.hstack([self.vector[i] for i in vecList])
        print(f'Get a new vector "{mergeVec}" with shape {self.vector[mergeVec].shape}...')

    def random_batch_data_stream(self, batchSize=128, type='train', device=torch.device('cpu')):
        idList = self.trainIdList if type=='train' else self.validIdList
        X,XLen = self.tokenizedRNASent,self.RNASeqLen
        while True:
            random.shuffle(idList)
            for i in range((len(idList)+batchSize-1)//batchSize):
                samples = idList[i*batchSize:(i+1)*batchSize]
                RNASeqMaxLen = XLen[samples].max()
                yield {
                        "seqArr":torch.tensor([i+[0]*(RNASeqMaxLen-len(i)) for i in X[samples]], dtype=torch.long).to(device), \
                        "seqLenArr":torch.tensor(XLen[samples], dtype=torch.int).to(device)
                      }, torch.tensor(self.Lab[samples], dtype=torch.long).to(device)
    
    def one_epoch_batch_data_stream(self, batchSize=128, type='valid', device=torch.device('cpu')):
        if type == 'train':
            idList = self.trainIdList
        elif type == 'valid':
            idList = self.validIdList
        elif type == 'test':
            idList = self.testIdList
        X,XLen = self.tokenizedRNASent,self.RNASeqLen
        for i in range((len(idList)+batchSize-1)//batchSize):
            samples = idList[i*batchSize:(i+1)*batchSize]
            RNASeqMaxLen = XLen[samples].max()
            yield {
                    "seqArr":torch.tensor([i+[0]*(RNASeqMaxLen-len(i)) for i in X[samples]], dtype=torch.long).to(device), \
                    "seqLenArr":torch.tensor(XLen[samples], dtype=torch.int).to(device)
                  }, torch.tensor(self.Lab[samples], dtype=torch.long).to(device)