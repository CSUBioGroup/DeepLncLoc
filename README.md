# DeepLncLoc
A deep learning-based lncRNA subcellular localization predictor



# Usage
## How to train the model
You can train the model with a very simple way by the command blow:
***python train.py --k 3 --d 64 --s 64 --f 128 --metrics MaF --device "cuda:0"***
>***k*** is the value of the k-mers features.  
>***d*** is the dimension of vector of k-mer features which are trained by gensim library.  
>***s*** is the number of subsequences.  
>***f*** is the filter number in the CNN layer.  
>***metrics*** is the evaluation metrics in the training process. "MaF" for macro f1, "ACC" for accuracy, "MaAUC" for macro auc, "MiAUC" for micro auc.  
>***device*** is the device you used to build and train the model. It can be "cpu" for cpu or "cuda" for gpu, and "cuda:0" for gpu 0.  

Also you can use the package provided by us to train your model.  
First, you need to import the package.  
```python
from model.utils import *
from model.DL_ClassifierModel import *
```
Then you need to create the data object and get the embedding features.  
```python
dataClass = DataClass('data.txt', validSize=0.2, testSize=0.0, kmers=3)
dataClass.vectorize("char2vec", feaSize=64)
```
Finally, you can create the model object and start training.
```python
s,f,k,d = 64,128,3,64
model = TextClassifier_SPPCNN(classNum=5, embedding=dataClass.vector['embedding'], SPPSize=s, feaSize=d, filterNum=f, contextSizeList=[1,3,5], embDropout=0.3, fcDropout=0.5, useFocalLoss=True, device="cuda")
model.cv_train(dataClass, trainSize=1, batchSize=16, stopRounds=200, earlyStop=10, epoch=100, kFold=5, savePath=f"out/DeepLncLoc_s{s}_f{f}_k{k}_d{d}", report=['ACC','MaF','MiAUC','MaAUC'])
```

==Note that the model need to be named as "..._sx_fx_kx_dx" ('x' represents the parameters' value) , therefore we can get the model parameters from the name to better initialize the model architecture while in prediction.==

## How to do prediction
First, import the package. 
```python
from model.lncRNA_lib import *
```
Then instantiate an object.
```python
model = lncRNALocalizer(weightPath="out/xxx_sx_fx_kx_dx.pkl", classNum=5, contextSizeList=[1,3,5], map_location={"cuda:0":"cpu"}, device="cpu")
```
Finally, do the prediction.
```python
res = model.predict("ATCG...")
print(res)
```
## Independent test set
The test_set.text in Independent_test_set folder is used in comparison with lncLocator and iLoc-lncRNA. The final prediction results of three predictors (DeepLncLoc, lncLocator, iLoc-lncRNA) can be seen the supplementary materials of the paper.

## Other details
The other details can see the paper and the codes.

# Citation
Min Zeng, Yifan Wu, Chengqian Lu, Fuhao Zhang, Fang-Xiang Wu, Min Li. DeepLncLoc: a deep learning framework for long non-coding RNA subcellular localization prediction based on subsequence embedding. Briefings in Bioinformatics 23 (1), 2022, bbab360.

# License
This project is licensed under the MIT License - see the LICENSE.txt file for details
