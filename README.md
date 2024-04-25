[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11067667.svg)](https://doi.org/10.5281/zenodo.11067667)

### 0. Install the python dependencies 
The main python dependencies are below
```
scikit-learn
dgl-cuda
dgl-life
pytorch
rdkit
numpy
pandas
matplotlib
```
### 1. How to use predcoffee
First you need prepare a csv file inculed a 'Smiles' header and a column molecular SMILES:
```
Smiles
CC1=CN=C(C=N1)C(C)C
CC[C@H](C)C=O
C1=COC(=C1)CSSCC2=CC=CO2
Oc1ccccc1
CC(C(C)O)O
CCOC=O
C1=CC(=CC=C1C=O)O
CC(=CCCC(=CCOC=O)C)C
```
If you want to https://zenodo.org/records/11067667 add other columns, please split by Tab('\t'). 

Please go to to download pretrained weights in corresponding folder.

Then please input commands below:
```
python predictor.py -i input.csv -o result.csv -m mlp
```
about arguments of predictors, you can run python preditor -h
### 2. How to reproduce the results of the paper
If you want to reproduce the results of performance comparision of MLP, SVM, RF and MPNN, please input below
```
python main.py
```
The command above will generate comparision result in ./analysis/model_result/performance.txt and 4 pretrained model in ./pretrained/  
Next you should finetune the KPGT with coffee/non-coffee odor dataset to generate KPGT-predcoffee result, 
first go to to https://zenodo.org/records/11067667 to download pretrained KPGT weights in corresponding folder. and input
```
cd kpgt
python load_data.py --dataset predcoffee --data_path ./datasets
python main.py
```
It will generate 5 pretrained predcoffee-kpgt models in kpgt/pretrained/ and finally will print the perfomance metrics like MLP, SVM, .etc above.

The performance metrics above are the result Figure 4 in the paper. 

The generated 5 models can be used to predict in Section 1.

If you want to reproduce the analysis part in the paper, please refer to next section.

### 3. Introduction for analysis codes
please go to https://zenodo.org/records/11067667 download full analysis folder

| folder | function | reference |
|---|---|---|
|analysis/cluster|molecular clustering|https://pubmed.ncbi.nlm.nih.gov/36077567/|
|analysis/dock|molecular docking|vina, openbabel, hippos-plif|
|explain.ipynb|mining molecular function groups|this paper, SHAP|
|analysis/model_result|plot performance comparision|this paper|
|analysis/property|factor analysis and t-test|this paper, rkdit|
|analysis/qm|quantum chemical calculation|Gaussian16, GaussView6|
|analysis/reduce|molecular dimension reduce|PCA,TSNE|
