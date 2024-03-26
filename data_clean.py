import pandas as pd
import numpy as np
import os
import random
from feature import Graph_smiles
parent_dir = os.path.abspath(os.path.dirname(__file__))
print(parent_dir)

def statistics():
    df = pd.read_csv(parent_dir+"/dataset/train/data.csv", header=0, sep="\t")
    dic = df['Odor'].value_counts().to_dict()
    
    # delete number of odor < 10
    print(dic)
    drop_ls = []
    for key,value in dic.items():
        if value < 10:
            drop_ls.append(key)
    print(drop_ls)
    for odor in drop_ls:
        df = df.drop (df [df ['Odor'] == odor].index) 
    dic = df['Odor'].value_counts().to_dict()
    print(dic)

    ls = []
    for index,row in df.iterrows():
        if row['Odor'] == 'coffee':
            ls.append(1)
        else:
            ls.append(0)
    print(len(ls))

    df.insert(4,'Label',ls)
    df.to_csv(parent_dir+"/dataset/train/data_labeled.csv",header=True, sep="\t", index=False)


def data_preprocess():
    import rdkit
    import rdkit.Chem as Chem

    df = pd.read_csv(parent_dir+"/dataset/train/data_labeled.csv", header=0, sep="\t")
    print(df['Label'].value_counts())

    # delete can't recognize by rdkit
    none_list=[]
    for index, row in df.iterrows():
        if Chem.MolFromSmiles(row['Smiles']) is None:
            none_list.append(index)
    df=df.drop(none_list)
    print("rdkit dropping {} unidentified molecules, remain {} molecules".format(len(none_list), len(df)))

    # delete can't recognize by dgl
    none_list=[]
    for index, row in df.iterrows():
        try:
            Graph_smiles(row['Smiles'])
        except:
            none_list.append(index)
    df=df.drop(none_list)
    print("dlg dropping {} unidentified molecules, remain {} molecules".format(len(none_list), len(df)))
    

    # smiles to canonical smiles
    canonical_smileses = []
    for index, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['Smiles'])
        canonical_smileses.append(Chem.MolToSmiles(mol))
    df.insert(5,'Canonical_Smiles',canonical_smileses)

    # drop duplicates
    df_new = pd.DataFrame()
    for key in df['Label'].value_counts().to_dict().keys():
        df_ = df.query('Label == @key')
        df_ = df_.drop_duplicates(subset=['Canonical_Smiles'], keep='first')
        print(key, len(df_))
        df_new = pd.concat([df_new,df_],axis = 0)
    df_new = df_new.drop_duplicates(subset=['Canonical_Smiles'], keep='last')

    # no sample
    print(df_new['Label'].value_counts())
    df_new.to_csv(parent_dir+"/dataset/train/cleaned_data.csv", header=True, sep="\t", index=False)
    
    # sample
    df_new = pd.read_csv(parent_dir+"/dataset/train/cleaned_data.csv", header=0, sep="\t")
    df_coffee = df_new[df_new['Odor']=='coffee']
    df_noncoffee = df_new[df_new['Odor']!='coffee']
    # df_noncoffee = df_noncoffee.groupby('Odor', group_keys=False).apply(lambda x: x.sample(frac=(len(df_coffee))/len(df_noncoffee)))
    df_noncoffee = df_noncoffee.sample(n=len(df_coffee))
    df_new = pd.concat([df_coffee,df_noncoffee],axis=0)
    df_new.to_csv(parent_dir+"/dataset/train/cleaned_data_sampled.csv", header=True, sep="\t", index=False)
    print(df_new['Label'].value_counts())



if __name__ == "__main__":
    # statistics()
    data_preprocess()
