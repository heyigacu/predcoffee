import pandas as pd
import rdkit 
from rdkit import Chem
import os

PARENT_DIR = os.path.abspath(os.path.dirname(__file__))
df = pd.read_csv(PARENT_DIR+'/cleaned_data_sampled.csv', header=0, sep='\t')
df = df[['Smiles','Label']]
df.columns = ['smiles','Class']
from sklearn.utils import shuffle
df = shuffle(df)
df.to_csv(PARENT_DIR+'/predcoffee.csv', header=True, sep=',', index=False)