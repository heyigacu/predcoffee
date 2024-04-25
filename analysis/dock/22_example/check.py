import pandas as pd
import os


PARENT_DIR = os.path.abspath(os.path.dirname(__file__))
df = pd.read_csv(PARENT_DIR+'/smiles_plif_hippos.csv',sep='\t',header=0)
bool_index = (df['PlifHippos'].str.startswith('0') | df['PlifHippos'].str.startswith('1'))
df = df[bool_index]
df.to_csv(PARENT_DIR+'/check_plif_hippos.csv',index=True,sep='\t',header=True)




