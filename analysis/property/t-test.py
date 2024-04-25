import pandas as pd
import os
# import joypy
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import kstest
import scipy.stats as stats
from scipy.stats import shapiro

parent_dir = os.path.abspath(os.path.dirname(__file__))

def test_all(df,name):
    A = df.query("Label==0")[name]
    B = df.query("Label==1")[name]
    if shapiro(A).pvalue > 0.01 and shapiro(B).pvalue > 0.01:
        if stats.levene(A,B).pvalue > 0.01:
            return stats.ttest_ind(A,B,equal_var=True).pvalue
        else:
            return stats.ttest_ind(A,B,equal_var=False).pvalue

    else:
        return stats.mannwhitneyu(A,B,alternative='two-sided').pvalue

def plot_bibox(A,B,labelA,labelB,title):
    pv = test_all(df,title)
    if pv < 0.01:
        color="#B22222"
    else:
        color="#00008B"
    plt.boxplot([A, B],
                showfliers=False,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '2'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                labels=[labelA,labelB])
    plt.title(title, fontsize = 14, fontweight="bold",)     # ,fontsize = 8
    plt.text(0.25,0.85,"P={:.2e}".format(pv),
             fontsize=16,
             fontweight="medium",
             color=color,
             transform=ax.transAxes)


# print(len(columns))
# print(len(df[df['Label']==0]))
# print(len(df[df['Label']==1]))

# print(test_all(df,'HeavyAtomMolWt'))

df = pd.read_csv(parent_dir+"/all_descriptors.csv", sep="\t", header=0)
columns = ['HeavyAtomMolWt', 'ExactMolWt', 'MW', 'HeavyAtomCount', 'LabuteASA', 'pyLabuteASA', 'NumValenceElectrons', 'FpDensityMorgan2', 'FpDensityMorgan3', 'FpDensityMorgan1', 'MolLogP', 'ALOGP',]


plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes',linewidth=1.5)
fig = plt.figure(figsize=(16,13),dpi=600)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2,hspace=0.3)
for i in range(12):
    ax=plt.subplot(3,4,i+1)
    plot_bibox(df[df['Label']==0][columns[i]],df[df['Label']==1][columns[i]],"non-coffee","coffee",'{}'.format(columns[i]))
plt.savefig(parent_dir+'/ttest.png',bbox_inches="tight")
plt.close()


