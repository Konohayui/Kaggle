import os

from tqdm.auto import tqdm
from scipy.stats import rankdata

import numpy as np
import pandas as pd

import Levenshtein

import matplotlib.pyplot as plt
import seaborn as sns

two_colors = sns.xkcd_palette(['red', 'bright blue'])

# Plot rank distributions
def plot_rank_dist(name, ax, show_del=False):

    sns.kdeplot(
        data=test_df.query('type=="SUB"'),
        x='{}_rank'.format(name),
        bw_adjust=0.3,
        lw=3,
        label='SUB',
        ax=ax,
        color='k'
    )

    ax.vlines(
        test_df.query('type=="DEL"')['{}_rank'.format(name)],
        ax.get_ylim()[0],
        ax.get_ylim()[1],
        lw=5,
        label='DEL',
        color=two_colors[0]
    )

    ax.vlines(
        test_df.query('type=="WT"')['{}_rank'.format(name)],
        ax.get_ylim()[0],
        ax.get_ylim()[1],
        lw=5,
        label='WT',
        color=two_colors[1]
    )

    if show_del:
        sns.kdeplot(
            data=test_df.query('type=="DEL"'),
            x='{}_rank'.format(name),
            bw_adjust=0.3,
            lw=3,
            label='DEL',
            ax=ax,
            color=two_colors[0]
        )

        ax.vlines(
            test_df.query('type=="DEL"')['{}_rank'.format(name)],
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            lw=5,
            label='DEL',
            color=two_colors[0]
        )

    ax.set_xlim(-50,2550)
    ax.set_title('{} rank distribution'.format(name), fontsize=20)
    ax.set_xlabel('{}_rank'.format(name), fontsize=20)
    ax.set_ylabel('Density', fontsize=20)

    ax.tick_params(labelsize=12)
    ax.legend(loc=1)

    return ax

# Wild type sequence provided in the "Dataset Description":
wt = 'VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK'

# Read testing set sequences and pH:
test_df = pd.read_csv('../input/novozymes-enzyme-stability-prediction/test.csv')

# Add mutation information to testing set:
result = []
for _, row in test_df.iterrows():
    ops = Levenshtein.editops(wt, row['protein_sequence'])
    assert len(ops) <= 1
    if len(ops) > 0 and ops[0][0] == 'replace':
        idx = ops[0][1]
        result.append(['SUB', idx + 1, wt[idx], row['protein_sequence'][idx]])
    elif len(ops) == 0:
        result.append(['WT', 0, '', ''])
    elif ops[0][0] == 'insert':
        assert False, "Ups"
    elif ops[0][0] == 'delete':
        idx = ops[0][1]
        result.append(['DEL', idx + 1, wt[idx], '_'])
    else:
        assert False, "Ups"

test_df = pd.concat([test_df, pd.DataFrame(data=result, columns=['type', 'resid', 'wt', 'mut'])], axis=1)

! wget https://ftp.ncbi.nih.gov/blast/matrices/BLOSUM100 -O BLOSUM100.txt

def blosum_apply(row):
    if row['type'] == 'SUB':
        return blosum.loc[row['wt'], row['mut']]
    elif row['type'] == 'DEL':
        return -10
    elif row['type'] == 'WT':
        return 0
    else:
        assert False, "Ups"
blosum = pd.read_csv('./BLOSUM100.txt', sep='\s+', comment='#')
test_df['blosum'] = test_df.apply(blosum_apply, axis=1)
test_df['blosum_rank'] = rankdata(test_df['blosum'])
!rm ./BLOSUM100.txt

fig, ax = plt.subplots(figsize=(25, 5))
plot_rank_dist(name='blosum', ax=ax, show_del=False)
plt.show()

# Read AlphaFold2 result for wild type sequence:
plddt = (
    pd.read_csv('../input/novozymes-enzyme-stability-prediction/wildtype_structure_prediction_af2.pdb', sep='\s+', header=None)[[0,5,10]]
    .rename(columns={0:'atom', 5:'resid', 10:'plddt'})
    .query('atom=="ATOM"')
    .drop_duplicates()
)

# Add B factor to the testing set:
test_df = pd.merge(
    test_df,
    plddt,
    left_on='resid',
    right_on='resid',
    how='left'
)

test_df['plddt_rank'] = rankdata(-1*test_df['plddt'])
fig, ax = plt.subplots(figsize=(25, 5))
plot_rank_dist(name='plddt', ax=ax, show_del=True)
plt.show()

plddtdiff = []

# Wild type result:
wt_plddt = (
    pd.read_csv('../input/nesp-kvigly-test-mutation-pdbs/WT_unrelaxed_rank_1_model_3.pdb', sep='\s+')
    .loc['ATOM'].reset_index()
    .loc[:, ['level_4', 'MODEL']].drop_duplicates()
    .rename(columns={'level_4':'resid', 'MODEL':'plddt'})
    .astype({'resid':int})
    .set_index('resid')
)

# Add difference in pLDDTto the testing set:>
for _,row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    file_path = '../input/nesp-kvigly-test-mutation-pdbs/{}{}{}_unrelaxed_rank_1_model_3.pdb'.format(row['wt'], row['resid'], row['mut'])
    if os.path.exists(file_path):
        tdf = (
            pd.read_csv(file_path, sep='\s+')
            .loc['ATOM'].reset_index()
            .loc[:, ['level_4', 'MODEL']].drop_duplicates()
            .rename(columns={'level_4':'resid', 'MODEL':'plddt'})
            .astype({'resid':int})
            .set_index('resid')
        )
        plddtdiff.append((tdf.loc[row['resid']] - wt_plddt.loc[row['resid']]).values[0])
    else:
        plddtdiff.append(np.nan)

test_df['plddtdiff'] = plddtdiff
test_df['plddtdiff_rank'] = rankdata(test_df['plddtdiff'])

fig, ax = plt.subplots(figsize=(25, 5))
plot_rank_dist(name='plddtdiff', ax=ax, show_del=True)
plt.show()

# Run DeepDDG on http://protein.org.cn/ddg.html by uploading the PDB file and clicking "Submit":>
ddg = pd.read_csv('../input/novozymes/ddgout.txt', sep='\s+', usecols=[0,1,2,3,4]).rename(columns={'WT':'wt', 'ResID':'resid', 'Mut':'mut'})

# Add DeepDDG output to the testing set:
test_df = pd.merge(
    test_df.set_index(['wt','resid','mut']),
    ddg.set_index(['wt','resid','mut']),
    left_index=True,
    right_index=True,
    how='left'
).reset_index()

test_df.loc[test_df['type']=='WT','ddG'] = 0
test_df.loc[test_df['type']=='DEL','ddG'] = test_df['ddG'].dropna().median()

test_df['ddG_rank'] = rankdata(test_df['ddG'])
fig, ax = plt.subplots(figsize=(25, 5))
plot_rank_dist(name='ddG', ax=ax, show_del=False)
plt.show()

# Run DeMaSk on https://demask.princeton.edu/query/ by pasting the wild type sequence and clicking "Compute":
demask = pd.read_csv('../input/novozymes/demaskout.txt', sep='\t', usecols=[0,1,2,3], names=['resid','wt','mut','demask'], skiprows=1)

# Add DeMask output to the testing set:
test_df = pd.merge(
    test_df.set_index(['wt','resid','mut']),
    demask.set_index(['wt','resid','mut']),
    left_index=True,
    right_index=True,
    how='left'
).reset_index()

test_df.loc[test_df['type']=='WT','demask'] = 0
test_df.loc[test_df['type']=='DEL','demask'] = test_df['demask'].dropna().min()


test_df['demask_rank'] = rankdata(test_df['demask'])
fig, ax = plt.subplots(figsize=(25, 5))
plot_rank_dist(name='demask', ax=ax, show_del=False)
plt.show()

# Read VMD/NAMD output:
namd = pd.read_csv('../input/novozymes-md2/residue_rmsd_sasa_last.dat', sep='\t', header=None, names=['resid','rmsd','sasa0','sasaf'])

# Add VMD/NAMD results to the testing set:
test_df = pd.merge(
    test_df,
    namd[['resid','rmsd']],
    left_on='resid',
    right_on='resid',
    how='left'
)

test_df.loc[test_df['type']=='WT','rmsd'] = test_df['rmsd'].dropna().max()
# test_df.loc[test_df['type']=='WT','sasaf'] = test_df['sasaf'].dropna().max()

test_df['rmsd_rank'] = rankdata(test_df['rmsd'])
# test_df['sasaf_rank'] = rankdata(test_df['sasaf'])
fig, ax = plt.subplots(figsize=(25, 5))
plot_rank_dist(name='rmsd', ax=ax, show_del=True)
plt.show()

# Read VMD/NAMD output:
namd = pd.read_csv('../input/novozymes-md/residue_rmsd_sasa_last.dat', sep='\t', header=None, names=['resid','rmsd','sasa0','sasaf'])

# Add VMD/NAMD results to the testing set:
test_df = pd.merge(
    test_df,
    namd[['resid','sasaf']],
    left_on='resid',
    right_on='resid',
    how='left'
)

# test_df.loc[test_df['type']=='WT','rmsd'] = test_df['rmsd'].dropna().max()
test_df.loc[test_df['type']=='WT','sasaf'] = test_df['sasaf'].dropna().max()

# test_df['rmsd_rank'] = rankdata(test_df['rmsd'])
test_df['sasaf_rank'] = rankdata(test_df['sasaf'])
fig, ax = plt.subplots(figsize=(25, 5))
plot_rank_dist(name='sasaf', ax=ax, show_del=True)
plt.show()

test_df['rosetta_rank'] = pd.read_csv('../input/nesp-relaxed-rosetta-scores/submission_rosetta_scores')['tm']
fig, ax = plt.subplots(figsize=(25, 5))
plot_rank_dist(name='rosetta', ax=ax, show_del=False)
plt.show()


test_df['thermonet'] = pd.read_csv('../input/nesp-thermonet-v2/submission.csv')['tm']
test_df['thermonet_rank'] = rankdata(test_df['thermonet'])
fig, ax = plt.subplots(figsize=(25, 5))
plot_rank_dist(name='thermonet', ax=ax, show_del=False)
plt.show()

def rank_nrom(name):
    s = test_df['{}_rank'.format(name)]
    return s/s.max()
# Scale ranks prior to ensembling:
# Global ensemble:
test_df['tm'] = (
    4 * rank_nrom('rosetta') + 2*rank_nrom('rmsd') + 2*rank_nrom('thermonet') + 2*rank_nrom('plddtdiff') +\
    rank_nrom('sasaf') + rank_nrom('plddt') + rank_nrom('demask') + rank_nrom('ddG') + rank_nrom('blosum')
) / 14

test_df['tm'] = test_df['tm']/test_df['tm'].max()
# Deletion type:
idx = test_df[test_df['type']=="DEL"].index
test_df.loc[idx, 'tm'] =  (2*rank_nrom('plddt')[idx] + 3*rank_nrom('plddtdiff')[idx] + rank_nrom('rmsd')[idx] + rank_nrom('sasaf')[idx]) / 7

# Wild type:
test_df.loc[test_df['type']=="WT",'tm'] = test_df['tm'].max()+1
test_df['tm'] = rankdata(test_df['tm'])
test_df['tm_rank'] = test_df['tm']

# Submission:
test_df[['seq_id','tm']].to_csv('submission.csv', index=False)
fig, ax = plt.subplots(figsize=(25, 5))
plot_rank_dist(name='tm', ax=ax, show_del=True)
plt.show()



fig, axs = plt.subplots(nrows=10, figsize=(20,40), gridspec_kw={'hspace':0.5})


plot_rank_dist(name='rosetta', ax=axs[0], show_del=False)
plot_rank_dist(name='rmsd', ax=axs[1], show_del=True)
plot_rank_dist(name='thermonet', ax=axs[2], show_del=False)
plot_rank_dist(name='plddtdiff', ax=axs[3], show_del=True)
plot_rank_dist(name='sasaf', ax=axs[4], show_del=True)
plot_rank_dist(name='plddt', ax=axs[5], show_del=True)
plot_rank_dist(name='demask', ax=axs[6], show_del=False)
plot_rank_dist(name='ddG', ax=axs[7], show_del=False)
plot_rank_dist(name='blosum', ax=axs[8], show_del=False)

plot_rank_dist(name='tm', ax=axs[9], show_del=True)

plt.show()




