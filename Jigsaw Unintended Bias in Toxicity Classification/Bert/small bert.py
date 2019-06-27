import numpy as np 
import pandas as pd

# Installing Nvidia Apex
! pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import pkg_resources
import seaborn as sns
import time
import scipy.stats as stats
import gc
import re
import operator 
import sys
from sklearn import metrics
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from nltk.stem import PorterStemmer
from sklearn.metrics import roc_auc_score
%load_ext autoreload
%autoreload 2
%matplotlib inline
from tqdm import tqdm, tqdm_notebook
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings(action='once')
import pickle
from apex import amp
import shutil


device=torch.device('cuda')

MAX_SEQUENCE_LENGTH = 220
SEED = 1234
EPOCHS = 1
Data_dir="../input/jigsaw-unintended-bias-in-toxicity-classification"
Input_dir = "../input"
WORK_DIR = "../working/"
num_to_load=1000000                         #Train size to match time limit
valid_size= 100000                          #Validation Size
TOXICITY_COLUMN = 'target'


# Add the Bart Pytorch repo to the PATH
# using files from: https://github.com/huggingface/pytorch-pretrained-BERT
package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.insert(0, package_dir_a)

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam

# Translate model from tensorflow to pytorch
BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
    BERT_MODEL_PATH + 'bert_model.ckpt',
BERT_MODEL_PATH + 'bert_config.json',
WORK_DIR + 'pytorch_model.bin')

shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'bert_config.json')

# This is the Bert configuration file
from pytorch_pretrained_bert import BertConfig

bert_config = BertConfig('../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'+'bert_config.json', hidden_dropout_prob = 0.3)

# Converting the lines to BERT format
# Thanks to https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming
def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm_notebook(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)
    

from torch.utils import data
from tqdm import tqdm_notebook as tqdm

class LenMatchBatchSampler(data.BatchSampler):
    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx][0] == 0)
            count_zeros = int(count_zeros / 64) 
            if len(buckets[count_zeros]) == 0:  buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, "produced an inccorect number of batches. expected %i, but yielded %i" %(len(self), yielded)

def trim_tensors(tsrs):
    max_len = torch.max(torch.sum( (tsrs[0] != 0  ), 1))
    if max_len > 2: 
        tsrs = [tsr[:, :max_len] for tsr in tsrs]
    return tsrs 
    
BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'    

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)
train_df = pd.read_csv(os.path.join(Data_dir,"train.csv")).sample(num_to_load+valid_size,random_state=SEED)
print('loaded %d records' % len(train_df))

# Make sure all comment_text values are strings
train_df['comment_text'] = train_df['comment_text'].astype(str) 

sequences = convert_lines(train_df["comment_text"].fillna("DUMMY_VALUE"),MAX_SEQUENCE_LENGTH,tokenizer)
train_df=train_df.fillna(0)
# List all identities
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
y_columns=['target']

train_df = train_df.drop(['comment_text'],axis=1)
# convert target to 0,1
train_df['target']=(train_df['target']>=0.5).astype(float)

# Overall
weights = np.ones((len(sequences),)) / 4
# Subgroup
weights += (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
# Background Positive, Subgroup Negative
weights += (((train_df['target'].values>=0.5).astype(bool).astype(np.int) +
    (train_df[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
# Background Negative, Subgroup Positive
weights += (((train_df['target'].values<0.5).astype(bool).astype(np.int) +
       (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
       
       
train_y = np.vstack([train_df['target'].values, weights]).T
train_aux_y = train_df[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values 

y_train = np.hstack([train_y, train_aux_y])

X = sequences[:num_to_load]                
y = y_train[:num_to_load]
X_val = sequences[num_to_load:]                
y_val = y_train[num_to_load:]

test_df=train_df.tail(valid_size).copy()
train_df=train_df.head(num_to_load)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.long), torch.tensor(y,dtype=torch.float))

def custom_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])
    
    return bce_loss_1 + bce_loss_2
    
output_model_file = "bert_pytorch.bin"

lr=2e-5
batch_size = 32
accumulation_steps=2
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

model = BertForSequenceClassification.from_pretrained("../working",cache_dir=None,num_labels=7)
model.zero_grad()
model = model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
train = train_dataset

num_train_optimization_steps = int(EPOCHS*len(train)/batch_size/accumulation_steps)

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=lr,
                     warmup=0.05,
                     t_total=num_train_optimization_steps)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
model=model.train()

tq = tqdm_notebook(range(EPOCHS))
for epoch in tq:
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)
    optimizer.zero_grad()   # Bug fix - thanks to @chinhuic
    for i,(x_batch, y_batch) in tk0:
#        optimizer.zero_grad()
        y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        loss =  custom_loss(y_pred,y_batch.to(device))
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()
        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
        tk0.set_postfix(loss = lossf)
        avg_loss += loss.item() / len(train_loader)
        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)


torch.save(model.state_dict(), output_model_file)

model = BertForSequenceClassification(bert_config,num_labels=7)
model.load_state_dict(torch.load(output_model_file ))
model.to(device)
for param in model.parameters():
    param.requires_grad=False
model.eval()
valid_preds = np.zeros((len(X_val)))
valid = torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.long))
valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)

tk0 = tqdm_notebook(valid_loader)
for i,(x_batch,)  in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
    valid_preds[i*32:(i+1)*32]=pred[:,0].detach().cpu().squeeze().numpy()
    
def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]>0.5
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)



SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]>0.5]
    return compute_auc((subgroup_examples[label]>0.5), subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[(df[subgroup]>0.5) & (df[label]<=0.5)]
    non_subgroup_positive_examples = df[(df[subgroup]<=0.5) & (df[label]>0.5)]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[(df[subgroup]>0.5) & (df[label]>0.5)]
    non_subgroup_negative_examples = df[(df[subgroup]<=0.5) & (df[label]<=0.5)]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]>0.5])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)
    
 MODEL_NAME = 'model1'
test_df[MODEL_NAME]=torch.sigmoid(torch.tensor(valid_preds)).numpy()
TOXICITY_COLUMN = 'target'
bias_metrics_df = compute_bias_metrics_for_model(test_df, identity_columns, MODEL_NAME, 'target')
bias_metrics_df
get_final_metric(bias_metrics_df, calculate_overall_auc(test_df, MODEL_NAME))

