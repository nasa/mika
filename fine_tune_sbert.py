"""
@author: hswalsh
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample, util, evaluation
import torch
from torch.utils.data import DataLoader
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import time
from math import floor

model = SentenceTransformer('msmarco-roberta-base-v3') # this is the model we are starting with and would like to fine tune further; possibly check input sequence length as well

start_time = time.time() # start timer

header_list = ['sent1', 'sent2', 'label']
llis_samples_df = pd.read_csv(os.path.join('data', 'LLIS','llis_bert_training_set.csv'), names = header_list)
llis_samples_sent1 = llis_samples_df['sent1'].tolist()
llis_samples_sent2 = llis_samples_df['sent2'].tolist()
llis_samples_label = llis_samples_df['label'].tolist() # this training data is not suitable for a query system... we need a query and result, not two sentences
  
# split into train, validation sets
num_samples = len(llis_samples_sent1)
num_train_samples = floor(num_samples*.9)

llis_samples_train = []
for i in range(0,num_train_samples):
    llis_samples_train.append(InputExample(texts=[llis_samples_sent1[i], llis_samples_sent2[i]],label=llis_samples_label[i]))
llis_validate_sent1 = llis_samples_sent1[num_train_samples:num_samples]
llis_validate_sent2 = llis_samples_sent2[num_train_samples:num_samples]
llis_validate_label = llis_samples_label[num_train_samples:num_samples]

# setup data loader and loss function
train_batch_size = 16
train_dataloader = DataLoader(llis_samples_train, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

# evaluator
evaluator = evaluation.EmbeddingSimilarityEvaluator(llis_validate_sent1, llis_validate_sent2, llis_validate_label)

# fine tune model
num_epochs = 2 # takes approx. 1 day per epoch on my macbook
warmup_steps = 0
model_save_path = os.path.join('models', 'fine_tuned_llis_model')
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, evaluation_steps=100, evaluator=evaluator, warmup_steps=warmup_steps, output_path=model_save_path)

fine_tune_time = time.time()
total_run_time = fine_tune_time - start_time

def print_runtime(run_time):
    if run_time < 60:
        print("--- %s seconds ---" % (run_time))
    else:
        print("--- %s minutes ---" % (run_time/60))

print('Total run time: ')
print_runtime(total_run_time)
