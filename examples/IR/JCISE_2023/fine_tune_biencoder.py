"""
@author: hswalsh
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, datasets, losses

# load training data from file (generated using generate_training_data.py)
training_data = pd.read_csv('examples/IR/JCISE_2023/training_data.csv')

# define model
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')

# using InputExamples to format training data
training_data_list = training_data.values.tolist()
train_examples = []
for row in training_data_list:
    train_examples.append(InputExample(texts=[row[1], row[2]]))

# data loader and loss function
train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=32)
train_loss = losses.MultipleNegativesRankingLoss(model)

# fine tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)

# save the model
model.save('examples/IR/JCISE_2023/llis_tuned_biencoder')