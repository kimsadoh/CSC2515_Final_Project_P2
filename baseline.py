import torchvision
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# to convert our data to the BERT language representation
class OurDataset(Dataset):
  def __init__(self, data, len_max):
    # expected data input is a pandas dataframe
    self.data = data
    self.len_max = len_max # the maximum length of a review to consider
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # add do_lower_case, if you want to do all lowercase text

  def __len__(self):
    return len(self.data)

  def __getitem__(self, ind):
    review = self.data.loc[ind, 'summary']
    rating = self.data.loc[ind, 'overall'] # Ratings=1,2,3,4,5
    # use the BERT Tokenizer to ensure review is represented similarly
    tokens = self.tokenizer.tokenize(review)
    # recall that BERT uses additional token embeddings
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    # [CLS] should be added to the beginning of the input
    # [SEP] should be added to the end of the input
    # to add [PAD] if sentence is too short
    if len(tokens) < self.len_max:
      # At the end of the tokens add PAD
      tokens = tokens + ['[PAD]' for i in range(self.len_max - len(tokens))]
    else:
      # tokens list is too long, need to cut off the tokens and then re-add SEP
      tokens = tokens[:self.len_max - 1] + ['[SEP]']

    # Converts tokens to an id using the vocabulary
    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    # convert to PyTorch tensor
    tokens_to_tensors = torch.tensor(token_ids)
    # MLM to distinguish between the PAD and the important tokens
    mask = (tokens_to_tensors != 0).long()
