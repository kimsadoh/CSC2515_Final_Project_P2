import torchvision
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import sklearn as sk

# to convert our data to the BERT language representation & use its vocabulary
class OurDataset(Dataset):
  def __init__(self, data, len_max):
    # expected data input is a pandas dataframe
    self.data = data
    # self.reviews = self.data['summary']
    # self.ratings = self.data['overall']
    self.len_max = len_max # the maximum length of a review to consider
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) # add do_lower_case, if you want to do all lowercase text

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
    attention_mask = (tokens_to_tensors != 0).long()
    # convert our labels into tensors
    label = torch.tensor(rating).long()
    return tokens_to_tensors, attention_mask, label


# to import the pretrained BERT Model
# purpose: multi-class classification where we try to predict the ratings
class RatingPredictor(nn.Module):
  def __init__(self, rating_scale):
    super(RatingPredictor, self).__init__()
    # load the BERT Model configuration, and update to match our dataset
    # self.bert_config = BertConfig(hidden_size=768,
    #                               num_hidden_layers=12,
    #                               num_attention_heads=12,
    #                               intermediate_size=3072,
    #                               num_labels=rating_scale) # change to multi-class
    # load in pretrained model
    self.bert = BertModel.from_pretrained('bert-base-uncased', do_lower_case=True)
    # self.bert = BertForSequenceClassification('bert-base-uncased', do_lower_case=True, num_labels=rating_scale) # this one has a linear layer after the pooled layer
    # because we want to fine-tune, make sure that the weights from BERT aren't updated
    for param in self.bert.parameters():
      param.requires_grad = False
    
    # our classifer on top of the BERT Model
    self.drop = nn.Dropout(0.5) # dropout with 50%
    self.fc = nn.LogSoftmax(nn.Linear(768, rating_scale), dim=1) # output of Bert Model is 768 to our rating scale

  def forward(self, tokens, attention_mask):
    # grab the BERT Model outputs after forward pass
    _, pooled_output = self.bert(input_ids=tokens, attention_mask=attention_mask) 
    # forward pass for Bert will return two outputs
    # 12 layers to one pooled output, so grab last output layer
    # pooled output size should be (1, 768) as we pass one review at a time
    pooled_output = self.drop(pooled_output)
    return self.fc(pooled_output)


# training and validation functions
def dataloader(fileName, bs):
  """Load the data into PyTorch DataLoader for train, val, test.
  """
  np.random.seed(42)
  # load data from fileName
  raw_data = pd.read_json(fileName, lines=True, orient='columns', dtype=True)
  raw_data = raw_data[['summary', 'overall']]
  # split the data into train, val, and test
  train_split = 0.5
  val_split = 0.3
  fullsize = len(raw)
  indices = list(range(fullsize))
  split1 = int(np.floor(train_split * fullsize))
  split2 = int(np.floor(val_split * fullsize))
  np.random.shuffle(indices)
  train_ind, val_ind, test_ind = indices[:split1], indices[split1:split1+split2], indices[split1+split2:]
  # using the split indices, get the samples
  train_sampler = SubsetRandomSampler(train_ind)
  val_sampler = SubsetRandomSampler(val_ind)
  test_sampler = SubsetRandomSampler(test_ind)
  # utilize OurDataset class to create & tokenize the data
  train_data = OurDataset(train_samples, 186)
  val_data = OurDataset(val_samples, 186)
  test_data = OurDataset(test_samples, 186)
  # use DataLoader
  train_loader = DataLoader(train_data, batch_size=bs, sampler=train_sampler, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=bs, sampler=val_sampler, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=bs, sampler=test_sampler) # ??
  return train_loader, val_loader, test_loader


def get_accuracy(pred, label):
  # determine the index of the most likely rating
  index = pred.max(dim=1)[1]
  return (index==label).sum().item()


def evaluate(model, loader, criterion):
  """Evaluate the network model based on validation set.
  """
  model.train(False)
  model.eval() # go into evaluation mode
  iter, acc, err = 0, 0, 0
  with torch.no_grad():
    total_loss = 0.0
    total_acc = 0.0
    for iter, (tokens, attention_mask, label) in enumerate(loader):
      pred = model(tokens, attention_mask)
      loss = criterion(pred, label)
      total_loss += loss.item()
      total_acc += get_accuracy(pred, label)
      iter += 1
    err = float(total_loss) / (iter)
    acc = float(total_acc) / (iter)
  return err, acc


def train(model, train_loader, val_loader, epochs, learning_rate):
  """Use training and validation, train the model.
  """
  # model.train(True)
  torch.manual_seed(42) # for reproducibility
  # define loss function and optimizer for weight updates
  target_weights = [20, 19, 8, 3, 1]# Weights that should be multiplied to the learning rate of the optimizer
  criterion = nn.NLLLoss(weight=torch.tensor(target_weights))
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  # to store values later
  train_acc, train_loss, val_acc, val_loss = [], [], [], []
  startTime = time.time()
  for epoch in range(epochs):
    total_loss = 0.0
    total_acc = 0.0
    # iterate through the batches
    for iter, (tokens, attention_mask, label) in enumerate(train_loader):
      optimizer.zero_grad()
      pred = model(tokens, attention_mask) # predict using tokens & attention mask
      # compute loss
      loss = criterion(pred, label)
      # backprop
      loss.backward()
      # weight updates
      optimizer.step()
      # add in loss and accuracy (number of correctly predicted ratings)
      total_loss += float(loss)
      total_acc += float(get_accuracy(pred, label)
    
      # for us to see where we are in training
      if ((iter+1) % 2000) == 0:
        print("Epoch {}  - Iteration {}  - Training Time: {}".format(epoch+1, iter+1, startTime-time.time()))

    train_loss.append(total_loss / (iter+1)) # calculate the average loss across all iterations per epoch
    train_acc.append(total_acc / (iter+1))
    total_loss = 0.0
    # compute validation loss at the end of each epoch
    val_err, val_acc = evaluate(model, val_loader, criterion)
    val_loss.append(val_error)
    val_acc.append(val_acc)

    print("END  ---  Epoch {}  ---  Training Error: {}  ---   Validation Error: {}".format(
        epoch, train_loss[epoch], val_loss[epoch]
    ))

  return train_loss, train_acc, val_loss, val_acc

if __name__ == "__main__": 
  # initiate instance of network
  mod = RatingPredictor(rating_scale=5)
  # load the data for training and validation, set aside the test
  train, val, test = dataloader('train.json', bs=16)