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
import time
from sklearn.model_selection import train_test_split

# to convert our data to the BERT language representation & use its vocabulary
class OurDataset(Dataset):
  def __init__(self, data, len_max):
    # expected data input is a pandas dataframe
    self.data = data
    self.data.reset_index(drop=True, inplace=True)
    # self.reviews = self.data['summary']
    # self.ratings = self.data['overall']
    self.len_max = len_max # the maximum length of a review to consider
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # add do_lower_case, if you want to do all lowercase text

  def __len__(self):
    return len(self.data)

  def __getitem__(self, ind):
    review = self.data.loc[ind, 'reviewText']
    rating = int(self.data.loc[ind, 'overall']) - 1 # Ratings=1,2,3,4,5
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
    # label = torch.tensor(rating).long()
    return tokens_to_tensors, attention_mask, rating


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
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    # self.bert = BertForSequenceClassification('bert-base-uncased', do_lower_case=True, num_labels=rating_scale) # this one has a linear layer after the pooled layer
    # because we want to fine-tune, make sure that the weights from BERT aren't updated
    for param in self.bert.parameters():
      param.requires_grad = False
    
    # our classifer on top of the BERT Model
    self.linear1 = nn.Linear(768, 500)
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(0.5) # dropout with 50%
    self.linear2 = nn.Linear(500, rating_scale)
    # self.fc = nn.LogSoftmax(dim=0) # to calculate the probabilities
    self.fc = nn.LogSoftmax(dim=1)

  def forward(self, tokens, attention_mask):
    # grab the BERT Model outputs after forward pass
    outputs = self.bert.forward(input_ids=tokens, attention_mask=attention_mask) 
    # forward pass for Bert will return two outputs
    # 12 layers to one pooled output, so grab last output layer
    # pooled output size should be (1, 768) as we pass one review at a time
    pooled_output = outputs.pooler_output
    #print(outputs.pooler_output)
    lin_output1 = self.linear1(pooled_output)
    relu_output = self.relu(lin_output1)
    dropped_output = self.drop(relu_output)
    lin_output2 = self.linear2(dropped_output)
    result = self.fc(lin_output2)
    return result

"""
  def forward(self, tokens, attention_mask):
    # grab the BERT Model outputs after forward pass
    output = self.bert(input_ids=tokens, attention_mask=attention_mask, return_dict=True) 
    # forward pass for Bert will return two outputs
    # 12 layers to one pooled output, so grab last output layer
    # pooled output size should be (1, 768) as we pass one review at a time
    # print(output.pooler_output[0].size())
    pooled_output = output['pooler_output'][0]
    lin_output1 = self.linear1(pooled_output)
    print("through first linear layer")
    relu_output = self.relu(lin_output1)
    print("ReLu'ed!!")
    dropped_output = self.drop(relu_output)
    print("dropped bish")
    lin_output2 = self.linear2(dropped_output)
    print("Linear 2, almost there")
    print(lin_output2.size())
    result = self.fc(lin_output2)
    print("Log soft max completo")
    print(result.size())
    return result
  """


# training and validation functions
def dataloader(fileName, bs):
  """Load the data into PyTorch DataLoader for train, val, test.
  """
  np.random.seed(42)
  # load data from fileName
  raw_data = pd.read_json(fileName, lines=True, orient='columns', dtype=True)
  raw_data = raw_data[['reviewText', 'overall']]
  raw_data = raw_data.dropna()
  # split the data into train, val, and test
  # reduce the original 200,000 to 50,000, with the original proportions of each class
  X_temp1, X_temp2, y_temp1, y_temp2 = train_test_split(raw_data['reviewText'], raw_data['overall'], test_size=0.25, stratify=raw_data['overall'], random_state=42)
  # now we have 50,000 examples in  X_temp2, y_temp2
  X_train, X_temp3, y_train, y_temp3 = train_test_split(X_temp2, y_temp2, test_size=0.5, stratify=y_temp2, random_state=42)
  # X_train now has 25,000 examples
  X_val, X_test, y_val, y_test = train_test_split(X_temp3, y_temp3, test_size=0.5, stratify=y_temp3, random_state=42)
  # X_val & X_test have 12,500 examples each
  # merge X and ys to feed into OurDataset
  train_set = pd.DataFrame(X_train)
  train_set['overall'] = y_train
  val_set = pd.DataFrame(X_val)
  val_set['overall'] = y_val
  test_set = pd.DataFrame(X_test)
  test_set['overall'] = y_test
  # train_set = raw_data.sample(frac=0.5, random_state=42)
  # temp = raw_data.drop(train_set.index)
  # val_set = temp.sample(frac=0.3, random_state=42)
  # test_set = temp.drop(val_set.index)
  # train_split = 0.5
  # val_split = 0.3
  # fullsize = len(raw_data)
  # indices = list(range(fullsize))
  # split1 = int(np.floor(train_split * fullsize))
  # split2 = int(np.floor(val_split * fullsize))
  # np.random.shuffle(indices)
  # train_ind, val_ind, test_ind = indices[:split1], indices[split1:split1+split2], indices[split1+split2:]
  # using the split indices, get the samples
  # train_sampler = torch.utils.data.SubsetRandomSampler(train_ind)
  # val_sampler = torch.utils.data.SubsetRandomSampler(val_ind)
  # test_sampler = torch.utils.data.SubsetRandomSampler(test_ind)
  # utilize OurDataset class to create & tokenize the data
  # all_data = OurDataset(raw_data, 186)
  train_data = OurDataset(train_set, 512)
  val_data = OurDataset(val_set, 512)
  test_data = OurDataset(test_set, 512)
  # use DataLoader
  train_loader = DataLoader(train_data, batch_size=bs, shuffle=False)
  val_loader = DataLoader(val_data, batch_size=bs, shuffle=False)
  test_loader = DataLoader(test_data, batch_size=bs) # ??
  return train_loader, val_loader, test_loader


def get_accuracy(pred, label):
  # determine the index of the most likely rating
  index = torch.argmin(pred, dim = 1) + 1
  # return the number of correctly predicted ratings / number of total examples in a batch
  return (index==label).sum().item() / len(label)


def evaluate(model, loader, criterion):
  """Evaluate the network model based on validation set.
  """
  #model.train(False)
  model.eval() # go into evaluation mode
  acc, err = 0, 0
  with torch.no_grad():
    total_loss = 0.0
    total_acc = 0.0
    for iter, (tokens, attention_mask, rating) in enumerate(loader):
      pred = model(tokens, attention_mask)
      # loss = criterion(nn.LogSoftmax(pred, dim=1), rating)
      loss = criterion(pred, rating)
      total_loss += loss.item()
      # total_acc += get_accuracy(nn.LogSoftmax(pred, dim=1), rating)
      total_acc += get_accuracy(pred, rating)
      if (iter + 1) % 100 == 0:
        print("Iter {}     -     Loss: {}     -       Accuracy: {}".format(iter+1, total_loss / (iter+1), total_acc / (iter+1)))
    
    err = (total_loss) / (iter + 1)
    acc = (total_acc) / (iter + 1) # average accuracy across all batches
  return err, acc


def train(model, train_loader, val_loader, epochs, learning_rate):
  """Use training and validation, train the model.
  """
  print("I'm at the start")
  model.train(True)
  torch.manual_seed(42) # for reproducibility
  # define loss function and optimizer for weight updates
  target_weights = torch.FloatTensor([20/41, 19/41, 8/41, 3/41, 1/41]) # Weights that should be multiplied to the learning rate of the optimizer
  criterion = nn.NLLLoss(weight=target_weights)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  # to store values later
  train_acc, train_loss, val_acc, val_loss = [], [], [], []
  startTime = time.time()
  for epoch in range(epochs):
    total_loss = 0.0
    total_acc = 0.0
    # iterate through the batches
    # iter = 0
    for iter, (tokens, attention_mask, rating) in enumerate(train_loader):
      optimizer.zero_grad()
      pred = model(tokens, attention_mask) # predict using tokens & attention mask
      # compute loss
      # loss = criterion(nn.LogSoftmax(pred, dim=1), rating)
      loss = criterion(pred, rating)
      # backprop
      loss.backward()
      # weight updates
      optimizer.step()
      # add in loss and accuracy (number of correctly predicted ratings)
      total_loss += (loss)
      total_acc += (get_accuracy(pred, rating))
    
      # for us to see where we are in training
      if (iter+1) % 100 == 0:
        print("Epoch {}  - Iteration {}  - Training Time: {} -     Loss: {}     -     Accuracy: {}".format(epoch+1, iter+1, time.time()-startTime,
        total_loss / (iter+1), total_acc / (iter+1)))

    train_loss.append(total_loss / (iter+1)) # calculate the average loss across all iterations per epoch
    train_acc.append(total_acc / (iter+1)) # calculate the average accuracy across all batches
    total_loss = 0.0
    # compute validation loss at the end of each epoch
    val_err, val_avg_acc = evaluate(model, val_loader, criterion)
    val_loss.append(val_err)
    val_acc.append(val_avg_acc)

    print("END  ---  Epoch {}  ---  Training Error: {}  ---   Validation Error: {}".format(
        epoch, train_loss[epoch], val_loss[epoch]
    ))

  return train_loss, train_acc, val_loss, val_acc


if __name__ == "__main__": 
  # load the data for training and validation, set aside the test
  train_loader, val_loader, test_loader = dataloader('train.json', bs=16)
  print("Finish loading our data splits!")
  
  # initiate instance of network
  mod = RatingPredictor(rating_scale=5)
  print("Initiated Instance of Our Network")

  # train and check validation
  # mod.train(True)
  train_loss, train_acc, val_loss, val_acc = train(mod, train_loader, val_loader, epochs=4, learning_rate=0.01)