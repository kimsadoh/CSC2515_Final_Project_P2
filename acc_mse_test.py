# to measure accuracy and MSE
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch

def get_test_accuracy(pred, label):
  # determine the index of the most likely rating
  # index = torch.argmin(pred, dim = 1)
  # return the number of correctly predicted ratings / number of total examples in a batch
  return (pred==label).sum().item()

def evaluate_test(model, loader, loader_len):
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
      pred_temp1 = pred.argmax(dim=1)
      loss = mean_squared_error(pred_temp1.numpy(), rating.numpy())
      total_loss += loss
      total_acc += get_test_accuracy(pred_temp1, rating)
      if (iter + 1) % 100 == 0:
        print("Iter {}     -     Loss: {}     -       Accuracy: {}".format(iter+1, total_loss / (iter+1), total_acc / (iter+1)))
    
    err = (total_loss) / (iter + 1)
    acc = (total_acc) / (loader_len) # the total number of correctly predicted 
    return err, acc

# now to check test set accuracy
test_err, test_acc = evaluate_test(mod, test_loader, test_len)