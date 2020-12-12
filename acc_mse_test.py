# to measure accuracy and MSE
import baseline
from sklearn.metrics import mean_squared_error
import torch.nn as nn

# load the data for training and validation, set aside the test
train_loader, val_loader, test_loader, train_len, val_len, test_len = baseline.dataloader('train.json', bs=16)
print("Finish loading our data splits!")
  
# initiate instance of network
mod = baseline.RatingPredictor(rating_scale=5)
print("Initiated Instance of Our Network")

# train and check validation
# mod.train(True)
train_loss, train_acc, val_loss, val_acc = baseline.train(mod, train_loader, val_loader, train_len, val_len, epochs=4, learning_rate=0.01)

criterion = nn.MSELoss()

# now to check test set accuracy
test_err, test_acc = baseline.evaluate(mod, test_loader, test_len, criterion)