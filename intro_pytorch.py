import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

def get_data_loader(training = True):
	custom_transform=transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))
	])

	train_set = datasets.MNIST('./data', train=True, download=True, transform=custom_transform)
	test_set = datasets.MNIST('./data', train=False, transform=custom_transform)

	if (training):
		loader = torch.utils.data.DataLoader(train_set, batch_size = 50)
		return loader
	else:
		loader = torch.utils.data.DataLoader(test_set, batch_size = 50, shuffle=False)
		return loader

def build_model():
  model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64,10))
  return model

def train_model(model, train_loader, criterion, T):
  #criterion = nn.CrossEntropyLoss()
  model.train()
  opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  for epoch in range(T):
    running_loss = 0.0
    correct = 0
    for i, data in enumerate(train_loader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data

      # zero the parameter gradients
      opt.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      opt.step()

      # updating running loss
      running_loss += loss.item()

    # print stats after each epoch
    string = 'Train Epoch: ' + str(epoch)
    string = string + '  Accuracy: ' + str(correct) + '/' + '60000'
    string = string + '(' + str(round((correct/60000),2)) + '%)'
    string = string + '  Loss: ' + str(round(running_loss/60000,3))

    print(string)


def evaluate_model(model, test_loader, criterion, show_loss = True):
  crit = nn.CrossEntropyLoss()
  model.eval()

  with torch.no_grad():
    tot = 0
    correct = 0
    loss = 0
    for data, label in test_loader:
      output = model(data)
      loss = crit(output, label)

      tot += data.size(0)
      correct += (torch.argmax(output, 1) == label).sum().item()
      loss += loss.item()

    stat = 100*(correct/tot)

    if show_loss == True:
      stat = loss/len(test_loader.dataset)
      print('Average loss: {:.4f}'.format(stat))
      print('Accuracy: {:.2f}%'.format(stat))
    else:
      print('Accuracy: {:.2f}%'.format(stat))

def predict_label(model, test_images, index):
  model.eval()
  classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
  prob = F.softmax(model(test_images[index]), dim=1)
  top, indexs = torch.topk(prob.flatten(),3)

  for i in range(3):
    print('{}: {:.2f}%'.format(classes[indexs[i]], top[i]*100))

if __name__ == '__main__':
	criterion = nn.CrossEntropyLoss()
	train_loader = get_data_loader()
	test_loader = get_data_loader(False)
	model = build_model()
	train_model(model, train_loader, criterion, T = 5)
	evaluate_model(model, test_loader, criterion, show_loss = False)
	evaluate_model(model, test_loader, criterion, show_loss = True)
	pred_set, test_images = iter(test_loader).next()
	predict_label(model, pred_set, 1)  
