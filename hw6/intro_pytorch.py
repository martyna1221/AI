import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

def get_data_loader(training=True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    # two choices based on boolean input parameter 'training'
    if training == True:
        train_set = datasets.MNIST('./data', train=True, download=True, transform=custom_transform)
        loader = torch.utils.data.DataLoader(train_set, batch_size=50)
        return loader
    if training == False:
        test_set = datasets.MNIST('./data', train=False, transform=custom_transform)
        loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=False)
        return loader

def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128), 
        nn.ReLU(), 
        nn.Linear(128, 64), 
        nn.ReLU(),
        nn.Linear(64, 10)
        )
    return model

def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    model.train()
    size = 60000
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            # loss calculation
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            # accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        running_loss = "{:.3f}".format(running_loss / size)
        percent = "{:.2f}".format((correct / total) * 100)
        print("Train Epoch: %d Accuracy: %i/%i(%s%%) Loss: %s" % (epoch, correct, total, percent, running_loss))

def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    size = 10000
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
          inputs, labels = data
          outputs = model(inputs)
          # loss calculation
          loss = criterion(outputs, labels)
          running_loss += loss.item()
          # accuracy calculation
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    running_loss = "{:.4f}".format(running_loss / size)       
    percent = "{:.2f}".format((correct / total) * 100)

    if show_loss == True:
        print("Average loss: %s" % (running_loss))
        print("Accuracy: %s%%" % (percent))
    if show_loss == False:
        print("Accuracy: %s%%" % (percent))

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    custom_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])

    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    model.eval()

    with torch.no_grad():
      images = test_images[index]
      outputs = model(images)
      prob = F.softmax(outputs, dim=1)
      res, ind = prob.topk(3)
      print("%s: %s%%" % (class_names[ind[0][0]], "{:.2f}".format(float(res[0][0]) * 100)))
      print("%s: %s%%" % (class_names[ind[0][1]], "{:.2f}".format(float(res[0][1]) * 100)))
      print("%s: %s%%" % (class_names[ind[0][2]], "{:.2f}".format(float(res[0][2]) * 100)))

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    
    # for predict_label() -> this is my toy set
    custom_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
    x = datasets.MNIST('./data', train=False, transform=custom_transform)
    pred_set = torch.stack([x[0][0], x[1][0], x[2][0], x[3][0], x[4][0]])
