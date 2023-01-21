import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import cv2
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from torch import Tensor

import time
import os
import numpy as np

import PIL.Image
import sklearn.metrics
from sklearn.metrics import average_precision_score

from typing import Callable, Optional
from YourNetwork1 import *
from RainforestDataset1 import *

def train_epoch(model, trainloader, criterion, device, optimizer):
    """
    This function trains an epoch
    """
    model.train()
    losses = []
    for batch_idx, data in enumerate(trainloader):
        if (batch_idx %100==0) and (batch_idx>=100):
          print('at batchidx',batch_idx)
        #get the image of the data
        #knowing that image is defined this way in the sample for the RainforestDataset
        #sample = {'image': image, 'label': self.split_labels, 'filename': self.image_paths[idx]}
        images = data["image"].to(device)
        prediction = model(images).to(device)
        label = data["label"].to(device)
        my_loss = criterion(prediction, label)
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
        losses.append(my_loss.item())

    return np.mean(losses)


def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):

    model.train()
    curcount = 0
    accuracy = 0

    concat_pred = np.empty((0, numcl)) #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels = np.empty((0, numcl)) #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs=np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader
    bs = dataloader.batch_size

    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):
          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ', batch_idx)
          inputs = data['image'].to(device)
          outputs = model(inputs)
          #print(np.shape(outputs))
          labels = data['label']
          loss = criterion(outputs, labels.to(device))
          losses.append(loss.item())

          #https://sparrow.dev/pytorch-sigmoid/
          #in order to have a sigmoid of the outputs
          preds = outputs.to('cpu')
          #print(np.shape(preds))

          #creating the interval of work
          pluspetit_index = batch_idx * bs
          plusgrand_index = np.minimum((batch_idx + 1) * bs, len(dataloader.dataset))

          concat_pred = np.concatenate((concat_pred,preds),axis=0)
          concat_labels = np.concatenate((concat_labels,labels),axis=0)
          fnames.extend(data["filename"]) # Add the fileneames
    #print(np.shape(concat_pred))
    for c in range(numcl):
          avgprecs[c] = average_precision_score(concat_labels[:, c], concat_pred[:, c])
          avgprecs[c] = np.nan_to_num(avgprecs[c])
    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames

def Tailacc(predictions, labels, t):
    """
    Create an average over all 17 classes for 10 to 20  @values of t.
    It'll show how the accuracy changes as the threshold changes
    #another method
    for c in range(numcl):
        pred = predictions[:,c]
        lab = labels[:,c]
        pred_thresholded = pred[pred>t]
        labels = labels[pred >t]
    return np.sum(pred_thresholded == labels)
    """
    #stores the indice of prediction
    ind = np.where(predictions > t)[0]
    if len(ind) == 0:
        return(labels[np.argmax(predictions)])
    tail_labels = labels[ind]
    return np.mean(tail_labels)

def traineval2_model_nocv(dataloader_train, dataloader_test ,  model ,  criterion, optimizer, scheduler, num_epochs, device, numcl):

  best_measure = 0
  best_epoch =-1

  trainlosses=[]
  testlosses=[]
  testperfs=[]
  mean_avg_precs = []

  for epoch in range(num_epochs):

    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)


    avgloss=train_epoch(model,  dataloader_train,  criterion,  device , optimizer )
    trainlosses.append(avgloss)

    if scheduler is not None:
      scheduler.step()

    perfmeasure, testloss,concat_labels, concat_pred, fnames  = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
    testlosses.append(testloss)
    testperfs.append(perfmeasure)

    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)

    avgperfmeasure = np.mean(perfmeasure)
    mean_avg_precs.append(avgperfmeasure)

    print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure)


    if avgperfmeasure > best_measure:
      bestweights= model.state_dict()
      best_measure = avgperfmeasure
      best_epoch = epoch
      best_preds = concat_pred
      best_labels = concat_labels
      best_names = fnames
  #np.save('best_preds',best_preds)
  #np.save('best_labels',best_labels)
  n_values = 10
  num_classes = 17
  t_max = np.max(best_preds)
  t_values = np.linspace(0.5, t_max, n_values)
  tail_list = np.zeros((n_values, num_classes))
  for i in range(n_values):
      for c in range(num_classes):
          tail_list[i, c] = Tailacc(best_preds[:, c], best_labels[:, c], t_values[i])
  mean_tail_acc = np.mean(tail_list, axis=1)
  """
  #plot tailacc
  plt.plot(t_values, mean_tail_acc)
  plt.xlabel("t")
  plt.ylabel("accuracy")
  plt.title("Tailaccuracies")
  plt.savefig("Tailacc.pdf")
  plt.close()

  #plot the train and test:
  plt.plot(range(num_epochs), trainlosses)
  plt.plot(range(num_epochs), testlosses)
  plt.legend(['train_loss', 'test_loss'])
  plt.title('Loss per epoch ')
  plt.savefig('loss_curve.pdf')
  plt.close()

  #plot the Mean Average precision:
  plt.plot(range(num_epochs), mean_avg_precs)
  plt.title("Mean Average precision over epochs")
  plt.xlabel("Epochs")
  plt.ylabel("Mean Average precision")
  plt.savefig("Mean_Average_precision.pdf")
  plt.close()
  """
  # Find the top 10 and bottom 10 images for one class
  precs = np.argmax(testperfs[best_epoch]) # Find a class with good/max precision
  ind1 = np.argsort(best_preds[:, precs])[-10:] # Ten highest value indicies
  ten_best = np.array(fnames)[ind1]
  ind2 = np.argsort(-best_preds[:, precs])[-10:] # Ten worst
  ten_worst = np.array(fnames)[ind2]
  #print(f"Top ten images for class {precs}: {ten_best}")
  #print(f"Bottom ten images for class {precs}: {ten_worst}")

  # Save the model
  #https://pytorch.org/tutorials/beginner/saving_loading_models.html
  torch.save(bestweights, 'SingleModel.pt')

  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, best_preds, best_labels


class yourloss(nn.modules.loss._Loss):

    def __init__(self, reduction: str = 'mean') -> None:
        super(yourloss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        pass
    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        #https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        BCE_loss = self.loss(input_.float(), target.float())
        return BCE_loss

def runstuff():
    config = dict()
    config['use_gpu'] = True #True #TODO change this to True for training on the cluster
    config['lr'] = 0.005
    config['batchsize_train'] = 32
    config['batchsize_val'] = 64
    config['maxnumepochs'] = 35
    config['scheduler_stepsize'] = 10
    config['scheduler_factor'] = 0.3

    # This is a dataset property.
    config['numcl'] = 17
      # Data augmentations.
    data_transforms = {
          'train': transforms.Compose([
              transforms.Resize(256),
              transforms.RandomCrop(224),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
          ]),
          'val': transforms.Compose([
              transforms.Resize(224),
              transforms.CenterCrop(224),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284])
          ]),
      }

    # Datasets
    image_datasets={}
    root = '/itf-fi-ml/shared/IN5400/2022_mandatory1/'
    image_datasets['train']= RainforestDataset(root,0,data_transforms['train'])
    image_datasets['val']= RainforestDataset(root,1,data_transforms['val'])
    # Dataloaders
    #Exercises week 3 part 1: Implement a dense neural network
    dataloaders = {}
    dataloaders['train'] =  torch.utils.data.DataLoader(image_datasets['train'], batch_size=config['batchsize_train'], shuffle=True, num_workers=1)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=config['batchsize_val'], shuffle=False, num_workers=1)
    #print(image_datasets['val'])

    Data = {}
    # Device
    if True == config['use_gpu']:
        device= torch.device('cuda:0')
    else:
        device= torch.device('cpu')

    # Model
    model_SingleNetwork = models.resnet18()
    model = SingleNetwork(model_SingleNetwork, "kaiminghe")# TwoNetworks()
    model = model.to(device)
    lossfct = yourloss()
    # how we do with the TwoNestworks :
    #pretrained_net1 = models.resnet18(pretrained=True)
    #pretrained_net2 = models.resnet18(pretrained=True)
    #model = TwoNetworks(pretrained_net1=pretrained_net1, pretrained_net2=pretrained_net2)
    #model = model.to(config['device'])
    #someoptimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #somelr_scheduler = torch.optim.lr_scheduler.StepLR(someoptimizer, step_size=1, gamma=0.1, last_epoch=- 1, verbose=False)

    # Observe that all parameters are being optimized
    someoptimizer =  torch.optim.SGD(model.parameters(), lr = config['lr'],momentum=0.9)

    # Decay LR by a factor of 0.3 every X epochs
    #https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html
    somelr_scheduler = torch.optim.lr_scheduler.ExponentialLR(someoptimizer, gamma=config["scheduler_factor"])
    #print('OOO')
    best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, best_preds, best_labels = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'] )
    #print('best_preds',best_preds, 'best_labels', best_labels)
    #print('avgperfmeasure', best_measure)

if __name__=='__main__':
    runstuff()
    #print('The scikit-learn version is {}.'.format(sklearn.__version__))
