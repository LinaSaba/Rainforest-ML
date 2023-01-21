import torch
import torch.nn as nn
from RainforestDataset1 import *
from torchvision import  models

class TwoNetworks(nn.Module):
    '''
    This class takes two pretrained networks,
    concatenates the high-level features before feeding these into
    a linear layer.

    functions: forward
    '''
    def __init__(self, pretrained_net1, pretrained_net2):
        super(TwoNetworks, self).__init__()

        _, num_classes = get_classes_list()

        # explanation of the question : drop the FC layer in each of these two,
        #and create a new one that concatenates the outputs of these two networks,
        #so instead of having one FC layer for each network, you would drop these two
        #and have on FC that gets values from both networks.
        #defining first model of resnet
        self.pretrained_net1 = nn.Sequential(*list(pretrained_net1.modules())[:-1])
        self.pretrained_net2 = nn.Sequential(*list(pretrained_net2.modules())[:-1])
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, inputs1, inputs2):
        # concatenate the features before the linear layer
        inputs1 = inputs1[:, 0:3, :, :]
        inputs2 = inputs2[:, 3:4, :, :]
        outputs1 = self.pretrained_net1(input1)
        outputs2 = self.pretrained_net2(input2)
        output = self.linear(torch.cat((outputs1 , outputs2, self.fc), dim=1))
        return output

class SingleNetwork(nn.Module):
    '''
    This class takes one pretrained network,
    the first conv layer can be modified to take an extra channel.

    functions: forward
    '''

    def __init__(self, pretrained_net, weight_init):
        super(SingleNetwork, self).__init__()
        _, num_classes = get_classes_list()
        if weight_init is not None:
            current_weights = pretrained_net.conv1.weight
            #print("2")
            #print(current_weights.size())
            #the result is torch.Size([64, 3, 7, 7])
            #print("3")
            new_weights = torch.zeros(64, 1, 7, 7)

            if weight_init == "kaiminghe":
                nn.init.kaiming_normal_(new_weights)
            weights = torch.cat((current_weights, new_weights), 1)
            pretrained_net.conv1.weight = nn.Parameter(weights)
        #https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        pretrained_net.fc = torch.nn.Linear(512, 17)
        self.net = pretrained_net

    def forward(self, inputs):
        return self.net(inputs)


if  __name__=='__main__':
    model = models.resnet18()
    #print(model)
    print("1")
    print(SingleNetwork(model,1))
    print("4")
