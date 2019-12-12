import random
from pygame.locals import *
import os
import argparse
import random
import pygame
import math
import sys
import numpy as np
import time
import pdb
import scipy.misc
sys.path.insert(0,'../utility_scripts/')
from game_functions import *
from random_polygon import *
from math import pi, cos, sin
from copy import deepcopy
from importlib import reload

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#from tensorboardX import SummaryWriter



# Colors
global BLACK, WHITE, RED, RED_ACTIVE, GREEN, BLUE
BLACK = (  0,   0,   0)	
WHITE = (255, 255, 255)
RED = (255,   100,   100)
RED_ACTIVE = (255, 0, 0)
GREEN = (  0, 255,   0)
BLUE = (  100,   100, 255)
BLUE_ACTIVE = (  0,   0, 255)


def train(args, model, device, train_loader, optimizer, criterion, epoch):
    
    model.train()

    for batch_idx, (data,target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        
        if not args.no_train_log:
            if batch_idx % args.log_interval == 0:
                print_out('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())args.print_log)                

def test(args, model, device, test_loader, criterion, epoch):
    #pdb.set_trace()
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            
            correct += pred.eq(target.view_as(pred)).sum().item()

           
            for i in range(len(pred)):
                indiv_acc_dict[int(pred[i])][2] += 1          
                indiv_acc_dict[int(target.view_as(pred)[i])][0] += 1
                if int(pred[i]) == int(target.view_as(pred)[i]):
                    indiv_acc_dict[int(pred[i])][1] += 1

                    
    test_loss /= len(test_loader.dataset)

    print_out('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)),args.print_log)


def main(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    #load model
    #model, input_size, params_to_update = initialize_model(args.model, args.num_classes, args.feature_extract, use_pretrained=args.pretrained)
    model = models.alexnet()
    model.features[0] = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))

    model.to(device)
    if args.multi_gpu:
        model = nn.DataParallel(model)

    #data loading
    train_loader = torch.utils.data.DataLoader(
        data_classes.movingblockstates(),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        data_classes.data_classes.movingblockstates(train=False),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
  
    #Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(params_to_update, lr=args.lr)
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum)
        #optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum)

    #loss
    criterion = nn.CrossEntropyLoss()


    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        test(args, model, device, test_loader, criterion, epoch)
        if epoch%args.save_model_interval == 0:
        	torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, os.path.join('../models/saved_models',args.outputdir,'model_%s.pt'%str(epoch)))


    

def print_out(text,log):
    print(text)
    log.write(str(text))
    log.flush()



####Data loading memories

default_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

mem_dict = pickle.load(open('../memories/2shapes2_256windim_12-11-2019:07_41.pkl','rb'))


class movingblockstates(Dataset):
	"""How many dots are in this pic dataset."""

	def __init__(self, mem_dict=mem_dict, train=True, num_classes=10, transform = default_transform):
		"""
		"""
		
		self.train=False   
		if train:
			self.indices = range(16000)
			self.train=True
		else:
			self.indices = len(range(16000,20000))

		self.transform=transform
		self.mem_dict = mem_dict
	def __len__(self):
		return len(self.indices)
    
	def __getitem__(self, idx):

		state = self.mem_dict['memories'][indices[idx]]['state one']
		state = self.transform(image)
		label = self.mem_dict['memories'][indices[idx]]['objective_function']
		label = from_numpy(label)
		sample = (state,label)
		return sample



if __name__ == '__main__':

    start_time = time.time()

        #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Supervised Learn to move blocks')
    parser.add_argument('--batch-size', type=int, default=400, metavar='BS',
                        help='input batch size for training (default: 400)')
    parser.add_argument('--test-batch-size', type=int, default=400, metavar='TBS',
                        help='input batch size for testing (default: 400)')
    parser.add_argument('--epochs', type=int, default=200, metavar='E',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--model', type=str, default='resnet', metavar='M',
                        help='neural net model to use (default: resnet, other options: alexnet, vgg, squeezenet, densenet)')
    parser.add_argument('--num_classes', type=int, default=16, metavar='C',
                        help='max number of label, all higher numbers will be binned to num_classes (default: 20, all correct labels)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='initialize model at trained ImageNet weights')
    parser.add_argument('--feature-extract', action='store_true', default=False,
                        help='do not train the whole network just the last classification layer. Will automatically set "pretrained" to true.')        
    parser.add_argument('--optimizer', type=str, default='adam', metavar='O',
                        help='optimization algorithm to use (default: adam other options: SGD)')   
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.2, metavar='m',
                        help='SGD momentum (default: 0.2)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--input-data', type=str, default='enumeration', metavar='I',
                        help='input folder name to use for input data (default: enumeration)')
    parser.add_argument('--save-model-interval', type=int, default=5, metavar='SM',
                        help='Every time the epoch number is divisible by this number, save the model (default: 5)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='W',
                        help='number of parallel batches to process. Rule of thumb 4*num_gpus (default: 4)')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='run model on multiple gpus')    
 

    args = parser.parse_args()
    # reconcile arguments
    if args.feature_extract:
        args.pretrained == True

    #output directory
    args.outputdir = '2shapes_alexnet_%s'%args.num_classes,time.strftime('%m-%d-%Y:%H_%M')
 

    if not os.path.exists(os.path.join('../models/saved_models',args.outputdir)):
        os.mkdir(os.path.join('../models/saved_models',args.outputdir))
    args.testing_log = open(os.path.join('../models/saved_models',args.outputdir,'testing_log.csv'),'w+')
    args_file = open(os.path.join('../models/saved_models',args.outputdir,'args.txt'),'w+')
    args_file.write(str(args))
    args_file.close()

    args.print_log = open(os.path.join('../models/saved_models',args.outputdir,'print_log.txt'),'w+') #file logs everything that prints to console

    print_out('running with args:',args.print_log)
    print_out(args,args.print_log)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print_out('using cuda',args.print_log)

    main(args)

    print_out('Total Run Time:',args.print_log)
    print_out("--- %s seconds ---" % (time.time() - start_time),args.print_log)




