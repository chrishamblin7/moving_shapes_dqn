#test
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
#import pdb
import scipy.misc
sys.path.insert(0,'../utility_scripts/')
from game_functions import *
import random_polygon
from math import pi, cos, sin
from copy import deepcopy
#from importlib import reload
import pickle


# Colors
global BLACK, WHITE, RED, RED_ACTIVE, GREEN, BLUE
BLACK = (  0,   0,   0)	
WHITE = (255, 255, 255)
RED = (255,   100,   100)
RED_ACTIVE = (255, 0, 0)
GREEN = (  0, 255,   0)
BLUE = (  100,   100, 255)
BLUE_ACTIVE = (  0,   0, 255)

start_time = time.time() 

start = time.time()    #Get Start time for logging timings


########   SETTINGS   ###########

# Command Line Arguments
parser = argparse.ArgumentParser(description='PyTorch dqn shape moving game')
parser.add_argument('--num-memories', type = int, default=10000,metavar='NM',
					help= 'number of memories to generate (default=10000)')
parser.add_argument('--load-path', type=str,default='NA',metavar='P',
					help= 'path to append memories to, if "NA" generate new file')
parser.add_argument('--world-transforms', action='store_true', default=False,
					help='include world transforms (camera zoom and rotation) in \
					available actions')
parser.add_argument('--reward-function', type=str, default='all', metavar='R',
					help='reward function to use. choices: pix_dif (pixel difference) \
					object_param (closeness to actual parameters of shapes, rot, trans, scale (default:all)')
parser.add_argument('--win-dim', type=int, default=84, metavar='WD',
					help='window dimension, input int, win = int x int (default: 84)')
parser.add_argument('--shape-size', type=int, default=8, metavar='SS',
					help='Average size of intial game shapes (default: 8)')
parser.add_argument('--num-shapes', type=int, default=2, metavar='NS',
					help='Number of shapes in game (default: 2)')
parser.add_argument('--trans-step', type=int, default=2, metavar='TS',
					help='Number of pixels jumped when translating shape (default: 2)')
parser.add_argument('--zoom-ratio', type=float, default=1.2, metavar='ZR',
					help='Scaling ratio when zooming (default: 1.2)')
parser.add_argument('--num-rotations', type=int, default=32, metavar='NR',
					help='Number of discrete rotation positions (default: 32)')    
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='disables CUDA training')
parser.add_argument('--show-window', action='store_true', default=False,
					help='show game window while running')
parser.add_argument('--seed', type=int, default=2, metavar='S',
					help='random seed (default: 2)')


args = parser.parse_args()
print('running with args:')
print(args)

if args.load_path == "NA":
	memories = {'metadata':{0:args},'memories':{}}
	index = 0
else:
	memories = pickle.load(open(args.load_path,'rb'))
	index = len(memories['memories'])
	memories['metadata'][len(memories['metadata'])] = args


#non command-line arguments
if not args.world_transforms:      #set output dimensions of network
	n_actions = 9
else:
	n_actions = 15
net_input_dim = (4,args.win_dim,args.win_dim)    #set input dimensions


#Window Settings
os.environ["SDL_VIDEO_CENTERED"] = "1"
#for running on cluster with no graphics
if not args.show_window:
	os.putenv('SDL_VIDEODRIVER', 'fbcon')
	os.environ["SDL_VIDEODRIVER"] = "dummy"

#initialize game
pygame.init()
screen = pygame.display.set_mode((args.win_dim,args.win_dim))
pygame.display.set_caption("Generating game states")
clock = pygame.time.Clock()

#######    RUN GAME     #######

#stores tuples of in memory of (State, Action, Rewards, Second State, ActionList, action)

for i in range(index, index+args.num_memories+1):
	print('\n\nmemory %s'%i)

	#initialize state
	#pdb.set_trace()
	#print('reinitializing game . . .')
	active_shape = 0
	phase, shapes = update_parameters(screen,args)
	draw_screen(shapes,active_shape,screen)
	pygame.display.update()

	#Generate Target image
	#print('Generating random transform . . .')
	currentscreen3d = get_screen(screen, grey_scale=False)  #get state as 3 channel image
	currentscreen = get_screen(screen)
	stored_shapes = deepcopy(shapes)              #store state   of shape objects
	stored_active_shape = deepcopy(active_shape)
	good_transform = False
	while not good_transform:  
		action_list = random_transformation(shapes,args)            # randomly transform state 
		draw_screen(shapes,active_shape,screen)   
		pygame.display.update()
		targetscreen = get_screen(screen, grey_scale=True)   #store target as grey scale
		if get_pix_ratio(targetscreen):      # check to make sure our target is not all white or all black
			good_transform = True  
	state = np.concatenate((currentscreen3d,np.array([targetscreen])))
	target_shapes = deepcopy(shapes)
	shapes = deepcopy(stored_shapes)
	active_shape = deepcopy(stored_active_shape)
	draw_screen(shapes,active_shape,screen)
	pygame.display.update()
	action_num = np.random.randint(0,n_actions)
	action = get_key(action_num,action_dict)

	memory = {'state one':state,'shapes one':shapes, 'target shapes':target_shapes,'action list':action_list,'action':action}

	currentscreen3d, currentscreen, state, active_shape = makeMove(action, shapes, active_shape, screen, targetscreen,args)
	R_combine,R_object_param,R_pix_diff = getReward(currentscreen,targetscreen, shapes, target_shapes, args, reward_type = args.reward_function)

	memory['shapes two'] = shapes
	memory['state two'] = state
	memory['rewards'] = {'pix_diff':R_pix_diff,'obj_param':R_object_param,'combo':R_combine}
	memories['memories'][i] = memory

file_name = '../memories/%sshapes_%swindim_%s.pkl'%(args.num_shapes,args.win_dim,time.strftime('%m-%d-%Y:%H_%M'))
pickle.dump(memories,open(file_name,'wb'))

print('Total Run Time:')
print("--- %s seconds ---" % (time.time() - start_time))






