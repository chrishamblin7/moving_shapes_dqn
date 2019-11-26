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



sys.path.insert(0,'../utility_scripts/')



np.set_printoptions(threshold=np.inf)

def main():
	start = time.time()    #Get Start time for logging timings


	########   SETTINGS   ###########

	# Command Line Arguments
	parser = argparse.ArgumentParser(description='PyTorch dqn shape moving game')
	parser.add_argument('num_memories', type = int, default=10000,metavar='NM',
						help= 'number of memories to generate (default=10000)')
	parser.add_argument('load_path', type=str,default='NA',metavar='P',
						help= 'path to append memories to, if "NA" generate new file')
	parser.add_argument('--world-transforms', type=bool, default=False, metavar='WT',
						help='include world transforms (camera zoom and rotation) in \
						available actions (default: False)')
	parser.add_argument('--reward-function', type=str, default='object_param', metavar='R',
						help='reward function to use. choices: pix_dif (pixel difference) \
						object_param (closeness to actual parameters of shapes, rot, trans, scale (default:object_param)')
	parser.add_argument('--win-dim', type=int, default=84, metavar='WD',
						help='window dimension, input int, win = int x int (default: 84)')
	parser.add_argument('--shape-size', type=int, default=8, metavar='SS',
						help='Average size of intial game shapes (default: 8)')
	parser.add_argument('--num-shapes', type=int, default=3, metavar='NS',
						help='Number of shapes in game (default: 4)')
	parser.add_argument('--trans-step', type=int, default=2, metavar='TS',
						help='Number of pixels jumped when translating shape (default: 4)')
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

	if args.load_path = "NA":
		memories = {'metadata':{0:args},'memories':{}}
		index = 0
	else:
		memories = pickle.load(args.load_path)
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
	# Colors
	global BLACK, WHITE, RED, RED_ACTIVE, GREEN, BLUE

	BLACK = (  0,   0,   0)
	WHITE = (255, 255, 255)
	RED = (255,   100,   100)
	RED_ACTIVE = (255, 0, 0)
	GREEN = (  0, 255,   0)
	BLUE = (  0,   0, 255)
	BLUE_ACTIVE = (  0,   0, 255)


	#initialize game
	pygame.init()
	screen = pygame.display.set_mode((args.win_dim,args.win_dim))
	pygame.display.set_caption("Generating game states")
	clock = pygame.time.Clock()

	#######    RUN GAME     #######

	#stores tuples of (S, A, R, S')
	replay = []
	h = 0

	for i in range(args.epochs):
		print('\n\nGame # %s'%i)

		#initialize state
		#pdb.set_trace()
		#print('reinitializing game . . .')
		active_shape = 0
		phase, shapes = update_parameters(args,screen)
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
			random_transformation(shapes,args.zoom_ratio,args.win_dim)            # randomly transform state 
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

		#print('transform found')



		#get_state_image(state)
		status = 1
		#while game still in progress
		iters = -1
		while(status == 1):
			iters += 1


			action = np.random.randint(0,n_actions)
				action_type = 'Random'
			else: #choose best action from Q(s,a) values
				action = (np.argmax(qval))
				action_type = 'Q Policy'
			#Take action, observe new state S'
			currentscreen3d, currentscreen, new_state, active_shape = makeMove(action, shapes, active_shape, screen, targetscreen,args.zoom_ratio,args.win_dim)
			#Observe reward
			reward = getReward(currentscreen,targetscreen, shapes, target_shapes, args, reward_type = args.reward_function)
			if iters%args.log_interval == 0:
				print('\nmove: %s    action: %s   (%s)     Reward: %s\n'%(iters,action,action_type,reward))
				print('Qval from model: %s'%qval)

			#Experience replay storage
			if (len(replay) < args.buffer): #if buffer not filled, add to it
				replay.append((state, action, reward, new_state))
			else: #if buffer full, overwrite old values
				if (h < (args.buffer-1)):
					h += 1
				else:
					h = 0
				replay[h] = (state, action, reward, new_state)
				#randomly sample our experience replay memory
				minibatch = random.sample(replay, args.batch_size)
				X_train = []
				y_train = []
				for memory in minibatch:
					#Get max_Q(S',a)
					old_state_memory, action_memory, reward_memory, new_state_memory = memory
					old_qval = predict(old_state_memory, model, device, args)
					newQ = predict(new_state_memory, model, device, args)
					maxQ = np.max(newQ)
					y = np.zeros((1,n_actions))
					y[:] = old_qval[:]
					#if reward_memory != 10: #non-terminal state
					update = (reward_memory + (args.gamma * maxQ))
					#else: #terminal state
					#	update = reward_memory
					y[0][action_memory] = update
					X_train.append(old_state_memory.reshape(net_input_dim))
					y_train.append(y.reshape(n_actions,))

				X_train = np.array(X_train)
				y_train = np.array(y_train)
				update_text = train(X_train, y_train, model, optimizer, loss_func, device, args)
				if iters%args.log_interval == 0:
					print(update_text)

			#pdb.set_trace()
			state = new_state
			#debugging
			#if not (len(replay) < buffer):
			#    pdb.set_trace()
			if iters > args.max_moves: #if reached terminal state, or too many moves taken update game status
				status = 0
			#clear_output(wait=True)
		if epsilon > 0.1: #decrement epsilon over time
			epsilon -= (1/args.epochs)
		#save model
		if i%args.save_interval == 0:
			end = time.time()
			elapse_time = end -start
			print('TIME to epoch %s: %s'%(i,elapse_time))
			print('saving model')
			torch.save(model,os.path.join('../models','saved_models','model_%s.pt'%str(i)))

	torch.save(model,os.path.join('models','saved_models','model_%s.pt'%str(i)))



