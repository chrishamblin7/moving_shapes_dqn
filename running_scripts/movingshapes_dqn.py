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
import random_polygon
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
BLUE = (  0,   0, 255)
BLUE_ACTIVE = (  0,   0, 255)



sys.path.insert(0,'../utility_scripts/')
from pytorch_utils import to_torch_net_input

sys.path.insert(0,'../models/scripts/')
import dqn_basic
#import dist_dqn
#import rainbow_dqn

np.set_printoptions(threshold=np.inf)

def main():
	start = time.time()    #Get Start time for logging timings


	########   SETTINGS   ###########

	# Command Line Arguments
	parser = argparse.ArgumentParser(description='PyTorch dqn shape moving game')
	parser.add_argument('--batch-size', type=int, default=100, metavar='N',
						help='input batch size for training (default: 100)')
	parser.add_argument('--world-transforms', type=bool, default=False, metavar='WT',
						help='include world transforms (camera zoom and rotation) in \
						available actions (default: False)')
	parser.add_argument('--epochs', type=int, default=4501, metavar='E',
						help='number of epochs to train (default: 4000)')
	parser.add_argument('--reward-function', type=str, default='object_param', metavar='R',
						help='reward function to use. choices: pix_dif (pixel difference) \
						object_param (closeness to actual parameters of shapes, rot, trans, scale \                         (default:object_param)')
	parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
						help='learning rate (default: 0.0001)')
	parser.add_argument('--gamma', type=float, default=0.975, metavar='G',
						help='discount factor for future reward (default: 0.975)')
	parser.add_argument('--buffer', type=int, default=20000, metavar='B',
						help='Number of states to store in exp_replay (default: 20000)')
	parser.add_argument('--max-moves', type=int, default=40, metavar='MM',
						help='Max moves before reinitializing (default: 40)')    
	parser.add_argument('--win-dim', type=int, default=84, metavar='WD',
						help='window dimension, input int, win = int x int (default: 84)')
	parser.add_argument('--shape-size', type=int, default=8, metavar='SS',
						help='Average size of intial game shapes (default: 8)')
	parser.add_argument('--num-shapes', type=int, default=4, metavar='NS',
						help='Number of shapes in game (default: 4)')
	parser.add_argument('--trans-step', type=int, default=4, metavar='TS',
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
	parser.add_argument('--model', type=str, default='dqn_basic', metavar='M',
						help='neural net model to use (default: dqn_basic, other options: dist_dqn, rainbow_dqn)')
	parser.add_argument('--loss', type=str, default='mse', metavar='L',
						help='loss function to use (default: mse, other options: cross_entropy)')
	parser.add_argument('--save-interval', type=int, default=500, metavar='SI',
						help='Save model every (save-interal) epochs (default: 500)')
	parser.add_argument('--log-interval', type=int, default=20, metavar='LI',
						help='log results every (log-interal) moves (default: 20)')
	parser.add_argument('--num-gpus', type=int, default=2, metavar='NG',
						help='Number of gpus to train in parallel (default: 2)')

	args = parser.parse_args()
	print('running with args:')
	print(args)


	#non command-line arguments
	if not args.world_transforms:      #set output dimensions of network
		n_actions = 9
	else:
		n_actions = 15
	net_input_dim = (4,args.win_dim,args.win_dim)    #set input dimensions
	epsilon = 1

	#Hardware setting (GPU vs CPU)

	use_cuda = not args.no_cuda and torch.cuda.is_available()
	if use_cuda:
		print('using cuda')
	#seed
	torch.manual_seed(args.seed)
	
	device = torch.device("cuda" if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	#initialize model
	model_dict = {'dqn_basic':dqn_basic.DQN}
	model = model_dict[args.model](net_input_dim,n_actions).to(device)
	#handle multiple gpus
	if args.num_gpus > 1:
		print("Running on", torch.cuda.device_count(), "gpus")
		args.batch_size = torch.cuda.device_count()*args.batch_size
		model = nn.DataParallel(model)
	
	loss_dict = {'mse':nn.MSELoss(),
				 'cross_entropy':nn.CrossEntropyLoss()}
	loss_func = loss_dict[args.loss]

	optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
	pygame.display.set_caption("Move the Shapes to Match the Target Image")
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

			#We are in state S
			#Let's run our Q function on S to get Q values for all possible actions
			qval = predict(state, model, device, args)

			if (random.random() < epsilon):
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





#######   FUNCTIONS    ########

def update_parameters(args, screen):
	spaces = int(args.win_dim/args.trans_step)
	phase = int(np.random.choice(list(range(args.trans_step))))
	shapes = {}
	shape_positions = list(range(int(args.shape_size/2+phase),int(args.win_dim-args.shape_size/2+phase),args.trans_step))
	shapes = []
	for s in range(args.num_shapes):
		shape = Polygon(screen,args.trans_step,args.num_rotations,args.win_dim,args.zoom_ratio)
		while shape.area < (args.win_dim/10)**2:
			shape = Polygon(screen,args.trans_step,args.num_rotations,args.win_dim,args.zoom_ratio)
		shape_pos = (np.random.choice(shape_positions),np.random.choice(shape_positions))
		shape.translate((shape_pos[0]-shape.centroid[0],shape_pos[1]-shape.centroid[1]))
		shapes.append(deepcopy(shape))
	return phase, shapes

def get_screen(screen,flatten = False, grey_scale = True):
	npscreen = pygame.surfarray.array3d(screen).transpose((2, 0, 1))  # transpose into torch order (CHW)
	npscreen = np.ascontiguousarray(npscreen, dtype=np.float32) / 255
	if grey_scale:
		new_screen = np.zeros((npscreen.shape[1],npscreen.shape[2]))
		for h in range(new_screen.shape[0]):
			for w in range(new_screen.shape[1]):
				for c in range(npscreen.shape[0]):
					if npscreen[c,h,w] != 0:
						new_screen[h,w] = 1
						break
		npscreen = new_screen
	if flatten:
		npscreen = npscreen.flatten()
	return npscreen

def get_pix_ratio(npscreen):
	unique_ls = np.unique(npscreen)
	if len(unique_ls) < 2:
		return False
	else:
		return True

def get_state_image(state,win_dim,name='none'):
	'''utility function to save an image of the numpy 'state', to make sure it matches game display'''
	if state.ndim == 1:
		imarray = np.reshape(state,(int(np.sqrt(len(state))),int(np.sqrt(len(state)))))
		imarray = np.array([imarray,imarray,imarray])
		imarray = imarray.transpose(2,1,0)
	elif state.ndim == 2:
		imarray = np.array([state,state,state])
		imarray = imarray.transpose(2,1,0)
	else:
		imarray = state.transpose(2,1,0)
		imarray = imarray.reshape(win_dim,win_dim)
	imarray[imarray > 0] = 255
	imarray[imarray != 255] = 0
	if name == 'none':
		scipy.misc.imsave('images/state_%s.png'%time.time(),imarray)
	else:
		scipy.misc.imsave('images/%s'%name,imarray)


class Polygon(object):

	def get_segments(self,pl):  #pl is points list
		return zip(pl, pl[1:] + [pl[0]])

	def get_area(self,pl):  #pl is points_list
		return 0.5 * abs(sum(x0*y1 - x1*y0
							 for ((x0, y0), (x1, y1)) in self.get_segments(pl)))

	def get_rotations(self):
		rotations = [self.points_list]
		for rot in range(1,self.num_rotations):
			rads = rot*2*pi/self.num_rotations
			new_points_list = []
			for point in self.points_list:
				x = point[0]
				y = point[1]
				a = self.centroid[0]
				b = self.centroid[1]
				new_x = (x-a)*cos(rads) - (y-b)*sin(rads)+ a
				new_y = (x-a)*sin(rads) + (y-b)*cos(rads) + b
				new_points_list.append((new_x,new_y))
			rotations.append(new_points_list)
		return rotations

	def get_centroid(self,pl):
		x = [p[0] for p in pl]
		y = [p[1] for p in pl]
		return (sum(x) / len(pl), sum(y) / len(pl))

	def __init__(self,screen,stride,num_rotations,win_dim,zoom_ratio,num_points = 'random',points_list = 'random'):
		self.num_rotations = num_rotations
		self.stride = stride
		self.win_dim = win_dim
		self.zoom = zoom_ratio
		if points_list != 'random':
			self.points_list = points_list
			self.num_points = len(points_list)
		else:
			if num_points != 'random':
				self.num_points = num_points
			else:
				self.num_points = int(np.random.choice([3,4,5,6]))
			self.points_list = random_polygon.gen_polygon(int(win_dim/2-win_dim/5),int(win_dim/2+win_dim/5),self.num_points)

		self.area = self.get_area(self.points_list) 		
		self.centroid = self.get_centroid(self.points_list)
		self.rotations = self.get_rotations()
		self.rotation = 0

	def translate(self,direction):
		for i in range(len(self.rotations)):
			new_points_list = []
			for point in self.rotations[i]:
				new_points_list.append(tuple(map(sum, zip(point, direction))))
			self.rotations[i] = new_points_list
			self.points_list = self.rotations[self.rotation]
			self.centroid = self.get_centroid(self.points_list)

	def rotate(self,direction):
		if direction == 'right':
			self.rotation = (self.rotation+1)%self.num_rotations
		if direction == 'left':
			self.rotation = (self.rotation-1)%self.num_rotations
		self.points_list = self.rotations[self.rotation]

	def scale(self,direction):
		for i in range(len(self.rotations)):
			new_points_list = []
			for point in self.rotations[i]:
				new_points_list.append(((self.zoom**direction)*(point[0]-self.centroid[0])+self.centroid[0], (self.zoom**direction)*(point[1]-self.centroid[1])+self.centroid[1]))
			self.rotations[i] = new_points_list
			self.points_list = self.rotations[self.rotation]
			self.centroid = self.get_centroid(self.points_list)
			self.area = self.get_area(self.points_list)

	def handle_keys(self):
		key = pygame.key.get_pressed()
		dist = 1
		if key[pygame.K_LEFT]:
			#if not self.centroid[0] < self.stride:
			self.translate((-1*self.stride, 0))
		if key[pygame.K_RIGHT]:
			#if not self.centroid[0] > max_dim-self.stride: 
			self.translate((self.stride, 0))
		if key[pygame.K_UP]:
			#if not self.centroid[1] < self.stride:
			self.translate((0, -1*self.stride))
		if key[pygame.K_DOWN]:
			#if not self.centroid[1] > max_dim-self.stride:
			self.translate((0, self.stride))
		if key[pygame.K_q]:
			self.rotate('left')
		if key[pygame.K_w]:
			self.rotate('right')
		if key[pygame.K_a]:
			self.scale(-1)
		if key[pygame.K_s]:
			self.scale(1)


	def robo_action(self,input):
		if input == 0:
			#if not self.centroid[0] < self.stride:
			self.translate((-1*self.stride, 0))
		if input == 1:
			#if not self.centroid[0] > max_dim-self.stride: 
			self.translate((self.stride, 0))
		if input == 2:
			#if not self.centroid[1] < self.stride:
			self.translate((0, -1*self.stride))
		if input == 3:
			#if not self.centroid[1] > max_dim-self.stride:
			self.translate((0, self.stride))
		if input == 4:
			self.rotate('right')
		if input == 5:
			self.rotate('left')
		if input == 6:
			self.scale(-1)
		if input == 7:
			self.scale(1)			

	def draw(self, surface, color = WHITE, width = 0):
		draw_points_list = []    #points are not the same as draw points must be shift by max_dim/10 in each dimension as screen has boundary area
		for point in self.points_list:
			draw_points_list.append((point[0],point[1]))
		pygame.draw.polygon(surface, color, draw_points_list, width)


def draw_screen(shapes,active_shape,screen):
	screen.fill((0, 0, 0))
	#Draw Shapes
	for s in range(len(shapes)):
		if s == active_shape:
			continue 
		shapes[s].draw(screen,color=RED)
		#if automated:
		#	player.robo_action(optimal_action(player,target))
	shapes[active_shape].draw(screen,color=RED_ACTIVE)

#global transformations
def zoom_screen(direction, shapes, zoom_ratio, win_dim):
	center = (int(win_dim/2),int(win_dim/2))
	for shape in shapes:
		for i in range(len(shape.rotations)):
			new_points_list = []
			for point in shape.rotations[i]:
				new_points_list.append(((zoom_ratio**direction)*(point[0]-center[0])+center[0], (zoom_ratio**direction)*(point[1]-center[1])+center[1]))
			shape.rotations[i] = new_points_list
			shape.points_list = shape.rotations[shape.rotation]
			shape.centroid = shape.get_centroid(shape.points_list)

def translate_screen(direction, shapes, win_dim):
	for shape in shapes:
		if direction == 0:
			shape.translate((-1*shape.stride, 0))
		elif direction == 1: 
			shape.translate((shape.stride, 0))
		if direction == 2:
			shape.translate((0, -1*shape.stride))
		if direction == 3:
			shape.translate((0, shape.stride))

def random_transformation(shapes,zoom_ratio,win_dim):
	active_shape = 0
	action_list = []
	for i in range(len(shapes)):
		rots = random.randint(0,shapes[active_shape].num_rotations/2+1)
		rot_dir = random.choice([4,5])
		for i in range(rots):
			action_list.append(rot_dir)
		y_dir = random.choice([2,3])
		x_dir = random.choice([0,1])
		s_dir = random.choice([6,7])
		y_amount = random.randint(0,10)
		x_amount = random.randint(0,10)
		s_amount = random.randint(0,10)
		for i in range(y_amount):
			action_list.append(y_dir)
		for i in range(x_amount):
			action_list.append(x_dir)
		for i in range(s_amount):
			action_list.append(s_dir)
		action_list.append(8)
	cy_dir = random.choice([9,10])
	cx_dir = random.choice([11,12])
	cz_dir = random.choice([13,14])
	cy_amount = random.randint(0,10)
	cx_amount = random.randint(0,10)
	cz_amount = random.randint(0,10)
	for i in range(cy_amount):
		action_list.append(cy_dir)
	for i in range(cx_amount):
		action_list.append(cx_dir)
	for i in range(cz_amount):
		action_list.append(cz_dir)
	#print(action_list)	
	for i in action_list:
		if i == 8:
			active_shape = (active_shape+1)%len(shapes)
		elif i < 8:
			shapes[active_shape].robo_action(i)
		#Translate
		elif i == 9:
			translate_screen(0, shapes,win_dim)
		elif i == 10:
			translate_screen(1, shapes,win_dim)
		elif i == 11:
			translate_screen(2, shapes,win_dim)
		elif i == 12:
			translate_screen(3, shapes,win_dim)
		#zoom
		elif i == 13:
			zoom_screen(-1, shapes, zoom_ratio, win_dim)
		elif i == 14:
			zoom_screen(1, shapes, zoom_ratio, win_dim)


def makeMove(action, shapes, active_shape, screen, targetscreen, zoom_ratio,win_dim):

	screen.fill((0, 0, 0))
	#Draw Shapes
	if action == 8:
		active_shape = (active_shape+1)%len(shapes)
	elif action < 8:
		shapes[active_shape].robo_action(action)
	#Translate
	elif action == 9:
		translate_screen(0, shapes,win_dim)
	elif action == 10:
		translate_screen(1, shapes,win_dim)
	elif action == 11:
		translate_screen(2, shapes,win_dim)
	elif action == 12:
		translate_screen(3, shapes,win_dim)
	#zoom
	elif action == 13:
		zoom_screen(-1, shapes, zoom_ratio, win_dim)
	elif action == 14:
		zoom_screen(1, shapes, zoom_ratio, win_dim)

	draw_screen(shapes,active_shape,screen)
	pygame.display.update()

	currentscreen3d = get_screen(screen,grey_scale=False)
	currentscreen = get_screen(screen)
	state = np.concatenate((currentscreen3d,np.array([targetscreen])))
	return currentscreen3d, currentscreen, state, active_shape

'''
class Reward(object):
	def pointwise(self):
		contrast = self.currentscreen - self.targetscreen
		unique, counts = np.unique(contrast, return_counts=True)
		score = dict(zip(unique,counts))[0]
		return score/self.currentscreen.size		

	def shapewise(self):
		num_shapes = len(shapes)

	def __init__(self, currentscreen,targetscreen, shapes, target_shapes, reward_type, scale = 1):
		self.currentscreen = currentscreen
		self.targetscreen = targetscreen
		self.shapes = shapes
		self.target_shapes = target_shapes
		self.reward_type = reward_type
		self.scale = scale
		self.reward = self.get_reward()
'''


def getReward(currentscreen,targetscreen, shapes, target_shapes, args, reward_type = 'object_param', reward_scale = 1):
	
	win_dim = args.win_dim
	if reward_type == 'pix_dif':
		contrast = currentscreen - targetscreen
		unique, counts = np.unique(contrast, return_counts=True)
		score = dict(zip(unique,counts))[0]
		reward = score/currentscreen.size
	else:
		object_scaling = {'x':.3,'y':.3,'r':.25,'s':.15}    # relative weight for translation, rotation, and scaling importance
		shape_scores = []
		for i in range(len(shapes)):
			shape = shapes[i]
			target_shape = target_shapes[i]
			x_dif = (args.win_dim - abs(shape.centroid[0]-target_shape.centroid[0]))/args.win_dim
			y_dif = (args.win_dim - abs(shape.centroid[1]-target_shape.centroid[1]))/args.win_dim
			rot_dif = (shape.num_rotations-abs(shape.rotation - target_shape.rotation))/shape.num_rotations
			scale_dif = (win_dim**2 - abs(shape.area-target_shape.area))/win_dim**2
			shape_score = x_dif*object_scaling['x']+y_dif*object_scaling['y']+rot_dif*object_scaling['r']+scale_dif*object_scaling['s']
			shape_scores.append(shape_score)
		reward = sum(shape_scores)/len(shape_scores)
	if reward_type == 'combined':
		contrast = currentscreen - targetscreen
		unique, counts = np.unique(contrast, return_counts=True)
		score = dict(zip(unique,counts))[0]
		reward += score/currentscreen.size    #add both score types together
		reward = reward/2
		print('Reward: %s'%reward*reward_scale)
	return reward*reward_scale

def train(data, target, model, optimizer, loss_func, device, args):
	
	model.train()
	net_input = to_torch_net_input(data)
	target = torch.from_numpy(target)
	target = target.type(torch.FloatTensor)
	net_input, target = net_input.to(device), target.to(device)
	optimizer.zero_grad()
	output = model(net_input)
	loss = loss_func(output, target)
	loss.backward()
	optimizer.step()
		
	return 'network Loss: %s'%loss.item()                

def predict(data, model, device, args, numpy_out = True):
	model.eval()
	net_input = to_torch_net_input(data)
	with torch.no_grad():
		net_input = net_input.to(device)
		output = model(net_input)
		if numpy_out:
			if device != 'cpu':
				output = output.to('cpu')
			return output.numpy()
		else:
			return output


if __name__ == '__main__':
	start_time = time.time()
	main()
	print('Total Run Time:')
	print("--- %s seconds ---" % (time.time() - start_time))