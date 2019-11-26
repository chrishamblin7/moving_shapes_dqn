import random
from pygame.locals import *
import os
import random
import pygame
import math
import sys
import numpy as np
import time
import scipy.misc
import random_polygon
from math import pi, cos, sin
from copy import deepcopy



# Colors
global BLACK, WHITE, RED, RED_ACTIVE, GREEN, BLUE
BLACK = (  0,   0,   0)	
WHITE = (255, 255, 255)
RED = (255,   100,   100)
RED_ACTIVE = (255, 0, 0)
GREEN = (  0, 255,   0)
BLUE = (  100,   100, 255)
BLUE_ACTIVE = (  0,   0, 255)


action_dict = {'left':0,'right':1,'down':2,'up':3,'rot_right':4,'rot_left':5,'small':6,'big':7} # number code for different actions


def update_parameters(screen, args):
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

def get_state_image(state,args,name='none'):
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
		imarray = imarray.reshape(args.win_dim,args.win_dim)
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
		if direction == 'smaller':
			direction = -1
		else:
			direction = 1
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
			if not self.centroid[0] < self.stride:
				self.translate((-1*self.stride, 0))
		if key[pygame.K_RIGHT]:
			if not self.centroid[0] > self.win_dim-self.stride: 
				self.translate((self.stride, 0))
		if key[pygame.K_UP]:
			if not self.centroid[1] < self.stride:
				self.translate((0, -1*self.stride))
		if key[pygame.K_DOWN]:
			if not self.centroid[1] > self.win_dim-self.stride:
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
		if input == 'left':
			if not self.centroid[0] < self.stride:
				self.translate((-1*self.stride, 0))
		if input == 'right':
			if not self.centroid[0] > self.win_dim-self.stride: 
				self.translate((self.stride, 0))
		if input == 'up':
			if not self.centroid[1] < self.stride:
				self.translate((0, -1*self.stride))
		if input == 'down':
			if not self.centroid[1] > self.win_dim-self.stride:
				self.translate((0, self.stride))
		if input == 'rot_right':
			self.rotate('right')
		if input == 'rot_left':
			self.rotate('left')
		if input == 'smaller':
			self.scale(-1)
		if input == 'bigger':
			self.scale(1)			

	def draw(self, surface, color = WHITE, width = 0):
		draw_points_list = []    #points are not the same as draw points must be shift by win_dim/10 in each dimension as screen has boundary area
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
def zoom_screen(direction, shapes, args):
	if direction == 'zoom_in':
		direction = 1
	else:
		direction = -1
	center = (int(args.win_dim/2),int(args.win_dim/2))
	for shape in shapes:
		for i in range(len(shape.rotations)):
			new_points_list = []
			for point in shape.rotations[i]:
				new_points_list.append(((args.zoom_ratio**direction)*(point[0]-center[0])+center[0], (args.zoom_ratio**direction)*(point[1]-center[1])+center[1]))
			shape.rotations[i] = new_points_list
			shape.points_list = shape.rotations[shape.rotation]
			shape.centroid = shape.get_centroid(shape.points_list)

def translate_screen(direction, shapes, args):
	for shape in shapes:
		if direction == 'cam_right':
			shape.translate((-1*shape.stride, 0))
		elif direction == 'cam_left': 
			shape.translate((shape.stride, 0))
		if direction == 'cam_down':
			shape.translate((0, -1*shape.stride))
		if direction == 'cam_up':
			shape.translate((0, shape.stride))

def random_transformation(shapes,args):
	active_shape = 0
	action_list = []
	
	for i in range(len(shapes)):
		rots = random.randint(0,shapes[active_shape].num_rotations/2+1)
		rot_dir = random.choice(['rot_right','rot_left'])
		for j in range(rots):
			action_list.append(rot_dir)
		x_dir = random.choice(['left','right'])
		y_dir = random.choice(['down','up'])
		s_dir = random.choice(['smaller','bigger'])
		if x_dir == 'left':
			x_amount = random.randint(0,int((shapes[i].centroid[0])/args.trans_step))
		else:
			x_amount = random.randint(0,int((args.win_dim-shapes[i].centroid[0])/args.trans_step))
		if y_dir == 'up':
			y_amount = random.randint(0,int((shapes[i].centroid[1])/args.trans_step))
		else:
			y_amount = random.randint(0,int((args.win_dim-shapes[i].centroid[1])/args.trans_step))
		s_amount = random.randint(0,5)
		for i in range(y_amount):
			action_list.append(y_dir)
		for i in range(x_amount):
			action_list.append(x_dir)
		for i in range(s_amount):
			action_list.append(s_dir)
		action_list.append('switch_shape')
	if args.world_transforms:
		cy_dir = random.choice(['cam_up','cam_down'])
		cx_dir = random.choice(['cam_left','cam_right'])
		cz_dir = random.choice(['zoom_in','zoom_out'])
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
	for action in action_list:
		if action == 'switch_shape':
			active_shape = (active_shape+1)%len(shapes)
		elif not('cam' in action or 'zoom' in action):
			shapes[active_shape].robo_action(action)
		#Translate
		elif action == 'cam_up':
			translate_screen('cam_up', shapes,args)
		elif action == 'cam_down':
			translate_screen('cam_down', shapes,args)
		elif action == 'cam_left':
			translate_screen('cam_left', shapes,args)
		elif action == 'cam_right':
			translate_screen('cam_right', shapes,args)
		#zoom
		elif action == 'zoom_in':
			zoom_screen('zoom_in', shapes, args)
		elif action == 'zoom_out':
			zoom_screen('zoom_out', shapes, args)
	print('generated random transform:')
	print(action_list)
	return action_list

def makeMove(action, shapes, active_shape, screen, targetscreen, args):
	print('action: %s'%action)
	screen.fill((0, 0, 0))
	#Draw Shapes
	if action == 'switch_shape':
		active_shape = (active_shape+1)%len(shapes)
	elif not('cam' in action or 'zoom' in action):
		shapes[active_shape].robo_action(action)
	#Translate
	elif i == 'cam_up':
		translate_screen('cam_up', shapes,args)
	elif i == 'cam_down':
		translate_screen('cam_down', shapes,args)
	elif i == 'cam_left':
		translate_screen('cam_left', shapes,args)
	elif i == 'cam_right':
		translate_screen('cam_right', shapes,args)
	#zoom
	elif i == 'zoom_in':
		zoom_screen('zoom_in', shapes, args)
	elif i == 'zoom_out':
		zoom_screen('zoom_out', shapes, args)

	draw_screen(shapes,active_shape,screen)
	pygame.display.update()

	currentscreen3d = get_screen(screen,grey_scale=False)
	currentscreen = get_screen(screen)
	state = np.concatenate((currentscreen3d,np.array([targetscreen])))
	return currentscreen3d, currentscreen, state, active_shape


def getReward(currentscreen,targetscreen, shapes, target_shapes, args, reward_type = 'object_param', reward_scale = 1):
	
	win_dim = args.win_dim
	if reward_type == 'pix_dif':
		contrast = currentscreen - targetscreen
		unique, counts = np.unique(contrast, return_counts=True)
		score = dict(zip(unique,counts))[0]
		reward = score/currentscreen.size
	else:
		object_scaling = {'x':.3,'y':.3,'r':.2,'s':.2}    # relative weight for translation, rotation, and scaling importance
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
