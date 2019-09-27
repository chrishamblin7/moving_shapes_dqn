#! /usr/bin/env python
from pygame.locals import *
import os
import random
import pygame
import math
import sys
import numpy as np
sys.path.insert(0,'../utility_scripts/')
import random_polygon
from math import pi, cos, sin
from copy import deepcopy
from subprocess import call
import time
import scipy.misc
np.set_printoptions(threshold=np.inf)

os.environ["SDL_VIDEO_CENTERED"] = "1"

pygame.init()

keypad = False
max_dim = int(500)
num_shapes = 4 #total number of shapes each player has control over
shape_size = int(10)
#target_size = int(player_size+2)
#spaces_list = [int(max_dim/10),int(max_dim/5),int(max_dim/2)]
spaces_list = [int(max_dim/5)]
#rotations_list = [4,8,16]
rotations_list = [64]
zoom_list = [1.2]


BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED = (255,   100,   100)
GREEN = (  0, 255,   0)
BLUE = (  100,   100, 255)
RED_ACTIVE = (255, 0,  0)
BLUE_ACTIVE = (0, 0, 255)
colors = {'p1':{'active':RED_ACTIVE,'inactive':RED},'p2':{'active':BLUE_ACTIVE,'inactive':BLUE}}

screen = pygame.display.set_mode((max_dim,max_dim))

pygame.display.set_caption("Line Up Shapes")

def update_parameters(spaces_list = spaces_list,rotations_list = rotations_list, zoom_list=zoom_list,max_dim = max_dim,shape_size = shape_size, screen=screen):
	spaces = int(random.choice(spaces_list))
	stride = int(max_dim/spaces)
	phase = int(np.random.choice(list(range(stride))))
	num_rotations = int(random.choice(rotations_list))
	zoom = random.choice(zoom_list)
	shapes = {'p1':[],'p2':[]}
	shape_positions = list(range(int(shape_size/2+phase),int(max_dim-shape_size/2+phase),stride))
	#shapes = []
	for player in shapes:
		for s in range(int(num_shapes/2)):
			shape = Polygon(screen,stride,num_rotations,max_dim,zoom,player)
			while shape.area < (max_dim/10)**2:
				shape = Polygon(screen,stride,num_rotations,max_dim,zoom,player)
			shape_pos = (np.random.choice(shape_positions),np.random.choice(shape_positions))
			shape.translate((shape_pos[0]-shape.centroid[0],shape_pos[1]-shape.centroid[1]))
			shapes[player].append(deepcopy(shape))
	return stride,phase,zoom,shapes

def get_screen(screen = screen,flatten = False, grey_scale = True):
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

	# Resize, and add a batch dimension (BCHW)
	return npscreen



def get_pix_ratio(npscreen):
	unique_ls = np.unique(npscreen)
	if len(unique_ls) < 2:
		return False
	else:
		return True


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

	def __init__(self,screen,stride,num_rotations,max_dim,zoom,player,num_points = 'random',points_list = 'random'):
		self.num_rotations = num_rotations
		self.stride = stride
		self.max_dim = max_dim
		self.zoom = zoom
		self.player = player
		if points_list != 'random':
			self.points_list = points_list
			self.num_points = len(points_list)
		else:
			if num_points != 'random':
				self.num_points = num_points
			else:
				self.num_points = int(np.random.choice([3,4,5,6]))
			self.points_list = random_polygon.gen_polygon(int(max_dim/2-max_dim/5),int(max_dim/2+max_dim/5),self.num_points)

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

	def handle_keys(self):
		key = pygame.key.get_pressed()
		dist = 1
		if self.player == 'p1':
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
		else:
			if key[pygame.K_7]:
				#if not self.centroid[0] < self.stride:
				self.translate((-1*self.stride, 0))
			if key[pygame.K_8]:
				#if not self.centroid[0] > max_dim-self.stride: 
				self.translate((self.stride, 0))
			if key[pygame.K_9]:
				#if not self.centroid[1] < self.stride:
				self.translate((0, -1*self.stride))
			if key[pygame.K_0]:
				#if not self.centroid[1] > max_dim-self.stride:
				self.translate((0, self.stride))
			if key[pygame.K_5]:
				self.rotate('left')
			if key[pygame.K_6]:
				self.rotate('right')
			if key[pygame.K_3]:
				self.scale(-1)
			if key[pygame.K_4]:
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

	def draw(self, surface, color = (255,255,255), width = 0):
		draw_points_list = []    #points are not the same as draw points must be shift by max_dim/10 in each dimension as screen has boundary area
		for point in self.points_list:
			draw_points_list.append((point[0],point[1]))
		pygame.draw.polygon(surface, color, draw_points_list, width)

def draw_screen(shapes,active_shapes,screen=screen):
	screen.fill((0, 0, 0))
	#Draw Shapes
	for player in shapes:
		for s in range(len(shapes[player])):
			if s == active_shapes[player]:
				continue 
			shapes[player][s].draw(screen,color=colors[player]['inactive'])
			#if automated:
			#	player.robo_action(optimal_action(player,target))
		shapes[player][active_shapes[player]].draw(screen,colors[player]['active'])


'''
#global transformations
def zoom_screen(direction, shapes, zoom, max_dim = max_dim):
	center = (int(max_dim/2),int(max_dim/2))
	for shape in shapes:
		for i in range(len(shape.rotations)):
			new_points_list = []
			for point in shape.rotations[i]:
				new_points_list.append(((zoom**direction)*(point[0]-center[0])+center[0], (zoom**direction)*(point[1]-center[1])+center[1]))
			shape.rotations[i] = new_points_list
			shape.points_list = shape.rotations[shape.rotation]
			shape.centroid = shape.get_centroid(shape.points_list)

def translate_screen(direction, shapes, max_dim = max_dim):
	for shape in shapes:
		if direction == 0:
			shape.translate((-1*shape.stride, 0))
		elif direction == 1: 
			shape.translate((shape.stride, 0))
		if direction == 2:
			shape.translate((0, -1*shape.stride))
		if direction == 3:
			shape.translate((0, shape.stride))
'''

def get_state_image(state,name='none'):
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
		imarray = imarray.reshape(max_dim,max_dim)
	imarray[imarray > 0] = 255
	imarray[imarray != 255] = 0
	if name == 'none':
		scipy.misc.imsave('images/state_%s.png'%time.time(),imarray)
	else:
		scipy.misc.imsave('images/%s'%name,imarray)

def random_transformation(shapes,zoom):
	for player in shapes:
		active_shape = 0
		action_list = []
		for i in range(len(shapes[player])):
			rots = random.randint(0,shapes[player][active_shape].num_rotations/2+1)
			rot_dir = random.choice([4,5])
			for i in range(rots):
				action_list.append(rot_dir)
			y_dir = random.choice([2,3])
			x_dir = random.choice([0,1])
			s_dir = random.choice([6,7])
			y_amount = random.randint(0,8)
			x_amount = random.randint(0,8)
			s_amount = random.randint(0,3)
			for i in range(y_amount):
				action_list.append(y_dir)
			for i in range(x_amount):
				action_list.append(x_dir)
			for i in range(s_amount):
				action_list.append(s_dir)
			action_list.append(8)

		#cy_dir = random.choice([9,10])
		#cx_dir = random.choice([11,12])
		#cz_dir = random.choice([13,14])
		#cy_amount = random.randint(0,10)
		#cx_amount = random.randint(0,10)
		#cz_amount = random.randint(0,10)
		#for i in range(cy_amount):
		#	action_list.append(cy_dir)
		#for i in range(cx_amount):
		#	action_list.append(cx_dir)
		#for i in range(cz_amount):
		#	action_list.append(cz_dir)
		#print(action_list)	
		for i in action_list:
			if i == 8:
				active_shape = (active_shape+1)%len(shapes[player])
			elif i < 8:
				shapes[player][active_shape].robo_action(i)
			#Translate
			#elif i == 9:
			#	translate_screen(0, shapes)
			#elif i == 10:
			#	translate_screen(1, shapes)
			#elif i == 11:
			#	translate_screen(2, shapes)
			#elif i == 12:
			#	translate_screen(3, shapes)
			#zoom
			#elif i == 13:
			#	zoom_screen(-1, shapes, zoom)
			#elif i == 14:
			#	zoom_screen(1, shapes, zoom)




clock = pygame.time.Clock()

stride,phase,zoom,shapes = update_parameters()
active_shapes = {'p1':0,'p2':0}

running = True 
while running:
	for event in pygame.event.get():
		if event.type == QUIT:
			pygame.quit()
			sys.exit()

		key = pygame.key.get_pressed()
		#switch Shapes
		if keypad:
			if key[pygame.K_KP5]:
				active_shapes['p1'] = (active_shapes['p1']+1)%len(shapes['p1'])
			if key[pygame.K_KP0]:
				active_shapes['p2'] = (active_shapes['p2']+1)%len(shapes['p2'])
		else:
			if key[pygame.K_1]:
				active_shapes['p1'] = (active_shapes['p1']+1)%len(shapes['p1'])
			if key[pygame.K_2]:
				active_shapes['p2'] = (active_shapes['p2']+1)%len(shapes['p2'])

		draw_screen(shapes,active_shapes)		
		'''
		screen.fill((0, 0, 0))
		#pygame.draw.circle(screen, BLUE, (int(max_dim/2), int(max_dim/2)), 20, 0)
		
		for s in range(len(shapes)):
			if s == active_shape:
				continue 
			shapes[s].draw(screen,color=BLUE)
			#if automated:
			#	player.robo_action(optimal_action(player,target))
		shapes[active_shape].draw(screen,color=RED)
		'''
		shapes['p1'][active_shapes['p1']].handle_keys()
		shapes['p2'][active_shapes['p2']].handle_keys()

		#get non-shape key commands

		#camera_keys

		#Translate
		#if keypad:
			#if key[pygame.K_KP6]:
			#	translate_screen(0, shapes)
			#if key[pygame.K_KP4]:
			#	translate_screen(1, shapes)
			#if key[pygame.K_KP2]:
			#	translate_screen(2, shapes)
			#if key[pygame.K_KP8]:
			#	translate_screen(3, shapes)
			#zoom
			#if key[pygame.K_KP7]:
			#	zoom_screen(-1, shapes, zoom)
			#if key[pygame.K_KP9]:
			#	zoom_screen(1, shapes, zoom)
		#else:
			#if key[pygame.K_l]:
			#	translate_screen(0, shapes)
			#if key[pygame.K_i]:
			#	translate_screen(1, shapes)
			#if key[pygame.K_k]:
			#	translate_screen(2, shapes)
			#if key[pygame.K_o]:
			#	translate_screen(3, shapes)
			#zoom
			#if key[pygame.K_n]:
			#	zoom_screen(-1, shapes, zoom)
			#if key[pygame.K_m]:
			#	zoom_screen(1, shapes, zoom)			


		#utility keys
		#update
		if key[pygame.K_u]:
			stride,phase,zoom,shapes= update_parameters()
		#Get Screen	
		if key[pygame.K_g]:
			npscreen = get_screen(screen,flatten = False, grey_scale = True)
			print(npscreen.shape)
			print(npscreen)
			print(get_pix_ratio(npscreen))
			get_state_image(npscreen)
		#Random Transform
		if key[pygame.K_t]:
			random_transformation(shapes,zoom)
		if key[pygame.K_r]: #store state
			stored_shapes = deepcopy(shapes)
			print('shapes stored')
			stored_active_shape = deepcopy(active_shapes)
		if key[pygame.K_y]: #retrieve stored state
			shapes = deepcopy(stored_shapes)
			active_shapes = deepcopy(stored_active_shapes)

		#print
		if key[pygame.K_p]:
			for player in shapes:
				print('shapes for %s'%player)
				for i in range(len(shapes[player])):
					print(i)
					print(shapes[player][i].points_list)
				print('stored_shapes')
				for i in range(len(stored_shapes[player])):
					print(i)
					print(stored_shapes[player][i].points_list)

		pygame.display.update()



		clock.tick(40)

