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
import game_functions
from math import pi, cos, sin
from copy import deepcopy
from subprocess import call
import time
import scipy.misc
from pprint import pprint
import argparse
import pdb

np.set_printoptions(threshold=np.inf)

os.environ["SDL_VIDEO_CENTERED"] = "1"

pygame.init()


'''
keypad = False
max_dim = int(84)
num_shapes = 3
shape_size = int(10)
#target_size = int(player_size+2)
#spaces_list = [int(max_dim/10),int(max_dim/5),int(max_dim/2)]
trans_step = 2
spaces_list = [int(max_dim/trans_step)] #spaces_list = [int(max_dim/5)]
#rotations_list = [4,8,16]
rotations_list = [64]
zoom_list = [1.2]
'''

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED = (255,   100,   100)
RED_ACTIVE = (255, 0, 0)
GREEN = (  0, 255,   0)
BLUE = (  0,   0, 255)


########   SETTINGS   ###########

# Command Line Arguments
parser = argparse.ArgumentParser(description='Human playable block game')

parser.add_argument('--world-transforms', action='store_true', default=False,
					help='include world transforms (camera zoom and rotation) in available actions (default: False)')
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
parser.add_argument('--keypad', action='store_true', default=False,
					help='include world transforms (camera zoom and rotation) in available actions (default: False)')


args = parser.parse_args()
print('running with args:')
print(args)

screen = pygame.display.set_mode((args.win_dim,args.win_dim))

pygame.display.set_caption("Line Up Shapes")


clock = pygame.time.Clock()

phase,shapes = game_functions.update_parameters(screen,args)
active_shape = 0

running = True 
while running:
	for event in pygame.event.get():
		if event.type == QUIT:
			pygame.quit()
			sys.exit()

		key = pygame.key.get_pressed()
		#switch Shapes
		if args.keypad:
			if key[pygame.K_KP5]:
				active_shape = (active_shape+1)%len(shapes)
		else:
			if key[pygame.K_0]:
				active_shape = (active_shape+1)%len(shapes)			

		screen.fill((0, 0, 0))
		#pygame.draw.circle(screen, BLUE, (int(max_dim/2), int(max_dim/2)), 20, 0)
		for s in range(len(shapes)):
			if s == active_shape:
				continue 
			shapes[s].draw(screen,color=RED)
			#if automated:
			#	player.robo_action(optimal_action(player,target))
		shapes[active_shape].draw(screen,color=RED_ACTIVE)
		shapes[active_shape].handle_keys()

		#get non-shape key commands

		#camera_keys
		#Translate
		if args.keypad:
			if key[pygame.K_KP6]:
				game_functions.translate_screen('cam_right', shapes, args)
			if key[pygame.K_KP4]:
				game_functions.translate_screen('cam_left', shapes, args)
			if key[pygame.K_KP2]:
				game_functions.translate_screen('cam_down', shapes, args)
			if key[pygame.K_KP8]:
				game_functions.translate_screen('cam_up', shapes, args)
			#zoom
			if key[pygame.K_KP7]:
				game_functions.zoom_screen('zoom_in', shapes, args)
			if key[pygame.K_KP9]:
				game_functions.zoom_screen('zoom_out', shapes, args)
		else:
			if key[pygame.K_l]:
				game_functions.translate_screen('cam_right', shapes, args)
			if key[pygame.K_i]:
				game_functions.translate_screen('cam_left', shapes, args)
			if key[pygame.K_k]:
				game_functions.translate_screen('cam_down', shapes, args)
			if key[pygame.K_o]:
				game_functions.translate_screen('cam_up', shapes, args)
			#zoom
			if key[pygame.K_n]:
				game_functions.zoom_screen('zoom_in', shapes, args)
			if key[pygame.K_m]:
				game_functions.zoom_screen('zoom_out', shapes, args)			


		#utility keys
		#update
		if key[pygame.K_u]:
			phase,shapes= game_functions.update_parameters(screen, args)
		#Get Screen	
		if key[pygame.K_g]:
			npscreen = get_screen(screen,flatten = False, grey_scale = True)
			print(npscreen.shape)
			print(npscreen)
			print(get_pix_ratio(npscreen))
			get_state_image(npscreen)
		#Random Transform
		if key[pygame.K_t]:
			game_functions.random_transformation(shapes,args)
		if key[pygame.K_r]: #store state
			stored_shapes = deepcopy(shapes)
			print('shapes stored')
			stored_active_shape = deepcopy(active_shape)
		if key[pygame.K_y]: #retrieve stored state
			shapes = deepcopy(stored_shapes)
			active_shape = deepcopy(stored_active_shape)
		if key[pygame.K_z]:
			print('objective function between current shapes and stored shapes')
			game_functions.objective_func(shapes,stored_shapes,active_shape,args)

		#print
		if key[pygame.K_p]:
			print('shapes')
			for i in range(len(shapes)):
				print(i)
				print(shapes[i].points_list)
			print('stored_shapes')
			try:
				for i in range(len(stored_shapes)):
					print(i)
					print(stored_shapes[i].points_list)
			except:
				print('no stored shapes')
			print('active shape')
			print('centroid: %s'%str(shapes[active_shape].centroid))


		pygame.display.update()



		clock.tick(40)

