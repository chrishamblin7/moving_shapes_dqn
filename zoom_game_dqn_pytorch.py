import random
from pygame.locals import *
import os
import random
import pygame
import math
import sys
import numpy as np
import time
np.set_printoptions(threshold=np.inf)
import pdb
import scipy.misc
import random_polygon
from math import pi, cos, sin
from copy import deepcopy
from importlib import reload

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

sys.path.insert(0,'models/scipts/')
import dqn_basic
import dist_dqn
import rainbow_dqn


start = time.time()


os.environ["SDL_VIDEO_CENTERED"] = "1"


with_window = True
model_kind = 'atari'
automated = False
max_dim = int(500)
num_shapes = 7
shape_size = int(10)
#target_size = int(player_size+2)
#spaces_list = [int(max_dim/10),int(max_dim/5),int(max_dim/2)]
spaces_list = [int(max_dim/5)]
#rotations_list = [4,8,16]
rotations_list = [64]
zoom_list = [1.2]
max_moves = 40

epochs = 4501
gamma = 0.975
epsilon = 1
batchSize = 100
buffer = 20000


BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED = (255,   0,   0)
GREEN = (  0, 255,   0)
BLUE = (  0,   0, 255)


#for running on cluster with no graphics
if not with_window:
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ["SDL_VIDEODRIVER"] = "dummy"


pygame.init()

screen = pygame.display.set_mode((max_dim,max_dim))

pygame.display.set_caption("Line Up Shapes")


def update_parameters(spaces_list = spaces_list,rotations_list = rotations_list, zoom_list=zoom_list,max_dim = max_dim,shape_size = shape_size, screen=screen):
	spaces = int(random.choice(spaces_list))
	stride = int(max_dim/spaces)
	phase = int(np.random.choice(list(range(stride))))
	num_rotations = int(random.choice(rotations_list))
	zoom = random.choice(zoom_list)
	shapes = {}
	shape_positions = list(range(int(shape_size/2+phase),int(max_dim-shape_size/2+phase),stride))
	shapes = []
	for s in range(num_shapes):
		shape = Polygon(screen,stride,num_rotations,max_dim,zoom)
		while shape.area < (max_dim/10)**2:
			shape = Polygon(screen,stride,num_rotations,max_dim,zoom)
		shape_pos = (np.random.choice(shape_positions),np.random.choice(shape_positions))
		shape.translate((shape_pos[0]-shape.centroid[0],shape_pos[1]-shape.centroid[1]))
		shapes.append(deepcopy(shape))
	return stride,phase,zoom,shapes

def get_screen(screen = screen,flatten = False, grey_scale = True):
	npscreen = pygame.surfarray.array3d(screen)  # transpose into torch order (CHW)
	npscreen = np.ascontiguousarray(npscreen, dtype=np.float32) / 255
	if grey_scale:
		new_screen = np.zeros((npscreen.shape[0],npscreen.shape[1]))
		for h in range(new_screen.shape[0]):
			for w in range(new_screen.shape[1]):
				for c in range(npscreen.shape[2]):
					if npscreen[h,w,c] != 0:
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

	def __init__(self,screen,stride,num_rotations,max_dim,zoom,num_points = 'random',points_list = 'random'):
		self.num_rotations = num_rotations
		self.stride = stride
		self.max_dim = max_dim
		self.zoom = zoom
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

	def draw(self, surface, color = (255,255,255), width = 0):
		draw_points_list = []    #points are not the same as draw points must be shift by max_dim/10 in each dimension as screen has boundary area
		for point in self.points_list:
			draw_points_list.append((point[0],point[1]))
		pygame.draw.polygon(surface, color, draw_points_list, width)


def draw_screen(shapes,active_shape,screen=screen):
	screen.fill((0, 0, 0))
    #Draw Shapes
	for s in range(len(shapes)):
		if s == active_shape:
			continue 
		shapes[s].draw(screen,color=BLUE)
		#if automated:
		#	player.robo_action(optimal_action(player,target))
	shapes[active_shape].draw(screen,color=RED)

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

def random_transformation(shapes,zoom):
	if target_screen:
		old_shapes = []
		for shape in shapes:
			old_shapes.append([shape.rotations,shape.rotation])   #store original rotations and rotation index
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
			translate_screen(0, shapes)
		elif i == 10:
			translate_screen(1, shapes)
		elif i == 11:
			translate_screen(2, shapes)
		elif i == 12:
			translate_screen(3, shapes)
		#zoom
		elif i == 13:
			zoom_screen(-1, shapes, zoom)
		elif i == 14:
			zoom_screen(1, shapes, zoom)


def makeMove(action, shapes, active_shape, screen, targetscreen, save=True):

    screen.fill((0, 0, 0))
    #Draw Shapes
	if action == 8:
		active_shape = (active_shape+1)%len(shapes)
	elif action < 8:
		shapes[active_shape].robo_action(action)
	#Translate
	elif action == 9:
		translate_screen(0, shapes)
	elif action == 10:
		translate_screen(1, shapes)
	elif action == 11:
		translate_screen(2, shapes)
	elif action == 12:
		translate_screen(3, shapes)
	#zoom
	elif action == 13:
		zoom_screen(-1, shapes, zoom)
	elif action == 14:
		zoom_screen(1, shapes, zoom)

	draw_screen(shapes,active_shape)
    pygame.display.update()

    currentscreen3d = get_screen(screen,grey_scale=False)
    currentscreen = get_screen(screen)
    state = np.concatenate((currentscreen3d,np.array([targetscreen])))
    return currentscreen3d, currentscreen, state


def getReward(currentscreen,targetscreen, reward_type = 'pointwise', scale = 10):
	if reward_type == 'pointwise':
		contrast = currentscreen - targetscreen
		unique, counts = np.unique(contrast, return_counts=True)
		score = dict(zip(unique,counts))[0]
		reward = score/currentscreen.size
    return reward*scale

def save_model(model,name):
    # serialize model to JSON
    model_json = model.to_json()
    with open("models/%s.json"%name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/%s.h5"%name)
    print("Saved model to disk")

def get_state_image(state,name='none'):
    '''utility function to save an image of the numpy 'state', to make sure it matches game display'''
    if state.ndim == 1:
        imarray = np.reshape(state,(int(np.sqrt(len(state))),int(np.sqrt(len(state)))))
        imarray = np.array([imarray,imarray,imarray])
        imarray = imarray.transpose(1,0,2)
    elif state.ndim == 2:
    	imarray = np.array([state,state,state])
    	imarray = imarray.transpose(1,0,2)
    else:
        imarray = state.transpose(1,0,2)
        imarray = imarray.reshape(max_dim,max_dim)
    imarray[imarray > 0] = 255
    imarray[imarray != 255] = 0
    if name == 'none':
        scipy.misc.imsave('images/state_%s.png'%time.time(),imarray)
    else:
        scipy.misc.imsave('images/%s'%name,imarray)

###MODEL###

in_dim = (max_dim,max_dim,4)
rms = RMSprop()
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

if model_kind == 'atari':
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(8, 8),
                 activation='relu',
                 strides = 4,
                 input_shape=in_dim))
    model.add(Conv2D(64, (4, 4), activation='relu',strides = 2))
    model.add(Conv2D(64, (3, 3), activation='relu',strides = 1))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    model.compile(loss='mse', optimizer=adam)  
else:
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(in_dim)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_actions, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    model.compile(loss='mse', optimizer=adam)


### RUN GAME ###

### initial start game initialization (might be redundant code)###
clock = pygame.time.Clock()
stride,phase,zoom,shapes = update_parameters()
active_shape = 0

### RUN MODEL WITH EXPERIENCE REPLAY ###
model.compile(loss='mse', optimizer=rms)#reset weights of neural network

inputl_shape = (1,max_dim,max_dim,4)
data_shape = (max_dim,max_dim,4)

replay = []
#stores tuples of (S, A, R, S')
h = 0
#pdb.set_trace()
for i in range(epochs):
    print('epoch %s'%i)
    #pdb.set_trace()

    stride,phase,zoom,shapes = update_parameters()

	draw_screen(shapes,active_shape)

    pygame.display.update()

    currentscreen3d = get_screen(screen, grey_scale=False)  #get state as 3 channel image
    currentscreen = get_screen(screen)
    stored_shapes = deepcopy(shapes)              #store state   of shape objects
	stored_active_shape = deepcopy(active_shape)
	good_transform = False
	print('finding good transform . . .')
	while not good_transform:  
    	random_transformation(shapes,zoom)            # randomly transform state 
    	draw_screen(shapes,active_shape)   
    	pygame.display.update()
    	targetscreen = get_screen(screen, grey_scale=True)   #store target as grey scale
    	print(currentscreen.shape)
    	print(currentscreen3d.shape)
    	print(currentscreen3d)
    	print(targetscreen.shape)
    	state = np.concatenate((currentscreen3d,np.array([targetscreen])))
    	shapes = deepcopy(stored_shapes)
		active_shape = deepcopy(stored_active_shape)
		draw_screen(shapes,active_shape)
		pygame.display.update()
		if get_pix_ratio:      # check to make sure our target is not all white or all black
			good_transformation = True  

	print('found')
    #get_state_image(state)
    status = 1
    #while game still in progress
    iters = 0
    while(status == 1):
        iters += 1
        print('move %s'%iters)
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state, batch_size=1)
        #optimal_options = optimal_action(player,target)
        #suboptimal_options = suboptimal_action(optimal_options)
        if (random.random() < epsilon):
            action = np.random.randint(0,15)
            print('action %s'%action)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
            print('action %s: Q policy'%action)
        #Take action, observe new state S'
        currentscreen3d, currentscreen, new_state = makeMove(action, shapes, active_shape, screen, targetscreen)
        #Observe reward
        getReward(currentscreen,targetscreen, reward_type = 'pointwise')
        print('Reward: %s'%reward)
        print('Qval from model: %s'%qval)

        #Experience replay storage
        if (len(replay) < buffer): #if buffer not filled, add to it
            replay.append((state, action, reward, new_state))
        else: #if buffer full, overwrite old values
            if (h < (buffer-1)):
                h += 1
            else:
                h = 0
            replay[h] = (state, action, reward, new_state)
            #randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            X_train = []
            y_train = []
            for memory in minibatch:
                #Get max_Q(S',a)
                old_state_memory, action_memory, reward_memory, new_state_memory = memory
                old_qval = model.predict(old_state_memory.reshape(inputl_shape), batch_size=1)
                newQ = model.predict(new_state_memory.reshape(inputl_shape), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,num_actions))
                y[:] = old_qval[:]
                #if reward_memory != 10: #non-terminal state
                update = (reward_memory + (gamma * maxQ))
                #else: #terminal state
                #	update = reward_memory
                y[0][action_memory] = update
                X_train.append(old_state_memory.reshape(data_shape))
                y_train.append(y.reshape(num_actions,))

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            print("Game #: %s" % (i,))
            model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=1)
        #pdb.set_trace()
        state = new_state
        #debugging
        #if not (len(replay) < buffer):
        #    pdb.set_trace()
        if iters > max_moves: #if reached terminal state, or too many moves taken update game status
            status = 0
        #clear_output(wait=True)
    if epsilon > 0.1: #decrement epsilon over time
        epsilon -= (1/epochs)
        #epsilon -= (1/100)
    if i in checkpoints:
        end = time.time()
        elapse_time = end -start
        print('TIME to epoch %s: %s'%(i,elapse_time))
        save_model(model,'%sepoch_%s'%(i,model_kind))
#pygame.quit()

# serialize model to JSON
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/model.h5")
print("Saved model to disk")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch dqn shape moving game')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=4001, metavar='N',
                        help='number of epochs to train (default: 4000)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.975, metavar='G',
                        help='discount factor for future reward (default: 0.975)')
    parser.add_argument('--buffer', type=int, default=20000, metavar='B',
                        help='Number of states to store in exp_replay (default: 20000)')
    parser.add_argument('--max_moves', type=int, default=40, metavar='MM',
                        help='Max moves before reinitializing (default: 20000)')    
    parser.add_argument('--win_dim', type=int, default=84, metavar='WD',
                        help='window dimension, input int, win = int x int (default: 84)')
    parser.add_argument('--shape_size', type=int, default=8, metavar='SS',
                        help='Average size of intial game shapes (default: 8)')
    parser.add_argument('--num_shapes', type=int, default=4, metavar='NS',
                        help='Number of shapes in game (default: 4)')
    parser.add_argument('--trans_step', type=int, default=4, metavar='TS',
                        help='Number of pixels jumped when translating shape (default: 4)')
    parser.add_argument('--zoom_ratio', type=float, default=1.2, metavar='ZR',
                        help='Scaling ratio when zooming (default: 1.2)')
    parser.add_argument('--num_rotations', type=int, default=32, metavar='NR',
                        help='Number of discrete rotation positions (default: 32)')    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--show_window', action='store_true', default=False,
                        help='show game window while running')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model', type=str, default='dqn_basic', metavar='M',
                        help='neural net model to use (default: dqn_basic, other options: dist_dqn, rainbow_dqn)')


spaces_list = [int(max_dim/5)]
#rotations_list = [4,8,16]
rotations_list = [64]
zoom_list = [1.2]
max_moves = 40

epochs = 4501
gamma = 0.975
epsilon = 1
batchSize = 100
buffer = 20000


    args = parser.parse_args()
    print('running with args:')
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('using cuda')

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model_dict = {'dqn_basic':dqn_basic.DQN,
                  'dist_dqn':dist_dqn.DistributionalDQN,
                  'rainbow_dqn':rainbow_dqn.RainbowDQN
				 }

    model = model_dict[args.model](num_classes=args.cutoff).to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    args.outputdir = '%s_%s_cutoff%s_adam2_%s'%(args.input_data.replace('/','_'),args.model.split('.')[-1],args.cutoff,time.strftime('%m-%d-%Y'))

    if args.save_outputs:
        if not os.path.exists(os.path.join('outputs',args.outputdir)):
            os.mkdir(os.path.join('outputs',args.outputdir))
        args.testing_log = open(os.path.join('outputs',args.outputdir,'testing_log.csv'),'w+')
        for i in range(1,args.cutoff+1):
             args.testing_log.write('%s acc,%s guessed,%s F1'%(str(i),str(i),str(i)))
        args.testing_log.write('\n')
        args.testing_log.flush()

    #writing to tensorboard
    ##writer.add_graph(model,next(iter(train_loader))[0])    
    
    model_epochs = [47]
    
    for epoch in range(1, args.epochs + 1):
        #adjust learning rate
        #if epoch > 600:
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = args.lr/5
        #if epoch > 700:
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = args.lr/10
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)
        if args.save_model and epoch in model_epochs:
            torch.save(model,os.path.join('outputs',args.outputdir,'model_%s.pt'%str(epoch)))



if __name__ == '__main__':
    start_time = time.time()
    main()
    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))