# Moving Shapes DQN

This repopsitory consists of a game in which 2 dimensional shapes are randomly generated, and can be selected, translated, scaled, and rotated. The game includes a human version with keyboard controls (zoom_game_human.py), and a scripts for training a dqn model to play the game. The dqn model must learn from pixel data how to manipulate the objects from a random initilization to a random target state as quickly as possible.

##Set Up
All scripts can be run if you have a python virtual environment set up correctly. This is easily done with conda, just run 
```
conda env create -f environment.yml
source activate shapes_game
```

## Human Controls
left:          move shape left

right:         move shape right

up:            move shape up

down:          move shape down

q:             rotate shape left

w:             rotate shape right

a:             scale shape down

s:             scale shape up

0:             switch shape

t:             put shapes through series of random transformation

u:             initialize new shapes

i:             move camera left

l:             move camera right

o:             move camera up

k:             move camera down

m:             zoom camera in

n:             zoom camera out

r:             save current state

y:             return to saved state

g:             save image of screen to images




