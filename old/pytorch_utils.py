'''  Utility Functions for use with pytorch '''

import torch
import numpy as np
import sys

def to_torch_net_input(array, input_dim = 4):
	if type(array) == np.ndarray or type(array) == np.array:
		while array.ndim < input_dim:
			array = np.array([array])
		return torch.from_numpy(array)
	elif isinstace(array,torch.Tensor):
		while len(array.shape) < input_dim:
			array = array.unsqueeze(0)
		return array
	else:
		sys.exit("Input to to_torch_net_input is not an np array or tensor, ya dangus!")
