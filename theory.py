import os
import pickle
import numpy as np
from os.path import join
from training_utils import append

class LinearNetwork (object):

	def __init__ (self, Ws, w_star, q=0.5, eta=0.001, cov=None, out_dir="."):

		self.out_dir = out_dir
		os.makedirs(out_dir, exist_ok=True)

		self.W0s = Ws.copy() # store initial values
		self.Ws = self.W0s.copy()
		self.w_star = w_star
		# add checks on matrix dimensions here


		assert (q > 0) and (q <= 1), "Probability of synaptic connection out of bounds (0,1]."
		self.q = q 		# probability of synaptic connection
		self.eta = eta 	# learning rate

		self.d_input = self.Ws[0].shape[-1]
		self.d_output = self.Ws[-1].shape[0]
		print(f"Input dimension: {self.d_input}; Output dimension: {self.d_output}")
		if cov is None:
			cov = np.eye(self.d_input)
		self.cov = cov
		print("Input-input covariance matrix\n", self.cov)

		self._M = self.cov + np.diagflat( np.diagonal(self.cov) )/self.q
		print("Modified covariance matrix -- entering the \"regularization\" term\n", self._M)

	def save (self, filename):
		with open(join(self.out_dir, filename), "wb") as f:
			pickle.dump(self.Ws, f)

	def load (self, filename):
		with open(join(self.out_dir, filename), "rb") as f:
			self.Ws = pickle.load(f)


	def step (self, t):
		'''
		One-step update of the weights
		------------------------------
		'''

		W = self.Ws[0].copy()
		a = self.Ws[1].copy()

		_w = np.dot( a, W )
		_v = np.dot( (self.w_star - _w), self.cov )

		# gradient of av loss term for W
		_del_W = np.dot( a.T, _v )
		_del_W = self.q * _del_W

		# gradient of av loss term for a
		_del_a = np.dot( _v, W.T )
		_del_a = self.q * _del_a

		# "regularisation" term for W
		_reg_W = np.diagonal( np.dot(a.T, a) )
		_reg_W = _reg_W[:,None] * np.dot( W, self._M )
		_reg_W = (1 - self.q) * _reg_W

		# "regularisation" term for a
		_reg_a = np.dot( W, self._M )
		_reg_a = np.dot( _reg_a,  W.T )
		_reg_a = (1 - self.q) * a * np.diagonal(_reg_a)[None,:]

		# weight update
		self.Ws[0] = W + self.eta * ( _del_W - _reg_W )
		self.Ws[1] = a + self.eta * ( _del_a - _reg_a )
		self.Ws[0] = W + self.eta * _del_W
		self.Ws[1] = a + self.eta * _del_a


	def simulate (self, n_steps, W0s=None, n_save=1):

		if W0s is None:
			W0s = self.W0s.copy()
		self.Ws = W0s

		n_save = min( n_steps, n_save )
		n_save = np.linspace(0, n_steps, n_save+1).astype(int)

		_weights = [np.array([]) for _ in range(len(self.Ws))]
		for n in range(n_steps+1):
			if n in n_save:
				for l, W in enumerate(self.Ws):
				    _weights[l] = append(_weights[l], W)
				    np.save( join(self.out_dir, f"weights_theory_{l+1}.npy"), _weights[l] )
			self.step(n)

		return n_save, _weights


	@property
	def loss (self):
		_loss = None
		return _loss
