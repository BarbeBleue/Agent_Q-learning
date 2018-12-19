# -*- coding: utf-8 -*

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend, initializers
from keras.utils.generic_utils import get_custom_objects
import numpy as np

from constant import *

import gym #pip install gym
import time

DEBUG = 0

def centered_sigmoid(x):
	""" Customized activation function """
	return (backend.sigmoid(x)) - 0.5

def centered_sigmoid_double(x):
	""" Customized activation function """
	return 2*((backend.sigmoid(x)) - 0.5)

def truncated_output(x):
	""" Activation function for -1 to +1 outputs """
	if x > 1:
		x = 1

	if x < -1:
		x = -1

	return x

def my_loss(ytrue,ypred):
	"""Loss function described in article. """

	return (ytrue - ypred)*(ytrue - ypred)

class AgentBrain :

	#Number of neurons
	_nbInput = 145
	_nbHidden = 30
	_nbOutput = 1

	_reward_sum = 0

	#Tunable parameters
	_T_inv_min = 20  # Inverse of temperature
	_T_inv_max = 60  # Max value of the inverse of temperature
	_discount = 0.9
	_momentum = 0.9  # Momentum factor of the backpropagation algorithm
	_lr = 0.01  # Learning rate of the backpropagation algorithm
	_r_w = 0.1  # Range of the initial weights
	_momentum = 0.9 # Momentum factor of the backpropagation algorithm
	_lr = 0.05 # Learning rate of the backpropagation algorithm
	_r_w = 0.1 # Range of the initial weights

	_learning = True

	_input_vectors = []

	def __init__(self):
		print("Initialization of AgentBrain")
		self._T_inv = self._T_inv_min
		self._input_vectors = []

		self._model = self.build_model()
		print("Built NN model : {0}".format(self._model.summary()))

	def build_model(self):
		model = tf.keras.Sequential()

		get_custom_objects().update({'centered_sigmoid_double': layers.Activation(centered_sigmoid_double)})
		get_custom_objects().update({'centered_sigmoid': layers.Activation(centered_sigmoid)})
		#get_custom_objects().update({'my_loss':layers.Loss(my_loss)})
		#get_custom_objects().update({'truncated_output': layers.Activation(truncated_output)})
		randUnif = initializers.RandomUniform(minval=-0.1, maxval=0.1)

		model.add(layers.Dense(self._nbHidden, input_dim=self._nbInput, kernel_initializer=randUnif))
		model.add(layers.Activation(centered_sigmoid_double))
		model.add(layers.Dense(self._nbOutput, kernel_initializer=randUnif))
		model.add(layers.Activation(centered_sigmoid_double))

		sgd = optimizers.SGD(lr=self._lr, momentum=self._momentum)

		model.compile(loss=my_loss, optimizer=sgd, metrics=['mse'])

		return model

	""" NN save/load functions """
	def save(self, name):
		start = time.time()
		self._model.save(name)
		print("Save time : " + str(time.time() - start))

	def load(self, name):
		start = time.time()
		self._model = models.load_model(name)
		print("Load time : " + str(time.time() - start))

	def savew(self, name):
		start = time.time()
		self._model.save_weights(name)
		print("Save time : " + str(time.time() - start))

	def loadw(self, name):
		start = time.time()
		self._model.load_weights(name)
		print("Load time : " + str(time.time() - start))

	def reset(self):
		self._T_inv = self._T_inv_min
		self._input_vectors = []
		self._reward_sum = 0

	def predict(self,vec):
		"""
		Wrapper for the model.predict
		""" 
		if type(vec) == list :
			print("WARNING - agent_brain.predict : conversion from list to numpy.array")

		#print("Predicting : {0}, type = {1} \n {2}".format(len(vec),type(vec),vec))

		return self._model.predict(vec.reshape(1,self._nbInput))
		#return 0.1

	def add_reward(self,reward):
		self._reward_sum += reward

	def compute_input_vectors(self, input_vec):
		"""
		Gives the input representation in the four directions (N,E,S,W)

		:param input_vec: input representation, current state of the agent 
		:return: a list of numpy.ndarray, the input representations in each direction 
		"""
		input_vectors = []
		if type(input_vec) == list:
			input_vectors.append(np.array(input_vec))
		elif type(input_vec) == np.ndarray:
			input_vectors.append(input_vec)
		else :
			print("WARNING - agent_brain.compute_input_vectors : unknown input_vec")
		
		angle = 0
		for i in range(1, 4):
			angle += 90

			vec = (rotate_sensors(0, input_vec[:52], angle)) #Food sensors
			vec += (rotate_sensors(1, input_vec[52:84], angle)) #Enemy sensors
			vec += (rotate_sensors(2, input_vec[84:124], angle)) #Obstacle sensors
			vec += list(input_vec[124:140]) #Energy
			vec += (rotate_action(input_vec[140:144], angle)) #Action
			vec += list(input_vec[144:145]) #Has collision
			input_vectors.append(np.array(vec))

		return input_vectors

	def show_vectors(self):
		print("Showing vectors")
		for vec in self._input_vectors :
			print("Vector : {0}\n".format(vec))

	def select_action(self, input_vec):
		"""
		Chooses the action to perform.

		This is the first step of the Q learning process. It evaluates the utility of each possible action. 
		Then, a stochastic selector computes the probability of each action to be elected, based on its utility and the temperature. 
		
		:param input_vec: input representation, current state of the agent
		:return: selected action

		"""

		# Compute utilities
		merits = np.zeros(4)
		if not self._input_vectors : #Input vectors have already been computed in adjust_network of the previous step
			self._input_vectors = self.compute_input_vectors(input_vec)

		#self.show_vectors()
		for i,vec in enumerate(self._input_vectors) :
			merits[i] = self.predict(vec)

		if not self._learning :
			# Choose action with maximum merit
			self._action = np.argmax(merits)
			if DEBUG: 
				print("CHOOSE MAX ACTION")
				print("LEAVING agent_brain.select_action : \n\t Merits={0}\n\tAction={1}\n".format(merits,self._action))
		else : 
			# Choose action with a stochastic selector
			sum = 0.0
			for m in merits :
				sum += np.exp(m*self._T_inv)

			proba = []
			for m in merits :
				proba.append( np.exp(m*self._T_inv)/sum )

			self._action = int( np.random.choice(4, 1, p=proba) )

			if DEBUG :
				print("LEAVING agent_brain.select_action : \n\t Merits={0}\n\tProba={1}\n\tAction={2}".format(merits,proba,self._action))

		return self._action

	def adjust_network(self, new_input_vec, reward):
		"""
		Adjusts the utility network of the agent after one simulation step.

		This is the second step of the Q learning process.

		:param new_input_vec:
		:param reward:

		"""

		self.reduce_temperature()

		self._reward_sum += reward
		if DEBUG :
			print("Reward sum : {0}".format(self._reward_sum))

		prev_input_vec = self._input_vectors[self._action]  #save for now the input representation of the previous state and action

		merits = np.zeros(4)
		self._input_vectors = self.compute_input_vectors(new_input_vec)
		for i,vec in enumerate(self._input_vectors) :
			merits[i] = self.predict(vec)

		target = reward + self._discount * np.max(merits)
		if target > 1:  #That way, the optimizer can fit target value with predicted value
			target = 1
		if target < -1:
			target = -1
		target = np.array(target).reshape(1, 1)

		if DEBUG :
			print("LEAVING agent_brain.adjust_network :\
			\n\tMerits={0}\n\ttargetU={1}\n\treward_sum={2}"\
			.format(merits, target, self._reward_sum))

		if self._learning :
			# try to fit the utilities before and after performing the action.
			self._model.fit(prev_input_vec.reshape(1, self._nbInput), target, epochs=1, batch_size=1,verbose=0)
		else:
			if DEBUG :
				print("NO LEARNING IN ADJUST WEIGHTS")

	def is_on_policy(self, input_vecs, action):
		"""
		Tells if action is on policy considering the input vector and current NN weights.
		"""

		on_policy = False

		# Compute utilities
		merits = np.zeros(4)
		for i,vec in enumerate(input_vecs):
			merits[i] = self.predict(vec)

		# Test action probability
		sum = 0.0
		for m in merits:
			sum += np.exp(m*self._T_inv)

		proba = []
		for m in merits:
			proba.append(np.exp(m*self._T_inv)/sum )

		if proba[action] >= 0.01:
			on_policy = True

		if DEBUG:
			print("LEAVING is_on_policy : \n\t Merits={0}\n\tProba={1}\n\tAction={2}".format(merits, proba, action))

		return on_policy

	def adjust_network_replay(self, input_vecs, action, new_input_vecs, reward):
		"""
		Network adjusting for experience replay
		"""
		#Compute qmax of new state
		merits = np.zeros(4)
		for i,vec in enumerate(new_input_vecs) :
			merits[i] = self.predict(vec)

		#Compute expected output
		target = reward + self._discount * np.max(merits)
		if target > 1 : #That way, the optimizer can fit target value with predicted value
			target = 1
		if target < -1:
			target = -1
		target = np.array(target).reshape(1,1)

		#Fit previous state prediction with new state target
		self._model.fit(input_vecs[action].reshape(1,self._nbInput),np.array(target), epochs=1, verbose=0)
	def reduce_temperature(self):
		if self._T_inv < self._T_inv_max:
			self._T_inv *= 1.05

		if DEBUG:
			print(self._T_inv)

	def get_nbHidden(self):
		return self._nbHidden

""" Functions for input representation processing """


def rotate_sensors(sensor_arr_type, sensor_arr_vec, angle):
	"""
	Rotates the input representation of the sensor array
	(algo général mais un peu lourd)

	:param sensor_arr_type: Type of sensor. 0 = food, 1 = enemy, 2 = obstacle
	:param sensor_arr_ vec: Input representation of the sensor array
	:param angle: Rotation angle of the sensor array, clockwise. Possible values : 90,180,270
	:return: Rotated input representation of the sensor array
	"""

	if sensor_arr_type == 0:
		ref_array = SENSOR_X + SENSOR_O + SENSOR_Y
	elif sensor_arr_type == 1:
		ref_array = SENSOR_X + SENSOR_O
	elif sensor_arr_type == 2:
		ref_array = SENSOR_o
	else:
		print("ERROR : unknown sensor_arr_type n°" + str(sensor_arr_type))

	if angle == 90:
		rot_mat = [[0, -1], [1, 0]]
	elif angle == 180:
		rot_mat = [[-1, 0], [0, -1]]
	elif angle == 270:
		rot_mat = [[0, 1], [-1, 0]]
	else:
		print("ERROR : unknown angle value = " + str(angle))

	rot_array = np.dot(ref_array, rot_mat)

	rot_vec = []
	for pos in rot_array:
		ind = ref_array.index(list(pos))
		rot_vec.append(sensor_arr_vec[ind])
	return rot_vec

def rotate_action(input_vec, angle):

	if angle == 90:
		res = np.roll(input_vec, -1)
	elif angle == 180:
		res = np.roll(input_vec, -2)
	elif angle == 270:
		res = np.roll(input_vec, -3)
	else :
		raise ValueError('Invalid angle'+str(angle))

	return list(res)

def centered_sigmoid(x):
	""" Customized activation function """
	return (backend.sigmoid(x)) - 0.5


def centered_sigmoid_double(x):
	""" Customized activation function """
	return 2*((backend.sigmoid(x)) - 0.5)


def truncated_output(x):
	""" Activation function for -1 to +1 outputs """
	if x > 1:
		x = 1

	if x < -1:
		x = -1

	return x
