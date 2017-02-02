import numpy as np

class Network(object):
	
	# sizes = [2,3,2]
	def __init__(self, sizes):
		self.weights = [[[.1,.8],[.4,.6]],[.3,.9]]
		# save the outputs of each layer to use in calculating error
		self.outputs = []
	
	def train(self, inputs, target, learning_rate):
		self.do_backprop(inputs, target)
		self.feed_forward(inputs)
	
	def feed_forward(self, inputs):
		
		print "inputs:\n{0}\n\ninput->h_layer weights:\n{1}\n".format(inputs, self.weights[0])
		
		# calculate input to hidden layer neurons
		in_hlayer = np.dot(self.weights[0], inputs)
		print "hidden layer inputs:\n{0}\n".format(in_hlayer)
		
		# feed inputs to hidden layer through activation function
		out_hlayer = np.array([self.sigmoid(z) for z in in_hlayer]).reshape(2,)
		self.outputs.append(out_hlayer)
		print "hidden layer outputs:\n{0}\n".format(out_hlayer)
		
		# multiply hidden layer outputs by weights to get input to output layer
		# the only reason to surround w/ [] is to make in_olayer iterable for feeding inputs through activation
		# not necessary for this example, but wanted to show more of a generalization
		in_olayer = [np.dot(self.weights[1], out_hlayer.transpose())]
		print "output layer input:\n{0}\n".format(in_olayer)
		
		# feed inputs to output layer through activation function
		out_olayer = np.array([self.sigmoid(z) for z in in_olayer])
		print "output layer output:\n{0}\n".format(out_olayer)
		return out_olayer
		
	def do_backprop(self, inputs, target):
		# forward pass
		output = self.feed_forward(inputs)
		
		# reverse pass
		
		# calculate error of output
		error_olayer = np.multiply(np.multiply(output, np.subtract(np.add(np.zeros((1,)), 1.0), output)), np.subtract(target, output)) 
		print "output error:\n{0}\n".format(error_olayer)
		
		# change hlayer->olayer weights
		for index in range(0,len(self.weights[1])):
			self.weights[1][index] = self.weights[1][index] + np.dot(error_olayer[0], self.outputs[0][index])
		print "new hlayer->olayer weights:\n{0}\n".format(self.weights[1])
		
		# calculate errors of hidden layer
		error_hlayer = np.multiply(np.multiply(error_olayer[0], self.weights[1]), np.multiply(self.outputs[0], np.subtract(np.add(np.zeros((1,)), 1.0), self.outputs[0])))
		print "hidden layer errors:\n{0}\n".format(error_hlayer)
		
		# change input->hlayer weights
		for index in range(0,len(self.weights[0])):
			self.weights[0][index] = self.weights[0][index] + np.dot(error_hlayer[0], inputs[index])
		print "new inputs->hlayer weights:\n{0}\n".format(self.weights[0])
	
	def sigmoid(self, z):
    	# sigmoid activation function
		return 1.0 / (1.0 + np.exp(-z))
