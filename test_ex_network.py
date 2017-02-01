import ex_network
import numpy as np

ex_network = ex_network.Network([2,2,1])

inputs = [[.35], [.9]]
target = [.5]

ex_network.train(inputs, target, 1)
