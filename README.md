# Super Simple Neural Net to Demonstrate Backpropagation 

The is very small neural net (2 node input layer, 2 node hidden layer, and 1 node output layer) that is intended to be a simple demonstration of how to implement the backpropagation learning algorithm. The implementation is based based off of the network diagram below.

![alt tag](https://github.com/nathansegan/simple_backpropagation_example/blob/master/simple_backprop.png)

## Working through this example

Assume a learning rate of 1 (if you dont know what this is, reference below or don't worry about it for now).  We will be using the sigmoid activation function to process inputs and turn them into outputs at the hidden layer and the output layer:
```math
1 / (1 + e^(-z))
```

### Forward Pass

input to top neuron:
```math
(0.35 * 0.1) + (0.9 * 0.8) = 0.755
```

input to bottom neuron:
```math
(0.9 * 0.6) + (0.35 * 0.4) = 0.68
```

output of top neuron:
```math
1 / (1 + e^(-0.755)) = 0.68
```

output of bottom neuron:
```math
1 / (1 + e^(-0.68)) = 0.6637
```

input to final neuron:
```math
(0.3 * 0.68) + (0.9 * 0.6637) = 0.80133
```

output of final neuron:
```math
1 / (1 + e^(-0.80133)) = 0.69
```
### Calculate output error

```math
δ = (target - output) * (1 - output) * output = (0.5 - 0.69) * (1 - 0.69) * 0.69 = -0.0406
```

### Calculate new weights for output layer

```math
w1_new = w1_old + (δ * input) = 0.3 + (-.0406 * 0.68) = 0.272392
w2_new = w2_old + (δ * input) = 0.9 + (-.0406 * 0.6637) = 0.87305
```

### Calculate error for hidden layer

```math
δ1 = δ * w1 * output * (1 - output) = -0.0406 * 0.272392 * (1 - 0.68) * 0.68
δ2 = δ * w2 * output * (1 - output) = -0.0406 * 0.87305 * (1 - 0.6637) * 0.6637
```

### Calculate new weights for hidden layer

```math
w3_new = 0.1 + (-2.406 * 10^-3 * 0.35) = 0.09916
w4_new = 0.1 + (-2.406 * 10^-3 * 0.9) = 0.7978
w5_new = 0.1 + (-7.916 * 10^-3 * 0.35) = 0.3972
w6_new = 0.1 + (-7.916 * 10^-3 * 0.9) = 0.5928
```

## Next: How to Generalize

![alt tag](https://github.com/nathansegan/mnist_neural_network/blob/master/scraps/sample_network.png)

Using the sigmoid activation function:
```math
1 / (1 + e(-z))
```

### Forward Pass
Input pattern is applied and the output is calculated


#### Calculate the input to the hidden layer neurons
```math
in_A = W_ΩA * Ω + W_λA * λ
in_B = W_ΩB * Ω + W_λB * λ
in_C = W_ΩC * Ω + W_λC * λ
```

#### Feed inputs of hidden layer neurons through the activation function
```math
out_A = 1 / (1 + e^( -1 * in_A))
out_B = 1 / (1 + e^( -1 * in_B))
out_C = 1 / (1 + e^( -1 * in_C))
```

#### Multiply the hidden layer outputs by the corresponding weights to calculate the inputs to the output layer neurons
```math
in_α = out_A * W_Aα + out_B * W_Bα + out_C * W_Cα
in_β = out_A * W_Aβ + out_B * W_Bβ + out_C * W_Cβ
```

#### Feed inputs of output layer neurons through the activation function
```math
out_α = 1 / (1 + e^( -1 * in_α))
out_β = 1 / (1 + e^( -1 * in_β))
```


### Reverse Pass
Error of each neuron is calculated and the error is used to mathematically change the weights to minimize them, repeatedly.

_ = subscript, W+ = new weight, W = old weight, δ = error, η = learning rate.

#### Calculate errors of output neurons
```math
δ_α = out_α * (1 - out_α) * (Target_α - out_α)
δ_β = out_β * (1 - out_β) * (Target_β - out_β)
```

#### Change output layer weights
```math
W+_Aα = W_Aα + η * δα * out_A
W+_Aβ = W_Aβ + η * δβ * out_A

W+_Bα = W_Bα + η * δα * out_B
W+_Bβ = W_Bβ + η * δβ * out_B

W+_Cα = W_Cα + η * δα * out_C
W+_Cβ = W_Cβ + η * δβ * out_C
```

#### Calculate (back-propagate) hidden layer errors
```math
δ_A = out_A * (1 – out_A) * (δ_α * W_Aα + δ_β * W_Aβ)
δ_B = out_B * (1 – out_B) * (δ_α * W_Bα + δ_β * W_Bβ)
δ_C = out_C * (1 – out_C) * (δ_α * W_Cα + δ_β * W_Cβ)
```

#### Change hidden layer weights
```math
W+_λA = W_λA + η * δ_A * in_λ 
W+_ΩA = W_ΩA + η * δ_A * in_Ω

W+_λB = W_λB + η * δ_B * in_λ 
W+_ΩB = W_ΩB + η * δ_B * in_Ω

W+_λC = W_λC + η * δ_C * in_λ
W+_ΩC = W_ΩC + η * δ_C * in_Ω
```
