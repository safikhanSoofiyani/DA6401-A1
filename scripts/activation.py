import numpy as np

def sigmoid(pre_act):
  try:
    return (1.0/(1.0+np.exp(-pre_act)))
  except:
    print("error in sigmoid")
    
def tanh(pre_act):
  return (np.tanh(pre_act))


def relu(pre_act):
  return (np.maximum(0,pre_act))

def softmax(x):
  try:
    return(np.exp(x)/np.sum(np.exp(x)))
  except:
    print("error in softmax")
    
    
def sigmoid_derivative(x):
  return sigmoid(x)*(1-sigmoid(x))

def tanh_derivative(x):
  return 1.0 -tanh(x)**2


def relu_derivative(x):
  return 1. * (x>0)

def softmax_derivative(x):
  return softmax(x) * (1-softmax(x))

def derivative(A, activation):

  '''This function is essentially a caller function. It takes in the 
  kindof activation funciton used as well as the value and calls the 
  respective activation functions derivative

  Input is the actual data and the choice of activation funtion.

  Output is the derivative of that data wrt the activation function'''

  if activation == "sigmoid":
    return sigmoid_derivative(A)
  elif activation == "tanh":
    return tanh_derivative(A)
  elif activation == "relu":
    return relu_derivative(A)
