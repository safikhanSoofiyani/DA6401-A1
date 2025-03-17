import numpy as np
import random


def Xavier(layer_sizes):

  '''This function is used to get the xavier initialization for the 
  given network architecture.

  It takes input layer_sizes which is a list of the number of neurons
  in each hidden layer.

  It returns a dictionary which has the various w and b parameters,
  initialized using xavier initialization'''

  params = {}
  for i in range(1,len(layer_sizes)):
      norm_xav=np.sqrt(6)/np.sqrt(layer_sizes[i]+layer_sizes[i-1])
      params["w"+str(i)]=np.random.randn(layer_sizes[i],layer_sizes[i-1])*norm_xav
      params["b"+str(i)]=np.zeros((layer_sizes[i],1))
  
  return params


def Random(layer_sizes):

  '''This function is used to get the random initialization for the 
  given network architecture.

  It takes input layer_sizes which is a list of the number of neurons
  in each hidden layer.

  It returns a dictionary which has the various w and b parameters,
  initialized using random initialization'''

  params = {}
  for i in range(1,len(layer_sizes)):
      params["w"+str(i)]=0.01*np.random.randn(layer_sizes[i],layer_sizes[i-1])
      params["b"+str(i)]=0.01*np.random.randn(layer_sizes[i],1)

  return params