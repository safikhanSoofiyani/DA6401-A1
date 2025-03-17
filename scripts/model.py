import numpy as np
import copy
from scripts.activation import sigmoid, tanh, relu, softmax, softmax_derivative, derivative
from scripts.initialization import Xavier, Random

def nn_init(layer_sizes, init_type = "random"):

  '''This funciton is used to initialize the neural network with the 
  choice of the initialization.

  Input is the broad sizes of the various layers stored in a list and the
  choice of initialization type: xavier or random

  Output is the parameters dictionary which stores the various parameters
  for each layer.'''

  # The parameters are stored as a dictionary with each layer having its own
  # key for W and b. 

  # initializing parameters for the neural network, 
  params={}
  if(init_type=="xavier"):
    params = Xavier(layer_sizes)

  elif(init_type=="random"):
    params = Random(layer_sizes)

  else:
    print("Enter a valid weight initilization type")

  return params

def forward_prop(X, y, params, active, layer_sizes):
  
  '''This function is used to forward propagate the data point and return 
  the predicted label.

  Input is the given data point (only one data point) and its respective true 
  label vector, the parameters dictionary, the activation functions choice
  and the overall architecture of the network.

  Output is the predicted y label i.e., y_hat and the list of
  various preactivations and post activations for each neuron of each layer.
  (Each is stored as a list of list) '''


  # Extracting only the image data not the label for the image data
  out=copy.deepcopy(X)
  out=out.reshape(-1,1)
  
  #These are stored just to make it easy to keep track of the indices along with layers.
  h=[out] # To save the activations for each neuron in a layer
  a=[out] # To save the preactivation for each neuron in a layer

  if(active=="sigmoid"):
    for i in range(1,len(layer_sizes)-1):
      weights = params["w"+str(i)]
      biases = params["b"+str(i)]
      
      #Actual Forward Propagation logic
      out = np.dot(weights,h[i-1])+biases
      a.append(out)
      post_a = sigmoid(out)
      h.append(post_a)
  
  elif(active=="tanh"):
    for i in range(1,len(layer_sizes)-1):
      weights=params["w"+str(i)]
      biases=params["b"+str(i)]
      
      #Actual Forward Propagation logic
      out=np.dot(weights,h[i-1])+biases
      a.append(out)
      post_a=tanh(out)
      h.append(post_a)
  
  elif(active=="relu"):
    for i in range(1,len(layer_sizes)-1):
      weights=params["w"+str(i)]
      biases=params["b"+str(i)]
      
      #Actual Forward Propagation logic
      out=np.dot(weights,h[i-1])+biases
      a.append(out)
      post_a=relu(out)
      h.append(post_a)       
  else:
    print("Enter a valid activation function") 

  # Final step for forward propagation, using softmax.
  weights=params["w"+str(len(layer_sizes)-1)]
  biases=params["b"+str(len(layer_sizes)-1)]
  
  out=np.dot(weights,h[len(layer_sizes)-2])+biases
  a.append(out)
  y_hat=softmax(out)
  h.append(y_hat)
  
  
  #in h we  are storing values for layers right from input till output
  #h0 is input
  #in a we are storing values for layers right from input till output
  #a0 is input

  return h,a,y_hat

def back_prop(y, y_hat, h, a, params, loss_type, layer_sizes, activation):
  
  '''This is the heart of the code. It is used to back propagate the calculated 
  error and calculating the gradients for each required entity

  The input is the true label, predicted label, preactivations, postactivations,
  paramters dicionary, the type of loss considered, overall network architecture
  and the choice of activation function.

  The output is the gradient dictionary which stores the gradients calculated
  for each parameter. '''

  # We are considering point by point. i.e., we are propagating only one point,
  # then back propagation that single point only. 

  #here both y_hat and y are assumed to be column vectors

  # Initializing the empty dictionary to store the gradients
  grad = {}

  if loss_type == "squared_loss":
    grad["dh"+str(len(layer_sizes)-1)] = (y_hat - y)
    grad["da"+str(len(layer_sizes)-1)] = (y_hat - y) * softmax_derivative(a[len(layer_sizes)-1])

  elif loss_type == 'cross_entropy':
    # Here actually it should be one hot vector (As seen in class).
    # But y does the same job (since it is also one hot encoded)
    grad["da"+str(len(layer_sizes)-1)] = -(y-y_hat)
    grad["dh"+str(len(layer_sizes)-1)] = -(y/y_hat)

  for i in range(len(layer_sizes)-1, 0, -1 ):
    #print(i)
    # Not considering L2 Regularization here. Instead will cumulate in the update section
    # As referred from the resource pointed in Regularization section.

    grad["dw"+str(i)] = np.dot(grad["da"+str(i)], np.transpose(h[i-1]))
    grad["db"+str(i)] = grad["da"+str(i)]

    #Since we are going backwards, we wont execute these for the final iteration
    if i > 1:
      grad["dh"+str(i-1)] = np.dot(np.transpose(params["w"+str(i)]), grad["da"+str(i)])
      grad["da"+str(i-1)] = np.multiply(grad["dh" + str(i-1)], derivative(a[i-1],activation))
 
  return grad


def grad_calculate_batchwise(X, Y, parameters, activation, layers, loss_function):

  ''' This function is used to calculate the cumulative gradient of the given
  batch of points (used for mini batch gradient descent and its variants)

  Input is the current batch of data points, corresponding true labels, 
  parameters of the network, choice of the activation function, overall
  architecure of the network and the loss function

  Output is the dictionary which has the cumulative gradients collected 
  for the entire batch of data points. (structure of the dictionary is
  same as the parameters dictionary)'''


  #Initializing the empty dictionary
  grads={}
  grads.clear() 

  #iterate over all the points in the current batch
  for j in range(len(X)):

    #Reshaping the labels, to get a column vector
    y = np.reshape(Y[j], (-1,1))

    #Feed forward the data point
    h,a,y_hat = forward_prop(X[j], y, parameters, activation, layers)

    #backpropagate the error.
    new_grads = back_prop(y,y_hat, h,a, parameters, loss_function, layers, activation)

    #keep collecting the gradients for all the data
    if j == 0:
      # if j is 0 means it is the first batch and hence it will be equal to the 
      # calculated gradients
      grads = copy.deepcopy(new_grads)

    else:
      # For remaining cases, we increment the previous gradient values with the 
      # current gradient values.
      for k in range(len(layers)-1,0,-1):
        grads["dw"+str(k)] += new_grads["dw"+str(k)]
        grads["db"+str(k)] += new_grads["db"+str(k)]
  
  return grads



def predict(X ,y,parameters,activation,layer_sizes):

  '''This function is used to simply take a models parameters
      and run the data points using forward prop, and return the outputs 
      of all the input data points
      
      The input is the data points, true labels, parameters of the network,
      Choice of the activation function, overall architecture of the network
      
      The output is the list of all the predicted values for all the data points'''

  result = []

  #Iterate over all the data points in the given data set
  for i in range(len(X)):

    #forward propagate the data point
    h,a,y_hat = forward_prop(X[i], y[i], parameters, activation, layer_sizes)

    #converting y_hat to a 1d array to match with the y
    y_hat = y_hat.flatten()

    #storing the result into the result list
    result.append(y_hat)
  
  return result
