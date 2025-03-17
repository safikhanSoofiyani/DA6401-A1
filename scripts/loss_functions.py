import numpy as np

def MSE(y, y_hat):
  error = np.sum(((y - y_hat)**2) / (2 * len(y)))
  return error

def CrossEntropy(y, y_hat):
  error = - np.sum( np.multiply(y , np.log(y_hat)))/len(y)
  return error



# Calculating loss 
def loss_calc(loss_name, y, y_hat, lambd, layer_sizes, parameters):

  '''This function is used to calculate the L2 Regularized loss.
  
  Input is the loss name which denotes the type of loss, the true labels,
  the predicted labels, lambda (i.e., for L2 Regularization), architecture 
  of the network and the parameters dictionary

  Output is the L2 Regularized Loss'''

  error=0
  if(loss_name == "squared_loss"):
    error=MSE(y, y_hat)
  elif(loss_name == "cross_entropy"):
    error=CrossEntropy(y, y_hat)
    #error = -np.sum(np.sum(y_t*np.log(y_hat)))

  #For L2 Regularization
  regularized_error = 0.0
  for i in range(len(layer_sizes)-1, 0, -1):
    regularized_error += (np.sum(parameters["w"+str(i)]))**2
  regularized_error = error + ((lambd/(2*len(y)))*(regularized_error))


  return regularized_error


def calc_accuracy(res, y_t):
    
    '''This function is used to calculate the accuracy of the given prediction

    Input is the true labels, and the predicted labels. Here, both true and
    predicted labels are in the probability distribution format.

    Output is the single float value that denotes the accuracy of the prediction'''

    acc=0.0
    
    for x in range(len(res)):
      if(res[x].argmax()==y_t[x].argmax()):
        acc+=1
    acc=acc/len(y_t)
    return(acc*100)


def calc_test_accuracy(y_pred, y_t):

  '''This function is used to calculate the accuracy of the given prediction

    Input is the true labels, and the predicted labels.

    Output is the single float value that denotes the accuracy of the prediction'''

  acc=0.0

  for i in range(len(y_pred)):
    if(y_pred[i]==y_t[i]):
      acc+=1
  acc=acc/len(y_t)
  return(acc*100)