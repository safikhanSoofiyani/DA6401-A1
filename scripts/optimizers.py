import wandb
import copy
import numpy as np
from tqdm import tqdm

from scripts.model import predict, grad_calculate_batchwise
from scripts.loss_functions import loss_calc, calc_accuracy


def mini_batch_gd(X_train, y_train, X_val, y_val, eta, max_epochs, layers, mini_batch_size, lambd, loss_function, activation, parameters,wandb_log=False):
  #parameters = nn_init(layers, 'random')

  ''' This function is essentially used to start the epochs, collect the gradients,
  update the parameters according to the vanilla mini batch gradient descent algorithm.
  It also keeps track of the train and validation accuracies for each epoch. Also log
  these value into wandb. 

  wandb_log is a flag. If true, the results are logged onto wandb workspace

  Input is training data, true labels for train data, validation data, labels
  (used from global context), eta i.e., learning rate, maximum epochs, overall
  architecture of the network, mini batch size, lambda for L2 Regularization, 
  activation funcion, parameters of the network.

  Output is the updated paramters (after training), training error and validation
  errors lists'''

  
  # Declaring an empty dicitonary for gradients
  grads={}
  
  # Declaring various lists to store the loss and accuracies
  train_errors_list = []
  val_errors_list = []
  train_acc_list = []
  val_acc_list = []

  # iterate till max epochs
  for t in tqdm(range(max_epochs)):


    # iterate over all batches
    for i in range(0, len(X_train), mini_batch_size):

      grads.clear()

      # Divide the data into batches and get the current batch
      X = X_train[i:i + mini_batch_size]
      Y = y_train[i:i + mini_batch_size]

      # Collect the gradients using the current batch of points
      grads = grad_calculate_batchwise(X,Y,parameters,activation,layers,loss_function)
      
    
      #Updating the parameters once every one batch
      for j in range(len(layers)-1,0,-1):
        # Here we have included the L2 Regularization
        parameters["w"+str(j)] = (1-((eta*lambd)/mini_batch_size))*parameters["w"+str(j)] - (eta * grads["dw"+str(j)])
        parameters["b"+str(j)] = parameters["b"+str(j)] - (eta * grads["db"+str(j)])

    #Calculating train loss and accuracies
    res = predict(X_train,y_train,parameters, activation, layers)
    train_err = loss_calc(loss_function, y_train, res, lambd, layers, parameters )
    train_acc = calc_accuracy(res, y_train)
    train_errors_list.append(train_err)
    train_acc_list.append(train_acc)

    #Calculating validation loss
    res = predict(X_val, y_val, parameters, activation, layers)
    val_err = loss_calc(loss_function, y_val, res, lambd, layers, parameters )
    val_acc = calc_accuracy(res,y_val)
    val_errors_list.append(val_err)
    val_acc_list.append(val_acc)

    if(wandb_log==True):
      # Logging the values into wandb
      log_dict = {"Train_Accuracy": train_acc, "Validation_Accuracy": val_acc, \
                  "Train_Loss": train_err, "Validation_loss": val_err, "epoch": t}
                  
      wandb.log(log_dict)
    else:
      print("Epoch: ", t, "Train Loss: ", train_err, "Validation Loss: ", val_err)
      print("Train Accuracy: ", train_acc, "Validation Accuracy: ", val_acc)
      print("**************")

  return parameters, train_acc_list, val_acc_list


def momentum_gd(X_train, y_train, X_val, y_val, eta, max_epochs, layers, mini_batch_size, lambd, loss_function, activation, parameters,wandb_log=False ):
  #parameters = nn_init(layers, 'random')

  ''' This function is essentially used to start the epochs, collect the gradients,
  update the parameters according to the momentum mini batch gradient descent algorithm.
  It also keeps track of the train and validation accuracies for each epoch. Also log
  these value into wandb. 

  wandb_log is a flag. If true, the results are logged onto wandb workspace

  Input is training data, true labels for train data, validation data, labels
  (used from global context), eta i.e., learning rate, maximum epochs, overall
  architecture of the network, mini batch size, lambda for L2 Regularization, 
  activation funcion, parameters of the network.

  Output is the updated paramters (after training), training error and validation
  errors lists'''
  
  # Declaring an empty dicitonary for gradients and update history
  grads={}
  update_history = {}
  gamma = 0.9 #Not treating this as a hyperparameter

  # Declaring various lists to store the loss and accuracies
  train_errors_list = []
  val_errors_list = []
  train_acc_list = []
  val_acc_list = []

  # iterate till max epochs
  for t in tqdm(range(max_epochs)):


    # iterate over all batches
    for i in range(0, len(X_train), mini_batch_size):

      grads.clear()

      # Divide the data into batches and get the current batch
      X = X_train[i:i + mini_batch_size]
      Y = y_train[i:i + mini_batch_size]
      
      # Collect the gradients using the current batch of points
      grads=grad_calculate_batchwise(X,Y,parameters,activation,layers,loss_function)
      
      #Storing the update history for each parameter.
      if i == 0 :
        # If i is 0, then it is the first batch and the update history
        # will be equal to the current gradients itself
        for j in range(len(layers)-1, 0, -1):
          update_history["w"+str(j)] = eta*grads["dw"+str(j)]
          update_history["b"+str(j)] = eta*grads["db"+str(j)]
      else:
        # else we store the update history according to momentum based gd algorihtm
        for j in range(len(layers)-1, 0, -1):
          update_history["w"+str(j)] = (gamma*update_history["w"+str(j)]) + (eta*grads["dw"+str(j)])
          update_history["b"+str(j)] = (gamma*update_history["b"+str(j)]) + (eta*grads["db"+str(j)])

    
      #Updating the parameters once every one batch with the update_history
      for j in range(len(layers)-1,0,-1):
        parameters["w"+str(j)] = (1-((eta*lambd)/mini_batch_size))*parameters["w"+str(j)] - update_history["w"+str(j)]
        parameters["b"+str(j)] = parameters["b"+str(j)] - update_history["b"+str(j)]

    #Calculating train loss and accuracies
    res = predict(X_train,y_train,parameters, activation, layers)
    train_err = loss_calc(loss_function, y_train, res, lambd, layers, parameters )
    train_acc = calc_accuracy(res, y_train)
    train_errors_list.append(train_err)
    train_acc_list.append(train_acc)

    #Calculating validation loss
    res = predict(X_val, y_val, parameters, activation, layers)
    val_err = loss_calc(loss_function, y_val, res, lambd, layers, parameters )
    val_acc = calc_accuracy(res,y_val)
    val_errors_list.append(val_err)
    val_acc_list.append(val_acc)

    if(wandb_log==True):
      # Logging the values into wandb
      log_dict = {"Train_Accuracy": train_acc, "Validation_Accuracy": val_acc, \
                  "Train_Loss": train_err, "Validation_loss": val_err, "epoch": t}
                  
      wandb.log(log_dict)

  return parameters, train_acc_list, val_acc_list


def nesterov_gd(X_train, y_train, X_val, y_val, eta, max_epochs, layers, mini_batch_size, lambd, loss_function, activation, parameters, wandb_log=False ):
 
  ''' This function is essentially used to start the epochs, collect the gradients,
  update the parameters according to the nesterov accelerated mini batch gradient descent algorithm.
  It also keeps track of the train and validation accuracies for each epoch. Also log
  these value into wandb. 

  wandb_log is a flag. If true, the results are logged onto wandb workspace

  Input is training data, true labels for train data, validation data, labels
  (used from global context), eta i.e., learning rate, maximum epochs, overall
  architecture of the network, mini batch size, lambda for L2 Regularization, 
  activation funcion, parameters of the network.

  Output is the updated paramters (after training), training error and validation
  errors lists'''

  # Declaring an empty dicitonary for gradients and update history and lookahead
  grads={}
  update_history = {}
  param_lookahead = {}
  gamma = 0.9 #not treating this as a hyperparameter.

  # Declaring various lists to store the loss and accuracies
  train_errors_list = []
  val_errors_list = []
  train_acc_list = []
  val_acc_list = []

  #iterate till max epochs
  for t in tqdm(range(max_epochs)):


    #iterate over all batches
    for i in range(0, len(X_train), mini_batch_size):

      grads.clear()

      #If it is the first batch, we still dont have the previous history.
      #So, lookahead will be same as the current parameters
      if i==0:
        param_lookahead = copy.deepcopy(parameters)
      
      #If its not the first batch then we calculate lookahead according to
      #the formula.
      else:
        for j in range(len(layers)-1, 0, -1):
          param_lookahead['w'+str(j)] = parameters['w'+str(j)] + (gamma*update_history["w"+str(j)])
                                                                  

      # Divide the data into batches and get the current batch
      X = X_train[i:i + mini_batch_size]
      Y = y_train[i:i + mini_batch_size]
      
      # Collect the gradients using the current batch of points
      grads=grad_calculate_batchwise(X,Y,param_lookahead,activation,layers,loss_function)
      
      # Storing the update history for each parameter.
      if i == 0 :
        # If its the first batch, we dont have any update history yet. So, it will
        # be same as the eta*gradients
        for j in range(len(layers)-1, 0, -1):
          update_history["w"+str(j)] = eta*grads["dw"+str(j)]
          update_history["b"+str(j)] = eta*grads["db"+str(j)]
      
      # If its not the first batch, we cumulate the update history as per the 
      # formula.
      else:
        for j in range(len(layers)-1, 0, -1):
          update_history["w"+str(j)] = (gamma*update_history["w"+str(j)]) + (eta*grads["dw"+str(j)])
          update_history["b"+str(j)] = (gamma*update_history["b"+str(j)]) + (eta*grads["db"+str(j)])

    
      #Updating the parameters once every one batch with the update_history
      for j in range(len(layers)-1,0,-1):
        parameters["w"+str(j)] = (1-((eta*lambd)/mini_batch_size))*parameters["w"+str(j)] - update_history["w"+str(j)]
        parameters["b"+str(j)] = parameters["b"+str(j)] - update_history["b"+str(j)]

    #Calculating train loss and accuracies
    res = predict(X_train,y_train,parameters, activation, layers)
    train_err = loss_calc(loss_function, y_train, res, lambd, layers, parameters )
    train_acc = calc_accuracy(res, y_train)
    train_errors_list.append(train_err)
    train_acc_list.append(train_acc)

    #Calculating validation loss
    res = predict(X_val, y_val, parameters, activation, layers)
    val_err = loss_calc(loss_function, y_val, res, lambd, layers, parameters )
    val_acc = calc_accuracy(res,y_val)
    val_errors_list.append(val_err)
    val_acc_list.append(val_acc)

    if(wandb_log==True):  
      # Logging the values into wandb
      log_dict = {"Train_Accuracy": train_acc, "Validation_Accuracy": val_acc, \
                  "Train_Loss": train_err, "Validation_loss": val_err, "epoch": t}
                  
      wandb.log(log_dict)

  return parameters, train_acc_list, val_acc_list



def rmsprop(X_train, y_train, X_val, y_val, eta, max_epochs, layers, mini_batch_size, lambd, loss_function, activation, parameters,wandb_log=False ):
    
  ''' This function is essentially used to start the epochs, collect the gradients,
  update the parameters according to the RmsProp mini batch gradient descent algorithm.
  It also keeps track of the train and validation accuracies for each epoch. Also log
  these value into wandb. 

  wandb_log is a flag. If true, the results are logged onto wandb workspace

  Input is training data, true labels for train data, validation data, labels
  (used from global context), eta i.e., learning rate, maximum epochs, overall
  architecture of the network, mini batch size, lambda for L2 Regularization, 
  activation funcion, parameters of the network.

  Output is the updated paramters (after training), training error and validation
  errors lists'''
  
  # Declaring an empty dicitonary for gradients and update history 
  grads={}
  update_history = {}
  v={}

  # Declaring various lists to store the loss and accuracies
  train_errors_list = []
  val_errors_list = []
  train_acc_list = []
  val_acc_list = []
  
  # Initializing update_history with zeros
  for i in range(len(layers)-1,0,-1):
    update_history["w"+str(i)]=np.zeros((layers[i],layers[i-1]))
    update_history["b"+str(i)]=np.zeros((layers[i],1))
  # Initializing v with zeros
  for i in range(len(layers)-1,0,-1):
    v["w"+str(i)]=np.zeros((layers[i],layers[i-1]))
    v["b"+str(i)]=np.zeros((layers[i],1))
  
  beta = 0.9 
  epsilon=1e-8

  #iterate till max epochs
  for t in tqdm(range(max_epochs)):
   
    #iterate over all batches
    for i in range(0, len(X_train), mini_batch_size):
      grads.clear()

      # Divide the data into batches and get the current batch
      X = X_train[i:i + mini_batch_size]
      Y = y_train[i:i + mini_batch_size]
      
      # Collect the gradients using the current batch of points
      grads=grad_calculate_batchwise(X,Y,parameters,activation,layers,loss_function)
        
      # Updating the values if v and the update history using the computed gradients
      for iq in range(len(layers)-1,0,-1):
        v["w"+str(iq)]=beta*v["w"+str(iq)]+(1-beta)*grads["dw"+str(iq)]**2
        v["b"+str(iq)]=beta*v["b"+str(iq)]+(1-beta)*grads["db"+str(iq)]**2
          
        update_history["w"+str(iq)]=eta*np.multiply(np.reciprocal(np.sqrt(v["w"+str(iq)]+epsilon)),grads["dw"+str(iq)])
        update_history["b"+str(iq)]=eta*np.multiply(np.reciprocal(np.sqrt(v["b"+str(iq)]+epsilon)),grads["db"+str(iq)])

      #Updating the parameters once every one batch with the update_history
      for j in range(len(layers)-1,0,-1):
        parameters["w"+str(j)] = (1-((eta*lambd)/mini_batch_size))*parameters["w"+str(j)] - update_history["w"+str(j)]
        parameters["b"+str(j)] = parameters["b"+str(j)] - update_history["b"+str(j)]

    #Calculating train loss and accuracies
    res = predict(X_train,y_train,parameters, activation, layers)
    train_err = loss_calc(loss_function, y_train, res, lambd, layers, parameters )
    train_acc = calc_accuracy(res, y_train)
    train_errors_list.append(train_err)
    train_acc_list.append(train_acc)

    #Calculating validation loss
    res = predict(X_val, y_val, parameters, activation, layers)
    val_err = loss_calc(loss_function, y_val, res, lambd, layers, parameters )
    val_acc = calc_accuracy(res,y_val)
    val_errors_list.append(val_err)
    val_acc_list.append(val_acc)

    if(wandb_log==True):
      # Logging the values into wandb
      log_dict = {"Train_Accuracy": train_acc, "Validation_Accuracy": val_acc, \
                  "Train_Loss": train_err, "Validation_loss": val_err, "epoch": t}
                  
      wandb.log(log_dict)

  return parameters, train_acc_list, val_acc_list




def adam(X_train, y_train, X_val, y_val, eta, max_epochs, layers, mini_batch_size, lambd, loss_function, activation, parameters,wandb_log=False ):
    
  ''' This function is essentially used to start the epochs, collect the gradients,
  update the parameters according to the Adam mini batch gradient descent algorithm.
  It also keeps track of the train and validation accuracies for each epoch. Also log
  these value into wandb. 

  wandb_log is a flag. If true, the results are logged onto wandb workspace

  Input is training data, true labels for train data, validation data, labels
  (used from global context), eta i.e., learning rate, maximum epochs, overall
  architecture of the network, mini batch size, lambda for L2 Regularization, 
  activation funcion, parameters of the network.

  Output is the updated paramters (after training), training error and validation
  errors lists'''
  
  # Declaring an empty dicitonary for gradients and update history 
  grads={}
  update_history = {}
  v={}
  m={}

  # Declaring various lists to store the loss and accuracies
  train_errors_list = []
  val_errors_list = []
  train_acc_list = []
  val_acc_list = []

  # Initializing update_history to zeros
  for i in range(len(layers)-1,0,-1):
    update_history["w"+str(i)]=np.zeros((layers[i],layers[i-1]))
    update_history["b"+str(i)]=np.zeros((layers[i],1))
  # Initializing m to zeros
  for i in range(len(layers)-1,0,-1):
    m["w"+str(i)]=np.zeros((layers[i],layers[i-1]))
    m["b"+str(i)]=np.zeros((layers[i],1))
  # Initializing v to zeros
  for i in range(len(layers)-1,0,-1):
    v["w"+str(i)]=np.zeros((layers[i],layers[i-1]))
    v["b"+str(i)]=np.zeros((layers[i],1))
  
  beta1 = 0.9 
  beta2=0.999
  epsilon=1e-8

  #iterate till max epochs
  step=0
  for t in tqdm(range(max_epochs)):


    #iterate over all batches
    for i in range(0, len(X_train), mini_batch_size):
      step+=1
      grads.clear()

      # Divide the data into batches and get the current batch
      X = X_train[i:i + mini_batch_size]
      Y = y_train[i:i + mini_batch_size]
      
      # Collect the gradients using the current batch of points
      grads=grad_calculate_batchwise(X,Y,parameters,activation,layers,loss_function)
      
      # Updating the values of v,m and the update history using the computed gradients
      for iq in range(len(layers)-1,0,-1):
          m["w"+str(iq)]=beta1*m["w"+str(iq)]+(1-beta1)*grads["dw"+str(iq)]
          m["b"+str(iq)]=beta1*m["b"+str(iq)]+(1-beta1)*grads["db"+str(iq)]
          
          v["w"+str(iq)]=beta2*v["w"+str(iq)]+(1-beta2)*(grads["dw"+str(iq)])**2
          v["b"+str(iq)]=beta2*v["b"+str(iq)]+(1-beta2)*(grads["db"+str(iq)])**2

          # Bias Correction:
          # calculating mt_hat and vt_hat for weights and biases 
          mw_hat=m["w"+str(iq)]/(1-np.power(beta1,step))
          mb_hat=m["b"+str(iq)]/(1-np.power(beta1,step))

          vw_hat=v["w"+str(iq)]/(1-np.power(beta2,step))
          vb_hat=v["b"+str(iq)]/(1-np.power(beta2,step))
          
          update_history["w"+str(iq)]=eta*np.multiply(np.reciprocal(np.sqrt(vw_hat+epsilon)),mw_hat)
          update_history["b"+str(iq)]=eta*np.multiply(np.reciprocal(np.sqrt(vb_hat+epsilon)),mb_hat)

      #Updating the parameters once every one batch with the update_history
      for j in range(len(layers)-1,0,-1):
          parameters["w"+str(j)] = (1-((eta*lambd)/mini_batch_size))*parameters["w"+str(j)] - update_history["w"+str(j)]
          parameters["b"+str(j)] = parameters["b"+str(j)] - update_history["b"+str(j)]

    #Calculating train loss and accuracies
    res = predict(X_train,y_train,parameters, activation, layers)
    train_err = loss_calc(loss_function, y_train, res, lambd, layers, parameters )
    train_acc = calc_accuracy(res, y_train)
    train_errors_list.append(train_err)
    train_acc_list.append(train_acc)

    #Calculating validation loss
    res = predict(X_val, y_val, parameters, activation, layers)
    val_err = loss_calc(loss_function, y_val, res, lambd, layers, parameters )
    val_acc = calc_accuracy(res,y_val)
    val_errors_list.append(val_err)
    val_acc_list.append(val_acc)

    if(wandb_log==True):
      # Logging the values into wandb
      log_dict = {"Train_Accuracy": train_acc, "Validation_Accuracy": val_acc, \
                  "Train_Loss": train_err, "Validation_loss": val_err, "epoch": t}
                  
      wandb.log(log_dict)

  return parameters, train_acc_list, val_acc_list




def nadam(X_train, y_train, X_val, y_val, eta, max_epochs, layers, mini_batch_size, lambd, loss_function, activation, parameters,wandb_log=False ):
    
  ''' This function is essentially used to start the epochs, collect the gradients,
  update the parameters according to the NAdam mini batch gradient descent algorithm.
  It also keeps track of the train and validation accuracies for each epoch. Also log
  these value into wandb. 

  wandb_log is a flag. If true, the results are logged onto wandb workspace

  Input is training data, true labels for train data, validation data, labels
  (used from global context), eta i.e., learning rate, maximum epochs, overall
  architecture of the network, mini batch size, lambda for L2 Regularization, 
  activation funcion, parameters of the network.

  Output is the updated paramters (after training), training error and validation
  errors lists'''

  # Declaring an empty dicitonary for gradients and update history 
  grads={}
  update_history = {}
  v={}
  m={}

  # Declaring various lists to store the loss and accuracies
  train_errors_list = []
  val_errors_list = []
  train_acc_list = []
  val_acc_list = []
  


  # Initializing update_history to zeros
  for i in range(len(layers)-1,0,-1):
    update_history["w"+str(i)]=np.zeros((layers[i],layers[i-1]))
    update_history["b"+str(i)]=np.zeros((layers[i],1))
  # Initializing m to zeros
  for i in range(len(layers)-1,0,-1):
    m["w"+str(i)]=np.zeros((layers[i],layers[i-1]))
    m["b"+str(i)]=np.zeros((layers[i],1))
  # Initializing v to zeros
  for i in range(len(layers)-1,0,-1):
    v["w"+str(i)]=np.zeros((layers[i],layers[i-1]))
    v["b"+str(i)]=np.zeros((layers[i],1))
  
  beta1 = 0.9 
  beta2=0.999
  epsilon=1e-8

  #iterate till max epochs
  step = 0
  for t in tqdm(range(max_epochs)):


    #iterate over all batches
    for i in range(0, len(X_train), mini_batch_size):
      step += 1

      grads.clear()

      # Divide the data into batches and get the current batch
      X = X_train[i:i + mini_batch_size]
      Y = y_train[i:i + mini_batch_size]
      
      # Collect the gradients using the current batch of points
      grads=grad_calculate_batchwise(X,Y,parameters,activation,layers,loss_function)
 
      # Updating the values of v,m and the update history using the computed gradients
      for iq in range(len(layers)-1,0,-1):
          m["w"+str(iq)]=beta1*m["w"+str(iq)]+(1-beta1)*grads["dw"+str(iq)]
          m["b"+str(iq)]=beta1*m["b"+str(iq)]+(1-beta1)*grads["db"+str(iq)]
          
          v["w"+str(iq)]=beta2*v["w"+str(iq)]+(1-beta2)*(grads["dw"+str(iq)])**2
          v["b"+str(iq)]=beta2*v["b"+str(iq)]+(1-beta2)*(grads["db"+str(iq)])**2

          # Bias Correction:
          # calculating mt_hat and vt_hat for weights and biases 
          mw_hat=m["w"+str(iq)]/(1-np.power(beta1,step))
          mb_hat=m["b"+str(iq)]/(1-np.power(beta1,step))

          vw_hat=v["w"+str(iq)]/(1-np.power(beta2,step))
          vb_hat=v["b"+str(iq)]/(1-np.power(beta2,step))
          
          update_history["w"+str(iq)]=eta*np.multiply(np.reciprocal(np.sqrt(vw_hat+epsilon)),(beta1*mw_hat+(1-beta1)*grads["dw"+str(iq)]))*(1/(1-np.power(beta1,t+1)))
          update_history["b"+str(iq)]=eta*np.multiply(np.reciprocal(np.sqrt(vb_hat+epsilon)),(beta1*mb_hat+(1-beta1)*grads["db"+str(iq)]))*(1/(1-np.power(beta1,t+1)))

      #Updating the parameters once every one batch with the update_history
      for j in range(len(layers)-1,0,-1):
          parameters["w"+str(j)] = (1-((eta*lambd)/mini_batch_size))*parameters["w"+str(j)] - update_history["w"+str(j)]
          parameters["b"+str(j)] = parameters["b"+str(j)] - update_history["b"+str(j)]

    #Calculating train loss and accuracies
    res = predict(X_train,y_train,parameters, activation, layers)
    train_err = loss_calc(loss_function, y_train, res, lambd, layers, parameters )
    train_acc = calc_accuracy(res, y_train)
    train_errors_list.append(train_err)
    train_acc_list.append(train_acc)

    #Calculating validation loss
    res = predict(X_val, y_val, parameters, activation, layers)
    val_err = loss_calc(loss_function, y_val, res, lambd, layers, parameters )
    val_acc = calc_accuracy(res,y_val)
    val_errors_list.append(val_err)
    val_acc_list.append(val_acc)

    if(wandb_log==True):
      # Logging the values into wandb
      log_dict = {"Train_Accuracy": train_acc, "Validation_Accuracy": val_acc, \
                  "Train_Loss": train_err, "Validation_loss": val_err, "epoch": t}
                  
      wandb.log(log_dict)

  


  return parameters, train_acc_list, val_acc_list






def new_optimization_function_name(X_train, y_train, X_val, y_val, eta, max_epochs, layers, mini_batch_size, lambd, loss_function, activation, parameters,wandb_log=False ):
    
  '''This is a template for defining and integrating new optimization
  functions in our code'''

  # Declaring an empty dicitonary for gradients and update history 
  grads={}


  
  '''Declare the dictionaries and other data structures as per
  the requirement of the optimization function '''




  # Declaring various lists to store the loss and accuracies
  train_errors_list = []
  val_errors_list = []
  train_acc_list = []
  val_acc_list = []
  



  '''Initialize the data structures to appropriate intial values as per
  the requirement of the optimization function'''
  




  #iterate till max epochs
  for t in tqdm(range(max_epochs)):


    #iterate over all batches
    for i in range(0, len(X_train), mini_batch_size):

      grads.clear()

      # Divide the data into batches and get the current batch
      X = X_train[i:i + mini_batch_size]
      Y = y_train[i:i + mini_batch_size]
      
      # Collect the gradients using the current batch of points
      grads=grad_calculate_batchwise(X,Y,parameters,activation,layers,loss_function)
 

      #Updating the parameters once every one batch with the update_history
      for j in range(len(layers)-1,0,-1):


          '''write the paramter update rule for the parameters of the network
          as per the requirements of the optimization function'''




    #Calculating train loss and accuracies
    res = predict(X_train,y_train,parameters, activation, layers)
    train_err = loss_calc(loss_function, y_train, res, lambd, layers, parameters )
    train_acc = calc_accuracy(res, y_train)
    train_errors_list.append(train_err)
    train_acc_list.append(train_acc)

    #Calculating validation loss
    res = predict(X_val, y_val, parameters, activation, layers)
    val_err = loss_calc(loss_function, y_val, res, lambd, layers, parameters )
    val_acc = calc_accuracy(res,y_val)
    val_errors_list.append(val_err)
    val_acc_list.append(val_acc)

    if(wandb_log==True):
      # Logging the values into wandb
      log_dict = {"Train_Accuracy": train_acc, "Validation_Accuracy": val_acc, \
                  "Train_Loss": train_err, "Validation_loss": val_err, "epoch": t}
                  
      wandb.log(log_dict)

  


  return parameters, train_acc_list, val_acc_list

'''After defining the optimization function, please add the function name in 
train function if you want to perform hyperparameter sweeps'''



