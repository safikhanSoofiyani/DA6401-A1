import argparse
from scripts.model import nn_init, predict
from scripts.data_prep import prepare_data
from scripts.optimizers import *
from scripts.loss_functions import loss_calc, calc_accuracy

def fit(X_train, 
        y_train, 
        X_val,
        y_val,
        layer_sizes, wandb_log, 
        learning_rate = 0.0001, 
        initialization_type = "random", 
        activation_function = "sigmoid", 
        loss_function = "cross_entropy", 
        mini_batch_Size = 32, 
        max_epochs = 5, 
        lambd = 0,
        optimization_function = mini_batch_gd
        ): 

  ''' This function is actually used to run the training process. It first
  initializes the network and calls the respective optimization function
  to train the model. It also prints the train and validation losses for each
  training epoch 
  
  The input is the training data, true labels, overall architecture of the network,
  learning rate, initialization type, activation function, loss function, mini 
  batch size, maximum epochs, lambda for regularization, and the optimization 
  function to be used 
  
  The output is the trained models parameters'''

  parameters = nn_init(init_type = initialization_type, layer_sizes = layer_sizes)
  print("Training the model")
  parameters, train_errors_list, val_errors_list = optimization_function(X_train, y_train, X_val, y_val, learning_rate, max_epochs, layer_sizes, mini_batch_Size, lambd, loss_function, activation_function, parameters,wandb_log)
  
  print("Training Accuracy:",train_errors_list[-1])
  print("Validation Accuracy:",val_errors_list[-1])
  

  return parameters


    



def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network model")
    parser.add_argument("--wandb_project", type=str, default="da6401_assignment", help="Weights and Biases project name")
    parser.add_argument("--wandb_entity", type=str, default="safikhan", help="Weights and Biases entity name")
    parser.add_argument("--dataset", type=str, default="fashion_mnist", help="Dataset to be used for training")
    parser.add_argument("--num_hidden_layers", type=int, default=5, help="Number of hidden layers in the model")
    parser.add_argument("--num_neurons", type=int, default=128, help="Number of neurons in each hidden layer")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the model")
    parser.add_argument("--init_type", type=str, default="xavier", help="Initialization type for the model")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function for the model")
    parser.add_argument("--loss", type=str, default="cross_entropy", help="Loss function for the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for the model")
    parser.add_argument("--lambd", type=float, default=0.0005, help="Regularization parameter for the model")
    parser.add_argument("--optimizer", type=str, default="nadam", help="Optimizer function for the model")
    parser.add_argument("--test", type=str, default="False", help="Test the model")
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    X_train, X_val, X_test, y_train, y_val, y_test, labels = prepare_data()
    
    #Calling the respective hyperparameters
    if args.optimizer == "adam":
        optimization_function = adam
    elif args.optimizer == "nadam":
        optimization_function = nadam
    elif args.optimizer == "mini_batch_gd":
        optimization_function = mini_batch_gd
    elif args.optimizer == "momentum_gd":
        optimization_function = momentum_gd
    elif args.optimizer == "nesterov_gd":
        optimization_function = nesterov_gd
    elif args.optimizer == "rmsprop":
        optimization_function = rmsprop
    else:
        print("Wrong optimization function")
        exit()

    
    layer_sizes = [784]
    for i in range(args.num_hidden_layers):
        layer_sizes = layer_sizes + [args.num_neurons]
    layer_sizes  = layer_sizes + [10]
    wandb_log = True
    parameters = fit(X_train, y_train, X_val, y_val, 
                     layer_sizes, False, args.learning_rate,
                     args.init_type, args.activation, 
                     args.loss, args.batch_size, 
                     args.epochs, args.lambd, optimization_function)
    
    
    if args.test == "True":
        res = predict(X_test,y_test,parameters, args.activation, args.layers)
        test_err = loss_calc(args.loss, y_train, res, args.lambd, layer_sizes, parameters )
        test_acc = calc_accuracy(res, y_train)
        print("Test Accuracy:", test_acc)
    
    
    
    
    
if __name__ == "__main__":
    main()
