import wandb
import argparse
from scripts.data_prep import prepare_data
from scripts.optimizers import *
from scripts.model import nn_init


def sweeper(entity_name, project_name, sweep_id=None):
  #Declaring the dictionary of all choices for the hyperparameters.
  hyperparameters = {
      "learning_rate":{
        'values': [0.001, 0.0001]
      },

      "number_hidden_layers": {
          'values' : [3, 4, 5]
      },

      "number_neurons": {
        'values': [32, 64, 128]
      },

      "initialization_type": {
          'values' : ["xavier", "random"]
      },

      "activation_function": {
          'values': ["sigmoid", "tanh", "relu"]
      },

      "mini_batch_size": {
          'values': [16,32,64,128]
      },

      "max_epochs": {
          'values': [5, 10, 20]
      },

      "lambd": {
          'values': [0, 0.0005, 0.5]
      },

      "optimization_function": {
          'values': ["mini_batch_gd", "momentum_gd", "nesterov_gd", "rmsprop", "adam", "nadam"]
      },
      
      "loss_function": {
          'values': ["cross_entropy", "squared_loss"]
      }

  }


  #Using bayes method for hyperparameter sweeps to curb the unnecessary configurations
  sweep_config = {
      'method' : 'random',
      'metric' :{
          'name': 'Validation_Accuracy',
          'goal': 'maximize'
      },
      'parameters': hyperparameters
  }

  if sweep_id is None:
    sweep_id = wandb.sweep(sweep_config, entity=entity_name, project=project_name)
    print("Initialized sweep with ID:", sweep_id)
    
  wandb.agent(sweep_id, train, entity=entity_name, project=project_name, count = 100)
  
def fit(X_train, 
        y_train, 
        X_val,
        y_val,
        layer_sizes,wandb_log, 
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
  parameters, train_errors_list, val_errors_list = optimization_function(X_train, y_train, X_val, y_val, 
                                                                         learning_rate, max_epochs, layer_sizes, 
                                                                         mini_batch_Size, lambd, loss_function, 
                                                                         activation_function, parameters,wandb_log
                                                                         )
  
  print("Training Accuracy:",train_errors_list[-1])
  print("Validation Accuracy:",val_errors_list[-1])

  return parameters



def train():

  '''This function is used to exploit the wandb hyperparameter sweep 
  function to get the best hyperparameters.

  It takes in no inputs and gives no outputs.
  
  Instead it logs everything into the wandb workspace'''
  
  X_train, X_val, X_test, y_train, y_val, y_test, labels = prepare_data()

  #Declaring the dictionary with the default hyperparameters
  config_defaults = {
      'number_hidden_layers': 2,
      'number_neurons': 32,
      'learning_rate': 0.001,
      'initialization_type': "xavier",
      'activation_function':'sigmoid',
      'mini_batch_size' : 64,
      'max_epochs': 5,
      'lambd': 0,
      'optimization_function': "adam"
      
  }

  # Initializing the wandb run
  wandb.init(config=config_defaults)
  config = wandb.config


  # Constructing the layer_sizes i.e., the architecture of our neural network
  layer_sizes = [784]
  for i in range(config.number_hidden_layers):
    layer_sizes = layer_sizes + [config.number_neurons]
  layer_sizes  = layer_sizes + [10]

  #Collecting all the hyperparameters from the wandb run
  learning_rate = config.learning_rate
  initialization_type = config.initialization_type
  activation_function = config.activation_function
  loss_function = config.loss_function
  mini_batch_size = config.mini_batch_size
  max_epochs = config.max_epochs
  lambd = config.lambd
  opt_fun = config.optimization_function

  #Calling the respective hyperparameters
  if opt_fun == "adam":
    optimization_function = adam
  elif opt_fun == "nadam":
    optimization_function = nadam
  elif opt_fun == "mini_batch_gd":
    optimization_function = mini_batch_gd
  elif opt_fun == "momentum_gd":
    optimization_function = momentum_gd
  elif opt_fun == "nesterov_gd":
    optimization_function = nesterov_gd
  elif opt_fun == "rmsprop":
    optimization_function = rmsprop
  else:
    print("Wrong optimization function")
    exit()


  #Forming meaningful run name using the hyperparameters
  name_run = str(learning_rate) + "_" + initialization_type[0] + "_" + \
  activation_function[0] + "_" + str(mini_batch_size) + "_" + str(max_epochs) + \
  "_" + str(lambd) + "_" + opt_fun[:4]

  wandb.run.name = name_run
  wandb_log=True
  #Calling the fit function to train the neural network with the current hyperparameters
  parameters = fit(X_train, y_train, X_val, y_val,
                   layer_sizes, wandb_log, learning_rate,
                   initialization_type, activation_function, 
                   loss_function, mini_batch_size, max_epochs, 
                   lambd, optimization_function)

  
  wandb.run.save()
  wandb.run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Sweeper")
    parser.add_argument("--entity", type=str, default="your_wandb_entity_name")
    parser.add_argument("--project", type=str, default="your_wandb_project_name")
    parser.add_argument("--sweep_id", type=str, default=None, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sweeper(args.entity, args.project, args.sweep_id)
    
    
    
# import wandb
# import argparse
# import multiprocessing
# from scripts.data_prep import prepare_data
# from scripts.optimizers import *
# from scripts.model import nn_init

# def sweeper(entity_name, project_name, num_agents=4, sweep_id=None):
#     """
#     This function initializes a wandb sweep and runs multiple parallel agents.

#     Args:
#         entity_name (str): WandB entity name.
#         project_name (str): WandB project name.
#         num_agents (int): Number of parallel runs (default: 4).
#     """

#     # Declaring the dictionary of all choices for hyperparameters.
#     hyperparameters = {
#         "learning_rate": {'values': [0.001, 0.0001]},
#         "number_hidden_layers": {'values': [3, 4, 5]},
#         "number_neurons": {'values': [32, 64, 128]},
#         "initialization_type": {'values': ["xavier", "random"]},
#         "activation_function": {'values': ["sigmoid", "tanh", "relu"]},
#         "mini_batch_size": {'values': [16, 32, 64, 128]},
#         "max_epochs": {'values': [5, 10, 20]},
#         "lambd": {'values': [0, 0.0005, 0.5]},
#         "optimization_function": {
#             'values': ["mini_batch_gd", "momentum_gd", "nesterov_gd", "rmsprop", "adam", "nadam"]
#         }
#     }

#     # Using Bayesian method for hyperparameter tuning.
#     sweep_config = {
#         'method': 'bayes',
#         'metric': {'name': 'Validation_Accuracy', 'goal': 'maximize'},
#         'parameters': hyperparameters
#     }

#     if sweep_id is not None:
#         sweep_id = wandb.sweep(sweep_config, entity=entity_name, project=project_name, sweep_id=sweep_id)
#     else:
#         sweep_id = wandb.sweep(sweep_config, entity=entity_name, project=project_name)
#         print("Initialized sweep with ID:", sweep_id)

#     # Start multiple agents in parallel
#     print(("Starting {} agents...").format(num_agents))
#     processes = []
#     for _ in range(num_agents):
#         p = multiprocessing.Process(target=start_agent, args=(entity_name, project_name, sweep_id))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

# def start_agent(entity_name, project_name, sweep_id):
#     """Runs a single WandB agent to train the model."""
    
#     wandb.agent(sweep_id, train)


# def fit(X_train, y_train, X_val, y_val, layer_sizes, wandb_log,
#         learning_rate=0.0001, initialization_type="random",
#         activation_function="sigmoid", loss_function="cross_entropy",
#         mini_batch_Size=32, max_epochs=5, lambd=0,
#         optimization_function=mini_batch_gd):
#     """
#     Runs the training process.

#     Args:
#         - Training and validation data.
#         - Hyperparameters for training.
    
#     Returns:
#         - Trained model parameters.
#     """
    
#     parameters = nn_init(init_type=initialization_type, layer_sizes=layer_sizes)
#     parameters, train_errors_list, val_errors_list = optimization_function(
#         X_train, y_train, X_val, y_val, learning_rate, max_epochs, layer_sizes,
#         mini_batch_Size, lambd, loss_function, activation_function, parameters, wandb_log
#     )

#     print("Training Accuracy:", train_errors_list[-1])
#     print("Validation Accuracy:", val_errors_list[-1])
    
#     return parameters


# def train():
#     """Runs a single training instance as part of the WandB sweep."""
    
#     X_train, X_val, X_test, y_train, y_val, y_test, labels = prepare_data()

#     # Default hyperparameters
#     config_defaults = {
#         'number_hidden_layers': 2,
#         'number_neurons': 32,
#         'learning_rate': 0.001,
#         'initialization_type': "xavier",
#         'activation_function': 'sigmoid',
#         'mini_batch_size': 64,
#         'max_epochs': 5,
#         'lambd': 0,
#         'optimization_function': "adam"
#     }

#     # Initialize wandb
#     wandb.init(config=config_defaults)
#     config = wandb.config

#     # Construct the layer sizes for the neural network
#     layer_sizes = [784] + [config.number_neurons] * config.number_hidden_layers + [10]

#     # Collect hyperparameters from wandb
#     learning_rate = config.learning_rate
#     initialization_type = config.initialization_type
#     activation_function = config.activation_function
#     loss_function = "cross_entropy"
#     mini_batch_size = config.mini_batch_size
#     max_epochs = config.max_epochs
#     lambd = config.lambd
#     opt_fun = config.optimization_function

#     # Map optimization function
#     optimization_functions = {
#         "adam": adam, "nadam": nadam, "mini_batch_gd": mini_batch_gd,
#         "momentum_gd": momentum_gd, "nesterov_gd": nesterov_gd, "rmsprop": rmsprop
#     }
    
#     optimization_function = optimization_functions.get(opt_fun)
#     if optimization_function is None:
#         print(f"Invalid optimization function: {opt_fun}")
#         exit()

#     # Naming the wandb run
#     name_run = f"{learning_rate}_{initialization_type[0]}_{activation_function[0]}_" \
#                f"{mini_batch_size}_{max_epochs}_{lambd}_{opt_fun[:4]}"
#     print(f"Starting run: {name_run}")

#     wandb.run.name = name_run
#     wandb_log = True

#     # Train the model
#     parameters = fit(X_train, y_train, X_val, y_val, layer_sizes, wandb_log,
#                      learning_rate, initialization_type, activation_function,
#                      loss_function, mini_batch_size, max_epochs, lambd, optimization_function)

#     wandb.run.save()
#     wandb.run.finish()


# def parse_args():
#     """Parse command-line arguments."""
#     parser = argparse.ArgumentParser(description="Sweeper")
#     parser.add_argument("--entity", type=str, required=True, help="Your WandB entity name")
#     parser.add_argument("--project", type=str, required=True, help="Your WandB project name")
#     parser.add_argument("--num_agents", type=int, default=4, help="Number of parallel agents (default: 4)")
#     parser.add_argument("--sweep_id", type=str, default=None, help="WandB sweep ID")
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     sweeper(args.entity, args.project, args.num_agents, args.sweep_id)
