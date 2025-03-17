# Developing Backpropagation from scratch
-----------------------------------------------------------
In this work, we developed a feed-forward neural network from scratch. We have used gradient descent method and its variants as optimization algorithm with backpropogation to classify images from Fashion-MNIST dataset. We used "wandb.ai" to perform experiments for hyperparameter tuning. 
# Libraries and their application :
1. Numpy: Mathematical operations are performed by this library
2. Keras: This library is used to obtain the dataset.
3. Matplotlib and Seaborn: Sample images from each class and Confusion Matrix are plotted using these libraries respetively
4. sklearn: The dataset is split into Train-Test-Validation by this library
5. wandb: This library is used to log the metrics to wandb.ai.\

## WandB Report Link
https://wandb.ai/safikhan/DA6401_Assignment_1/reports/DA6401-Assignment-1--VmlldzoxMTcxNzgwOQ
# Installations:
The above mentioned libraries can be installed on local machine by using the following code snippet in the command prompt:
```python
pip install numpy
pip install keras
pip install matplotlib
pip install seaborn
pip install sklearn
pip install wandb
```

# Code Structure
```
code/
│── README.md             # README
│── requirements.txt      # List of dependencies (for pip)
│── scripts/                 
│   ├── activation.py      # Functions for all the activation functions
│   ├── data_prep.py       # Functions for data preparation and train-test split
│   ├── initialization.py  # Functions for different intializations
│   ├── loss_functions.py  # Functions for different loss functions
│   ├── model.py           # Functions defining the main neural network
│   ├── optimizers.py      # Functions for different optimizers
│── train.py               # The main training script
│── sweep.py               # The script for running sweeps
│── assignment_1.ipynb     # The script for plotting the data
```

# Training the Neural Network:
To train the neural network run the below command
  ```shell
python train.py --wandb_project "<your_wandb_project>" --wandb_entity "<your_wandb_entity>"
  ```
The script takes the following arguments:
  1. `wandb_project` the wandb project name for logging the runs.
  2. `wandb_entity` the wandb entity name for logging the runs.
  3. `dataset` either fashion_mnist or mnist.
  4. `num_hidden_layers` the number of hidden layers in the model.
  5. `num_neurons` stores  the number of neurons present in each layer, including the input and the output layers.
  7. `learning_rate` stores the learning rate of the gradient descent(and its variants) optimization functions
  8. `init_type` stores the weight initialization type, you can choose: `Xavier` or `random`
  9. `activation` stores the activation function that is applied to all the hidden layers
  10. `loss` stores the type of loss function, you can choose: `cross_entropy` or `squared error`
  11. `batch_size` stores the number of data points per batch.
  12. `epochs` stores the maximum number of epochs 
  13. `lambd` stores the regularization constant for weight decay
  14. `optimizer` stores the name of gradient descent algorithm

To run the sweeps, call the below script:
```shell
python sweep.py --entity "<your_wandb_entity>" --project "<your_wandb_project>"
```
   
# Addition of a new Optimization Function:
We have given a template for adding a optimization function on the similar lines of previous functions. 
<\br> The user need to add the following code snippets to form a new optimization function.
  1. Declare and Initialize dictionaries and other data structures as per the requirement of optimization function.
  2. New parameter update rule  for the network parameters.
 
 The new optimization function looks like this :
```python
new_optimization_function_name(X_train, y_train, eta, max_epochs, layers, mini_batch_size, lambd, loss_function, activation, parameters,wandb_log=False )
```
  1. `X_train` stores the list of flattened images of training dataset.
  2. `y_train` stores the list of labels for the images of training dataset in one-hot encoded format.
  3. `eta` stores the learning rate.
  4. `max_epochs` stores the maximum number of epochs.
  5. `layers` stores the number of neurons per each layer.
  6. `mini_batch_size` stores the number of data points per batch.
  7. `lambd` stores the regularization constant for weight decay.
  8. `loss_function` stores the type of loss function, you can choose: `cross_entropy` or `squared error`.
  9. `activation` stores the activation function that is applied to all the hidden layers.
  10. `parameters` stores the intial parameters (weights and biases).
  11. `wandb_log` stores the boolean variable which determines whether or not the data is logged into wandb.ai
  


# Wandb Functionality:

1. To `use wandb mode`, find your `API key` from your wandb account and paste it in the output box after you executed this code snippet :     
  ```python
!wandb login --relogin
# enter the entity and project name in these variables
entity_name="_entity_name_"
project_name="_project_name_"
  ```
2. You can `perform experiments` by running the sweeps, using this function:
```python
sweeper(entity_name,project_name)
```
3. You can `compare` the performance of two `loss functions` by using this function:
```python
loss_compare_sweeper(entity_name,project_name)
```
4. You can plot the `confusion matrix` for the test dataset by using this function, this returns predicted labels and true labels:
```python
y_pred,y_t=plot_confmat_wandb(entity_name,project_name)
``` 
--------------------------------------------------------------     
# Available options to customize the Neural Network:
  
<h4> 1) Loss functions
  
  ```python
  
  MSE()
  CrossEntropy()
  ```
<h4> 2) Optimization functions
  
  ```python
  
  mini_batch_gd()
  momentum_gd()
  nesterov_gd()
  rmsprop()
  adam()
  nadam()
  ```
<h4> 3) Weight Initializations
  
  ```python
  Xavier()
  Random()
  ```
<h4> 4) Activation Functions
  
  ```python
  sigmoid()
  tanh()
  relu()
  softmax()
  ```
