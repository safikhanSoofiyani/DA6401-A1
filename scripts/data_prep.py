from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
import copy
import numpy as np
#sklearn library is used only for train test validation split


def prepare_data():

  '''This function is used to load the data, define the class labels, performing
      the train-test-validation split, normalizing the data, flattening each data
      point, converting the class labels to one hot encoded vector.

      It return all the split data sets '''


  # Loading data from online source
  (train_x,train_y),(test_x,test_y)=fashion_mnist.load_data()

  # Defining labels for data
  num_classes = 10
  labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

  print("Number of data points in train data (initially) - ", len(train_x))
  print("Number of data points in test data (initially) - ", len(test_x))


  #performing the train-validation split
  train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=40)
  

  print("Shape of each image - 28x28" )
  image_shape=train_x.shape[1]*train_x.shape[2]
  print("shape of each image (1D) - ",image_shape)
  
  #storing the number of points in each set
  train_image_count=len(train_x)
  val_image_count = len(val_x)
  test_image_count=len(test_x)
  
  # Creating a matrix of image data 
  # each image is represented as a row by flattening the matrix: converting (60000,28,28) tensor to (60000,784) matrix
  X_train=np.zeros((train_image_count,image_shape))
  X_val=np.zeros((val_image_count,image_shape))
  X_test=np.zeros((test_image_count,image_shape))
  
  # converting the images into grayscale by normalizing
  for i in range(train_image_count):
    X_train[i]=(copy.deepcopy(train_x[i].flatten()))/255.0 
  for i in range(val_image_count):
    X_val[i]=(copy.deepcopy(val_x[i].flatten()))/255.0
  for i in range(test_image_count):
    X_test[i]=(copy.deepcopy(test_x[i].flatten()))/255.0
  


  #One hot encoding the label vectors to represent a probability distribution
  y_train = np.zeros((train_y.size, 10))
  y_train[np.arange(train_y.size), train_y] = 1

  y_val = np.zeros((val_y.size, 10))
  y_val[np.arange(val_y.size), val_y] = 1

  y_test = np.zeros((test_y.size, 10))
  y_test[np.arange(test_y.size), test_y] = 1

  

  #returning all the datasets along with the labels
  return X_train,X_val,X_test,y_train,y_val,y_test,labels
  