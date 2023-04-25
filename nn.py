import numpy as np
# import tensorflow as tf

# # Load the MNIST dataset
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# # Normalize the pixel values
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# # Reshape the input data to a flattened vector
# X_train = X_train.reshape(X_train.shape[0], -1)
# X_test = X_test.reshape(X_test.shape[0], -1)

# # Convert the target variable to one-hot encoded vectors
# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)

#sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

#softmax function for output
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

#cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


#neural net class
class NeuralNet:

    #intitialize weights to random values
    def __init__(self, dimensions):
        dimensions = np.array(dimensions)
        """
        each elements in weights holds the weights between each layer
        for example, if the dimensions = [2,3,2,1], then
        weights[0] would be a 2 x 3 array, weights[1] would be a 3 x 2 array
        and weights[2] would be a 2 x 1 array  
        """

        #initializing weights
        self.weights = []
        for x in (range(len(dimensions)-1)):
            weight = np.random.randn(dimensions[x], dimensions[x+1])
            self.weights.append(weight)
        
        #intializing biases
        self.biases= []
        for x in range(1,len(dimensions)):
            bias = np.zeros(1,dimensions[x])
            self.biases.append(bias)

    #forward calculations
    def forward(self, X_batch,y_batch ):

        #keeps track of all of the activations
        self.activations = []

        #input layer
        hidden_layer = sigmoid(np.dot(X_batch, self.weights[0]) + self.biases[0])
        self.activations.insert(0,hidden_layer)
        # hidden_layer.append(activations)

        for x in range(1,len(self.weights) - 1):
            hidden_layer = sigmoid(np.dot(hidden_layer, self.weights[x]) + self.biases[x])
            self.activations.insert(0,hidden_layer)

        #put through a softmax function instead of sigmoid to improve performance
        output_layer = softmax(np.dot(hidden_layer, self.weights[len(self.weights)-1]) + self.biases[len(self.biases)-1])

        #return the output layer
        return output_layer
    
    #back propagation
    def backProp(self, X_batch, prediction, y_batch, batch_size, learning_rate):
        
        #stores the gradients that will be used to alter the weights/biases
        gradients = []

        delta = (prediction - y_batch) / batch_size
        gradients.append(delta)

        #calculating the gradients for each layer of weights
        for x in range(0, len(self.activations)):
            error = delta.dot(self.weights[len(self.weights)-1-x].T)
            delta = error * sigmoid_derivative(self.activations[x])
            gradients.append(delta)
        
        #applying the gradients for each layer of weights
        for x in range(0, len(gradients) - 1):
            self.weights[len(self.weights)-1-x] = self.weights[len(self.weights)-1-x] - learning_rate * self.activations[x].T.dot(gradients[x])
            self.biases[len(self.biases)-1- x] = self.biases[len(self.biases)-1 - x] - learning_rate * np.sum(gradients[x], axis = 0)
        
        #appyling the gradients for the input layer of weights, seperate because it uses the X_batch
        self.weights[0] = self.weights[0] - learning_rate * X_batch.T.dot(gradients[len(gradients)-1])
        self.biases[0] = self.biases[0] - learning_rate * np.sum(gradients[len(gradients)-1], axis = 0)



# nn = NeuralNet([(28*28),100,100,10])
# batch_size = 1000


# #trains neural network
# count = 0
# for x in range(10000):
#     print("Training", count)
#     count+=1
#     indices = np.random.randint(len(X_train), size=batch_size)
#     X_batch = X_train[indices]
#     y_batch = y_train[indices]

#     result = nn.forward(X_batch,y_batch )
#     nn.backProp(X_batch,result, y_batch,batch_size, 0.1)

# #runs neural network on test batches
# for x in range(1000):
#     indices = np.random.randint(len(X_test), size=batch_size)
#     X_batch = X_test[indices]
#     y_batch = y_test[indices]

#     result = nn.forward(X_batch,y_batch )
#     loss = cross_entropy_loss(y_batch, result)
#     #print(loss)
#     accuracy = np.mean(np.argmax(result, axis=1) == np.argmax(y_batch, axis=1))
#     print(accuracy)