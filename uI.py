from PIL import Image
import numpy as np
import nn
import tensorflow as tf
import matplotlib.pyplot as plt
# # Open the image file
# img = Image.open('/home/cgambla/Downloads/archive(2)/testSample/img_2.jpg')
# img = Image.open('/home/cgambla/Downloads/autodraw 4_24_2023(1).png')
# # Resize the image to 28x28 pixels
# img = img.resize((28, 28))

# plt.imshow(img)
# plt.show()

# img2 = img.resize((28, 28))
# plt.imshow(img)
# plt.show()
# Convert the image to grayscale
# img = img.convert('L')

# # Convert the image to a numpy array
# arr = np.array(img)
# print(arr.shape)
import tensorflow as tf

# # Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the input data to a flattened vector
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Convert the target variable to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

net = nn.NeuralNet([(28*28),100,100,10])
batch_size = 128


#trains neural network
count = 0
min_loss = 0.01
epochs = 10000
for x in range(epochs):
    print("Training", count, end = "")
    count+=1
    indices = np.random.randint(len(X_train), size=batch_size)
    X_batch = X_train[indices]
    y_batch = y_train[indices]

    result = net.forward(X_batch,y_batch )
    loss = nn.cross_entropy_loss(y_batch, result)
    print(" Loss:", loss)
    if(abs(loss) < min_loss):
        break
    net.backProp(X_batch,result, y_batch,batch_size, 0.1)

#runs neural network on test batches
acc = []
for x in range(1000):
    indices = np.random.randint(len(X_test), size=batch_size)
    X_batch = X_test[indices]
    y_batch = y_test[indices]

    result = net.forward(X_batch,y_batch )
    # loss = nn.cross_entropy_loss(y_batch, result)
    #print(loss)
    accuracy = np.mean(np.argmax(result, axis=1) == np.argmax(y_batch, axis=1))
    acc.append(accuracy)
    # print(accuracy)
print(np.mean(acc))
# arr = arr.flatten()
# print()
# # mask = arr <= 26
# # arr[mask] = 0
# # # Set all values where the mask is True to 0
# # print(arr)

# arr = arr/ 255.0


# result = net.forward([arr],[1,0,0,0,0,0,0,0,0,0,0] )
# print(result)
# print(np.argmax(result, axis=1))
