#BASIC PREMISE: Using neural network/Machine Learning to curve fit a regression of cryptocurrency data when linked
#FUTURE PLANS: Use this model to predict future prices of stock

#imports
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.pyplot import figure, legend, show, ion, title, pause

#Collecting data
#Links for eth (2 years): https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20160726&end=20180726
#Link for bit (2 years): https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20160726&end=20180726

link_crypto = input("Paste correct link to CoinMarketCap data: ")

data_reader = pd.read_html(link_crypto)[0]
data_reader = data_reader.drop(['Date'], 1)
data_reader.head()

row = data_reader.shape[0]
col = data_reader.shape[1]

data_reader = np.array(data_reader)

#Separating train and test data as well as data cleaning
test_start = 0  #index of the most recent piece of data
test_end = int(np.floor(0.2*row))   #rounds data and scales

train_start = test_end + 1  #sets the rest of the data as training
train_end = row

data_train = data_reader[np.arange(train_start, train_end), :]  #evenly spaces out data points
data_test = data_reader[np.arange(test_start, test_end), :]

scaler = MinMaxScaler(feature_range=(-1, 1))    #-1 to 1 for the z-scale (stats stuff)
scaler.fit(data_train)  #scales and fits data for training

data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

#initializing x and y values for training and testing sessions
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

#total number of stocks in training data
n_stocks = X_train.shape[1] #.shape is like len

#beginning neurons
n_neurons_1 = 1024
n_neurons_2 = 51
n_neurons_3 = 256
n_neurons_4 = 128

# Begins the Session
net = tf.InteractiveSession()

# Placeholders for the session to utilize
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

#Weight of neurons -- creates variables that are used in the network
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Variable for output
W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
bias_out = tf.Variable(bias_initializer([1]))

#Hidden layers in the convoluted neural network
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

#output layer (see above)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

#cost/loss function
cost = tf.reduce_mean(tf.squared_difference(out, Y))

#optimizer
opt = tf.train.AdamOptimizer().minimize(cost)

# Init
net.run(tf.global_variables_initializer())

# Setup plot
ion()
fig = figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test, label='Actual')
line2, = ax1.plot(y_test * 0.5, label='Prediction')
legend(handles=[line1, line2])
show()

# Fit neural net
batch_size = 256
cost_train = []
cost_test = []

# Run
epochs = 200
for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    #batch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 50) == 0:
            # cost train and test
            cost_train.append(net.run(cost, feed_dict={X: X_train, Y: y_train}))
            cost_test.append(net.run(cost, feed_dict={X: X_test, Y: y_test}))
            print('cost train: ', cost_train[-1])
            print('cost test: ', cost_test[-1])

            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            title('Epoch ' + str(e))
            pause(0.025)


exit_ind = input("Type 0 when finished: ")
mse_final = net.run(cost, feed_dict={X: X_test, Y: y_test})
print("Error: " + str(mse_final))
data_new = scaler.inverse_transform(data_test)
print("Current Closing Price: $"+str(data_new[0][0]))
net.close()