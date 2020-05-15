#!/usr/bin/env python
# coding: utf-8

# In[16]:


#conda create -n yourenvname python=x.x anaconda x.x is the python version and this
#creates a new environmente

#sorce activate yourenvname for activating the environmente

#

# python -m ipykernel install --user --name=firstEnv     First do this to set a new working environment
#where firstEnv = the new environment

import tensorflow.compat.v1 as tf

import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
tf.disable_v2_behavior()
#Define hyperparameters and placeholders

learning_rate = 0.0001
batch_size = 100
update_step = 10
layer_1_nodes = 500
layer_2_nodes = 500
layer_3_nodes = 500
output_nodes = 10

#Placeholders are empty tensors. They have 1 required value: Data type & Default values: Shape = None name = None

#A general feedforward network requires only two placeholders: one for the 
#networks input layer and one for the networks output layer

network_input = tf.placeholder(tf.float32, [None, 784])
#tf.float32 is the datatype of the tensor and refers to 32bit floating point numbers
#Floating point is a technique that allows the decimal point of a number to float.
#it just helps computers work with fractions
#The values in brackets give the shape to the tensor, None = Lenght and
#784 = width. The width is defined by the length of the input image: 28 x 28 pixels

#Finally this line of code defines a tensor that has a length value yet to be determined
#with a height of 784 and a datatype of 32-bit. This will be stored in the placeholder
#waiting for data to be fed.


target_output = tf.placeholder(tf.float32, [None, output_nodes])

## Defining the parameters that the network will tweak as it trains: Weights and biases

layer_1 = tf.Variable(tf.random.normal([784, layer_1_nodes]))
layer_1_bias = tf.Variable(tf.random.normal([layer_1_nodes]))

layer_2 = tf.Variable(tf.random.normal([layer_1_nodes, layer_2_nodes]))
layer_2_bias = tf.Variable(tf.random.normal([layer_2_nodes]))

layer_3 = tf.Variable(tf.random.normal([layer_2_nodes, layer_3_nodes]))
layer_3_bias = tf.Variable(tf.random.normal([layer_3_nodes]))

output_layer = tf.Variable(tf.random.normal([layer_3_nodes, output_nodes]))
output_layer_bias = tf.Variable(tf.random.normal([output_nodes]))

# tf.Variable is a class that creates a tensor that holds and updates parameters. It requires one argument: initial value
# This initial value defines the data type and shape of the tensor. In this case the initial value defines both of these and
# the values are sampled from a random_normal distribution
# A random_normal is a Tensorflow operation that can create a random distribution of numbers. These initial values will define the 
#initial value of tf.Variable. Therefore the initial values of the weights between input layer and hidden layer_1 are defined
# by this random_distribution of numbers. Random_distribution is fed two arguments: 784 & layer_1_nodes. For a neural network
# this represents the total number of connections between network_input nodes and layer_1 nodes & each connection requires
# a weight.

layer_1_output = tf.nn.relu(tf.matmul(network_input, layer_1)+layer_1_bias)
layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, layer_2)+layer_2_bias)
layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, layer_3)+layer_3_bias)
ntwk_output_1 = tf.matmul(layer_3_output, output_layer) + output_layer_bias
ntwk_output_2 = tf.nn.softmax(ntwk_output_1)

#relu is the activation function that will be applied to the matrix multiplication (matmul) of the arguments inputed +
# the biases.

#softmax is an activation function that is specifically used for classification

cf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ntwk_output_1,
                                                           labels = target_output))
ts = tf.train.GradientDescentOptimizer(learning_rate).minimize(cf)
cp = tf.equal(tf.argmax(ntwk_output_2, 1), tf.argmax(target_output,1))
acc = tf.reduce_mean(tf.cast(cp, tf.float32))

#cf computes the cost (or error) of the network (cf = cost function)
#tf.reduce_mean computes the mean of a dimension 
#softmax_cross_entropy_with_logits is a tf operation that performs two tasks at once:
#It applies the softmax activation function to the net input of every output node
#it applies the cross entropy cost function to the new output it generated above(the networks error)
#Logits are the output of all output layer nodes without going through an activation function.
#Labels are the target output of every training image.

#ts = training step, which is the size of step that the network will take torwards minimizing the cost
#function ->This happens by minimizing the cost function(.minimize) through gradient descent
#(.tf.train.gradientdescentoptimizer) while using the learning rate

#cp = correct predictions. It uses tf.argmax in conjunction with tf.equal to check if 
#the networks predictions and lebls are a match. tf.equal takes two arguments.
#tf.argmax- Firstly, every output node in ntwk_output_2 has an index posiition and value.
#The index position is the node's position within an array. The output node with the highest valie
#is considered the networks "prediction". This function finds the max and returns the
#index position as the output.

#step 5 CREATE A SESSION OBJECTION
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_epochs = 58
    for epoch in range(num_epochs):
        total_cost = 0
        for _ in range(int(mnist.train.num_examples/batch_size)):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
        t, c = sess.run([ts, cf], feed_dict={network_input: batch_x,target_output: batch_y})
        total_cost += c
        print('Epoch ' + str(epoch) + ' completed out of ' + str(num_epochs) + 
             ' loss: ' + str(total_cost))
        print('Accuracy: ' + str(acc.eval({network_input:mnist.test.images, target_output: mnist.test.labels})))
 
#sess.run turns on the engine of the tensor graph
#global_variables_initializer turns on all variables defined with tf.Variable


# In[ ]:




