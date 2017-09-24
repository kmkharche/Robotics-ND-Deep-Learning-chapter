#!usr/bin/env python

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import math

def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    output_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        print (end_i)
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
        
    return output_batches


n_input = 784 #input image size (28 by 28)
n_classes = 10

# import data
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# split data into training, validation and test sets
train_features = mnist.train.images
valid_features = mnist.validation.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
valid_labels = mnist.validation.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

features = tf.placeholder(tf.float32,[None,n_input])
labels = tf.placeholder(tf.float32,[None,n_classes])

weights = tf.Variable(tf.random_normal([n_input,n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

logits=tf.add(tf.matmul(features,weights),bias)

# Define loss and optimizer
learning_rate=tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn_softmax_cross_entropy_with_logits(logits=logits,labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal (tf.argmax(logits,1),tf.argmax(labels,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,float32))

init = tf.global_variables_initializer()

batch_size = 128
epochs = 10
learn_rate = 0.001

train_batches = batches(batch_size, train_features, train_labels)

with tf.Session() as sess:

	sess.run(init)

	for epoch_i in range(epochs):
		for batch_features, batch_labels in train_batches:
			sess.run(optimizer,feed_dict={learning_rate: learn_rate, features: batch_features, labels: batch_labels})


	test_accuracy = sess.run(accuracy, feed_dict={features: test_features, labels: test_labels})



