"""
not initializer
cross_validation

hidden layer = 3
learning rate = 0.01
sigmoid
GradientDescent
Num of loop = 100


Num of row = 47703
Num of col = 523
---------------------------------
ACC=0.995
AUC=0.98
"""


import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split

tf.set_random_seed(777) # for reproducibility
learning_rate = 0.01
keep_prob = tf.placeholder(tf.float32)
#reading training Data
with open("TBEa.csv") as f:
    ncols = len(f.readline().split(','))


xy = np.loadtxt("TBEa.csv", skiprows=1, usecols=range(1,ncols), delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.1, random_state = 42)



#calculating AUC


X = tf.placeholder(tf.float32, [None, 523])
Y = tf.placeholder(tf.float32, [None, 1])


with tf.name_scope("layer1") as scope:

	W1 = tf.get_variable("W1", shape=[523, 256],initializer=tf.contrib.layers.xavier_initializer())
	b1 = tf.Variable(tf.random_normal([256]))
	L1 = tf.sigmoid(tf.matmul(X, W1) + b1)
	
	w1_hist = tf.summary.histogram("weights1",W1)
	b1_hist = tf.summary.histogram("biases1",b1)
	L1_hist = tf.summary.histogram("layer1", L1);

with tf.name_scope("layer2") as scope:
	W2 = tf.get_variable("W2", shape=[256, 256],initializer=tf.contrib.layers.xavier_initializer())
	b2 = tf.Variable(tf.random_normal([256]))
	L2 = tf.sigmoid(tf.matmul(L1, W2) + b2)

	w2_hist = tf.summary.histogram("weights2",W2)
	b2_hist = tf.summary.histogram("biases2",b2)
	L2_hist = tf.summary.histogram("layer2", L2);

with tf.name_scope("layer3") as scope:

	W3 = tf.get_variable("W3", shape=[256, 1],initializer=tf.contrib.layers.xavier_initializer())
	b3 = tf.Variable(tf.random_normal([1]))
	hypothesis = tf.sigmoid(tf.matmul(L2, W3) + b3)
	auc_update = tf.contrib.metrics.streaming_auc(predictions = hypothesis,labels=Y,curve='ROC')
	

	w3_hist = tf.summary.histogram("weights3",W3)
	b3_hist = tf.summary.histogram("biases3",b3)
	hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis);
# define cost/loss & optimizer
with tf.name_scope("cost") as scope:
	cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 -  hypothesis))
	cost_summ = tf.summary.scalar("cost",cost)

with tf.name_scope("train") as scope:
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Accuracy computation
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)
auc_sum = tf.summary.scalar("AUC",tf.reduce_mean(auc_update))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/TBE4")
    writer.add_graph(sess.graph) #show graph

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    for step in range(100):
        summary, _ = sess.run([merged_summary,optimizer], feed_dict={X: x_train, Y: y_train}) # cost, hyphothesis was calaulated with optimizer
	writer.add_summary(summary, global_step=step)

        if step % 10 == 0:
            print(sess.run(cost, feed_dict={X: x_train, Y: y_train}))
	    print(sess.run(auc_update, feed_dict={X: x_test, Y: y_test}))

    # Accuracy report
    h, a = sess.run([hypothesis, accuracy], feed_dict={X: x_test, Y: y_test})
    train_auc = sess.run(auc_update, feed_dict={X: x_test, Y: y_test})


print("Hypothesis: ", h)
print("Accuracy: ", a)
print("AUC: ", train_auc)



