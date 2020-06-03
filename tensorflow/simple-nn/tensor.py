import tensorflow_datasets as tfds
#from tensorflow.examples.tutorials.mnist import input_data
import os
#from tensorflow.examples.tutorials.mnist import input_data
import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#from dataset import fashion_MNIST
data = tfds.load('mnist', split='train', shuffle_files=True)

# Download the MNIS dataset
#mnist=input_data.read_data_sets('../model_data/', one_hot=True)
# import tensorflow to the environment
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# initializing parameters for the model
batch = 100
learning_rate = 0.01
training_epochs = 10

# creating placeholders
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

# creating variables
W = tf.compat.v1.Variable(tf.zeros([784, 10]))
b = tf.compat.v1.Variable(tf.zeros([10]))

# initializing the model
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Defining Cost Function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.math.log(y)))

# Determining the accuracy of parameters
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Implementing Gradient Descent Algorithm
train_op = tf.optimizers.GradientDescent(learning_rate).minimize(cross_entropy)

# Initializing the session
with tf.compat.v1.Session() as sess:
  sess.run(tf.compat.v1.global_variables_initializer())

  # Creating batches of data for epochs
  for epoch in range(training_epochs):
      batch_count = int(data.train.num_examples / batch)
      for i in range(batch_count):
          batch_x, batch_y = data.train.next_batch(batch)
          # Executing the model
          sess.run([train_op], feed_dict={x: batch_x, y_: batch_y})
          # Print Accuracy of the model

      if epoch % 2 == 0:
          print("Epoch: ", epoch)
          print("Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

      print("Final Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print("Model Execution Complete")
