import tensorflow as tf
import tensorflow_datasets as tfds
#from tensorflow.examples.tutorials.mnist import input_data
import os
#from tensorflow.examples.tutorials.mnist import input_data
import os
tf.compat.v1.disable_eager_execution()
#from dataset import fashion_MNIST
data = tfds.load('mnist', split='train', shuffle_files=True)



#from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.keras.datasets import mnist
#mnist = input_data.read_data_sets("../tmp/data/", one_hot = True)
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.compat.v1.placeholder('float', [None, 784])
y = tf.compat.v1.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random.normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random.normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights':tf.Variable(tf.random.normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random.normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random.normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random.normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random.normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random.normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.optimizers.Adam()

    hm_epochs = 10
    with tf.compat.v1.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(data.train.num_examples/batch_size)):
                epoch_x, epoch_y = (mnist.train.next_batch(batch_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))


        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
