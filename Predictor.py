import tensorflow as tf
from DataHandler import DataHandler
from ModelHandler import ModelHandler

input_file_path = "state_responses.csv"
input_test_file_path = "state_responses.csv"
n_classes = 4
batch_size = 100

grid_size = 4
input_count = grid_size * grid_size
conv1_out = 68
conv2_out = 68
conv3_out = 68
fc_input = 512
learning_rate = 0.01
epoch_size = 15

x = tf.placeholder('float', [None, input_count])
y = tf.placeholder('float')

keep_rate = 0.5
keep_prob = tf.placeholder(tf.float32)

data_handler = DataHandler(input_count, n_classes, input_file_path)
data_handler_test = DataHandler(input_count, n_classes, input_test_file_path)
model_handler = ModelHandler(model_name="2048-cnn")


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([2, 2, 1, conv1_out])),
               'W_conv2': tf.Variable(tf.random_normal([2, 2, conv1_out, conv2_out])),
               'W_conv3': tf.Variable(tf.random_normal([2, 2, conv2_out, conv3_out])),
               'W_fc': tf.Variable(tf.random_normal([grid_size * grid_size * conv3_out, fc_input])),
               'out': tf.Variable(tf.random_normal([fc_input, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([conv1_out])),
              'b_conv2': tf.Variable(tf.random_normal([conv2_out])),
              'b_conv3': tf.Variable(tf.random_normal([conv3_out])),
              'b_fc': tf.Variable(tf.random_normal([fc_input])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, grid_size, grid_size, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    # Not using max pooling for conv1
    #conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    # Not using max pooling for conv1
    #conv2 = maxpool2d(conv2)

    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
    #conv3 = maxpool2d(conv3)

    #conv4 = tf.nn.relu(conv2d(conv3, weights['W_conv4']) + biases['b_conv4'])
    #conv4 = maxpool2d(conv4)

    fc = tf.reshape(conv3, [-1, grid_size * grid_size * conv3_out])

    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']
    return tf.nn.softmax(output)


prediction = convolutional_neural_network(x)
saver = tf.train.Saver()


def predict(data):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model_handler.restore_model(saver_object=saver, session=sess, log=False)

        return sess.run([prediction], feed_dict={x: data})
