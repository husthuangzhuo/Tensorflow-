##多层感知器
import tensorflow as tf
import math ##需要向上取整函数
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

sess = tf.InteractiveSession()
n_input = 784
n_hidden = 300
n_output = 10
w1 = tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1))
b1 = tf.Variable(tf.zeros([1, n_hidden]))
w2 = tf.Variable(tf.zeros([n_hidden, n_output]))
b2 = tf.Variable(tf.zeros([n_output]))
x = tf.placeholder(tf.float32, [None, n_input])
y_hat = tf.placeholder(tf.float32, [None, n_output])
keep_prob = tf.placeholder(tf.float32)
hidden = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
hidden_drop = tf.nn.dropout(hidden, keep_prob)
y = tf.nn.softmax(tf.add(tf.matmul(hidden_drop, w2), b2))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_hat*tf.log(y),axis=1)) ##与下一行等价
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_hat*tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(learning_rate=0.3).minimize(cross_entropy)
tf.global_variables_initializer().run()


training_epoch = 10
batch_size = 128
sample_numble = len(mnist.train.images)
total_batch = math.ceil(sample_numble/batch_size)
for epoch in range(training_epoch):
    X_train = mnist.train.images
    Y_train = mnist.train.labels
    for i in range(total_batch-1):
        batch_xs = X_train[:batch_size]
        batch_ys = Y_train[:batch_size]
        sess.run(train_step, feed_dict={x: batch_xs, y_hat:batch_ys, keep_prob:0.75})
        X_train = X_train[batch_size:]
        Y_train = Y_train[batch_size:]
    batch_xs = X_train
    batch_ys = Y_train
    train_step.run(feed_dict={x: batch_xs, y_hat:batch_ys, keep_prob:0.75})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) ##tf.cast是转换函数将True false等布尔值转换成数值
#print(sess.run(accuracy, feed_dict= {x:mnist.test.images, y_hat:mnist.test.labels, keep_prob:1.0})) 与下列方法结果相同
print('accuracy in test: '+ str(accuracy.eval({x:mnist.test.images, y_hat:mnist.test.labels, keep_prob:1.0})))





