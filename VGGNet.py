from datetime import datetime
import math
import time
import tensorflow as tf

def conv_op(input_op, name, kh, kw, n_out, dh, dw, parameter):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w', shape=[kh, kw, n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, strides=[1,dh,dw,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_out], name='b'))
        z = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(z,name=scope)
        parameter += [kernel, biases]
        return conv

def fc_op(input_op, name, n_out, parameter):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope+'w', shape=[n_in,n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_out]),name='b')
        z = tf.nn.relu_layer(input_op, w, b, name=scope)
        parameter +=[w,b]
        return z

def max_pool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize = [1,kh,kw,1], strides=[1,dh, dw, 1], padding='SAME', name=name)

def inferecen_op(input_op, keep_prob):
    parameter = []
    conv1_1 = conv_op(input_op, name='conv1_1', kh=3,kw=3, n_out=64, dh= 1, dw=1, parameter=parameter)
    conv1_2 = conv_op(conv1_1, name='conv1_2',kh=3, kw=3, n_out=64, dh=1,dw=1, parameter= parameter)
    pool1 = max_pool_op(conv1_2, name='pool1', kh=2, kw=2, dh=2, dw=2)
    conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, parameter=parameter)
    conv2_2 = conv_op(conv2_1, name='conv2_2',kh=3,kw=3, n_out=128,dh=1, dw=1, parameter=parameter)
    pool2 = max_pool_op(conv2_2, name='pool2', kh=2, kw=2, dh=2, dw=2)
    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1,dw=1,parameter=parameter)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, parameter=parameter)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, parameter=parameter)
    pool3= max_pool_op(conv3_3, name='pool3', kh=2,kw=2, dh=2, dw=2)
    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, parameter=parameter)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, parameter=parameter)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, parameter=parameter)
    pool4 = max_pool_op(conv4_3, name='pool4', kh=2,kw=2, dh=2,dw=2)
    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, parameter=parameter)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, parameter=parameter)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, parameter=parameter)
    pool5 = max_pool_op(conv5_3, name='pool5', kh=2, kw=2, dh=2, dw=2)
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    pool5_flat = tf.reshape(pool5, shape=[-1,flattened_shape],name='resh1')
    fc6 = fc_op(pool5_flat,name='fc6',n_out=4096, parameter=parameter)
    fc6_drop = tf.nn.dropout(fc6,keep_prob=keep_prob,name='fc6_drop')
    fc7 = fc_op(fc6_drop, name='fc7', n_out = 4096, parameter=parameter)
    fc7_drop = tf.nn.dropout(fc7, keep_prob=keep_prob,name='fc7_drop')
    fc8 = fc_op(fc7_drop,name='fc8', n_out=1000, parameter=parameter)
    softmax= tf.nn.softmax(fc8)
    prediction = tf.argmax(softmax,axis=1)
    return prediction, softmax, fc8, parameter

def time_tensorflow_run(session, target, feed, info_string):
    num_step_burn_in = 10
    total_duration = 0.0
    total_duration_squared=0.0

    for i in range(num_batches + num_step_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time()-start_time
        if i >=num_step_burn_in:
            if not i%10:
                print('%s:step %d, duraton= %.3f'%(datetime.now(), i-num_step_burn_in, duration))
            total_duration +=duration
            total_duration_squared += duration*duration
    mn = total_duration/num_batches
    vr = total_duration_squared/num_batches - mn*mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch'%(datetime.now(), info_string, num_batches, mn, sd))

def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3],dtype=tf.float32, stddev=1e-1))
        keep_prob = tf.placeholder(tf.float32)
        prediction, softmax, fc8, parameter =inferecen_op(images, keep_prob)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess,prediction, {keep_prob:1.0},'Forward')
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, parameter)
        time_tensorflow_run(sess, grad, {keep_prob:0.5}, 'Forward-backward')

batch_size = 32
num_batches = 100
run_benchmark()


