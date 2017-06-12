import os
import sys
import tensorflow as tf
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('Loading Data...')

bed = open(sys.argv[1], 'r')
tracks = pickle.load(open(sys.argv[2], 'rb'))
model_name = sys.argv[3]

max_track = tracks['max']
res = tracks['res']

input_dim = 100000 / res

flanking_signals = 100

boundaries = []

for i, line in enumerate(bed):
	if i == 0:
		continue
	chrm = line.split()[0]
	boundary_index = int(int(line.split()[2]) / res)

	if boundary_index + flanking_signals >= len(max_track[chrm]):
		continue
	else:
		boundary_signal = max_track[chrm][boundary_index-flanking_signals:boundary_index+flanking_signals]
	boundaries.append(boundary_signal)

boundaries = np.asarray(boundaries)
print(boundaries.shape)

# Autoencoder
def autoencoder(X, weights, biases):
    with tf.name_scope('hidden_layer'):
        hiddenlayer = tf.sigmoid(
            tf.add(
                tf.matmul(
                    X, weights['hidden']
                ),
                biases['hidden']
            )
        )
    with tf.name_scope('output_layer'):
        out = tf.sigmoid(
            tf.add(
                tf.matmul(
                    hiddenlayer, weights['out']
                ), 
                biases['out']
            )
        )
    return {'out': out, 'hidden': hiddenlayer}

# Cost
def KL_Div(rho, rho_hat):
    invrho = tf.subtract(tf.constant(1.), rho)
    invrhohat = tf.subtract(tf.constant(1.), rho_hat)
    logrho = tf.add(logfunc(rho,rho_hat), logfunc(invrho, invrhohat))
    return logrho

def logfunc(x, x2):
    cx = tf.clip_by_value(x, 1e-10, 1.0)
    cx2 = tf.clip_by_value(x2, 1e-10, 1.0)
    return tf.multiply( x, tf.log(tf.div(cx,cx2)))

sess = tf.InteractiveSession()

BETA = tf.constant(3.)
LAMBDA = tf.constant(.0001)
RHO = tf.constant(0.01)
EPSILON = .000001

n_input = flanking_signals * 2
n_hidden = 72
num_samples = boundaries.shape[0]

x = tf.placeholder("float", [None, n_input])
hidden = tf.placeholder("float", [None, n_hidden])

weights = {
	'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
	'out': tf.Variable(tf.random_normal([n_hidden, n_input]))
}

biases = {
	'hidden': tf.Variable(tf.random_normal([n_hidden])),
	'out': tf.Variable(tf.random_normal([n_input]))	
}

pred = autoencoder(x, weights, biases)

# loss variables
rho_hat = tf.div(tf.reduce_sum(pred['hidden'],0),tf.constant(float(num_samples)))
diff = tf.subtract(pred['out'], x)

cost_J = tf.div(tf.nn.l2_loss(diff),tf.constant(float(num_samples)))

cost_sparse = tf.multiply(BETA, tf.reduce_sum(KL_Div(RHO, rho_hat)))

cost_reg = tf.multiply(LAMBDA, tf.add(tf.nn.l2_loss(weights['hidden']), tf.nn.l2_loss(weights['out'])))

cost = tf.add(tf.add(cost_J, cost_reg), cost_sparse)

optimizer = tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()

sess.run(init)

# Training cycle
c = 0.
c_old = 1.
i = 0
while np.abs(c - c_old) > EPSILON:
    sess.run([optimizer], feed_dict={x: boundaries})
    if i % 1000 == 0:
        c_old = c
        c,j,reg,sparse = sess.run([cost,cost_J,cost_reg,cost_sparse], feed_dict={x: boundaries})
        print ("EPOCH %d: COST = %f, LOSS = %f, REG_PENALTY = %f, SPARSITY_PENTALTY = %f" %(i,c,j,reg,sparse))
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, './' + model_name + '.ckpt')
    i += 1

print("Optimization Finished!")