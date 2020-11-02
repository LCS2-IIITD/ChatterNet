import sys
import json
import numpy as np
import tensorflow as tf
from model import DeepCas
import os
import six.moves.cPickle as pickle
import gzip
tf.set_random_seed(0)
import time

NUM_THREADS = 100
delta_t = 30
print(delta_t)
sys.stdout.flush()
tf.flags.DEFINE_float("learning_rate", 0.01, "learning_rate.")
tf.flags.DEFINE_integer("sequence_batch_size", 20, "sequence batch size.")
tf.flags.DEFINE_integer("batch_size", 100, "batch size.")
tf.flags.DEFINE_integer("n_hidden_gru", 32, "hidden gru size.")
tf.flags.DEFINE_float("l1", 5e-5, "l1.")
tf.flags.DEFINE_float("l2", 1e-8, "l2.")
tf.flags.DEFINE_float("l1l2", 1.0, "l1l2.")
tf.flags.DEFINE_string("activation", "relu", "activation function.")
tf.flags.DEFINE_integer("n_sequences", 200, "num of sequences.")
tf.flags.DEFINE_integer("training_iters", 50*3200 + 1, "max training iters.")
tf.flags.DEFINE_integer("display_step", 100, "display step.")
tf.flags.DEFINE_integer("embedding_size", 50, "embedding size.")
tf.flags.DEFINE_integer("n_input", 50, "input size.")
tf.flags.DEFINE_integer("n_steps", 10, "num of step.")
tf.flags.DEFINE_integer("n_hidden_dense1", 32, "dense1 size.")
tf.flags.DEFINE_integer("n_hidden_dense2", 16, "dense2 size.")
tf.flags.DEFINE_string("version", "v4", "data version.")
tf.flags.DEFINE_integer("max_grad_norm", 100, "gradient clip.")
tf.flags.DEFINE_float("stddev", 0.01, "initialization stddev.")
tf.flags.DEFINE_float("emb_learning_rate", 5e-05, "embedding learning_rate.")
tf.flags.DEFINE_float("dropout_prob", 1., "dropout probability.")

config = tf.flags.FLAGS

def get_batch(x, y, sz, step, batch_size=128):
    batch_x = np.zeros((batch_size, len(x[0]), len(x[0][0])))
    batch_y = np.zeros((batch_size, 1))
    batch_sz = np.zeros((batch_size, 1))
    start = step * batch_size % len(x)
    for i in range(batch_size):
        batch_y[i, 0] = y[(i + start) % len(x)]
        batch_sz[i, 0] = sz[(i + start) % len(x)]
        batch_x[i, :] = np.array(x[(i + start) % len(x)])
    return batch_x, batch_y, batch_sz

version = config.version
x_train, y_train, sz_train, vocabulary_size = pickle.load(open('../data/data_train_2.pkl','r'))
print(len(x_train), len(y_train), len(sz_train))
x_test, y_test, sz_test, _ = pickle.load(open('../data/data_test_2.pkl','r'))
print(len(x_test), len(y_test), len(sz_test))
x_val, y_val, sz_val, _ = pickle.load(open('../data/data_val_2.pkl','r'))
print(len(x_val), len(y_val), len(sz_val))
node_vec = pickle.load(open('../data/node_vec.pkl', 'r'))

training_iters = config.training_iters
batch_size = config.batch_size
display_step = min(config.display_step, len(sz_train)/batch_size)


np.set_printoptions(precision=2)
sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True))
start = time.time()
model = DeepCas(config, sess, node_vec)
step = 1
best_val_loss = 1000
best_test_loss = 1000
# Keep training until reach max iterations or max_try
train_loss = []
max_try = 10
patience = max_try
while step * batch_size < training_iters:
    batch_x, batch_y, batch_sz = get_batch(x_train, y_train, sz_train, step, batch_size=batch_size)
    model.train_batch(batch_x, batch_y, batch_sz)
    train_loss.append(model.get_error(batch_x, batch_y, batch_sz))
    if step % display_step == 0:
        # Calculate batch loss
        val_loss = []
        for val_step in range(len(y_val)/batch_size):
            val_x, val_y, val_sz = get_batch(x_val, y_val, sz_val, val_step, batch_size=batch_size)
            val_loss.append(model.get_error(val_x, val_y, val_sz))
        test_loss = []
        for test_step in range(len(y_test)/batch_size):
            test_x, test_y, test_sz = get_batch(x_test, y_test, sz_test, test_step, batch_size=batch_size)
            test_loss.append(model.get_error(test_x, test_y, test_sz))
        
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            best_test_loss = np.mean(test_loss)
            patience = max_try
        print("#" + str(step/display_step) + 
              ", Training Loss= " + "{:.6f}".format(np.mean(train_loss)) + 
              ", Validation Loss= " + "{:.6f}".format(np.mean(val_loss)) + 
              ", Test Loss= " + "{:.6f}".format(np.mean(test_loss)) + 
              ", Best Valid Loss= " + "{:.6f}".format(best_val_loss) + 
              ", Best Test Loss= " + "{:.6f}".format(best_test_loss)
             )
        sys.stdout.flush()
        train_loss = []
        patience -= 1
        if not patience:
            break
        
    step += 1
print "Finished!\n----------------------------------------------------------------"
print "Time:", time.time()-start
print "Valid Loss:", best_val_loss
print "Test Loss:", best_test_loss
sys.stdout.flush()
x_t_t, y_t_t, sz_t_t = get_batch(x_test,y_test,sz_test,step=1,batch_size=len(x_test))
p_pred_get= model.sess.run(model.pred,feed_dict={
                             model.x: x_t_t,
                             model.y: y_t_t,
                             model.sz: sz_t_t
                         })
pred_test_save=[]
y_test_save=[]
sz_test_save=[]
for results in zip(p_pred_get,y_t_t,sz_t_t):
    pred_test_save.append(float(results[0][0]))
    y_test_save.append(float(results[1][0]))
    sz_test_save.append(float(results[2][0]))

with open("../data/test-net/nov_test"+str(delta_t)+".json","w") as f:
    json.dump({
        "predicted_y": pred_test_save,
        "original_y": y_test_save,
        "size": sz_test_save
        }, f,indent=True)
