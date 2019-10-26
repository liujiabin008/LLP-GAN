import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
import argparse
import utils
import load_svhn



# settings
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../datasets/cifar')
parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints")
parser.add_argument('--images_dir', type=str, default="./out")
parser.add_argument('--load', dest='LOAD', action='store_true')
parser.add_argument('--pull_away', dest='pull_away', action='store_true')
args = parser.parse_args()
print(args)

#trainx, trainy = cifar10_data.load(args.data_dir)
#testx, testy = cifar10_data.load(args.data_dir, subset='test')

trainx, trainy, validx, validy, testx, testy = utils.load_data()
#trainx, trainy = load_svhn.load_train_data()
#testx, testy = load_svhn.load_test_data()


def plot(samples):
    width = min(12,int(np.sqrt(len(samples))))
    fig = plt.figure(figsize=(width, width))
    gs = gridspec.GridSpec(width, width)
    gs.update(wspace=0.05, hspace=0.05)

    for ind, sample in enumerate(samples):
        if ind >= width*width:
            break
        ax = plt.subplot(gs[ind])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample = sample
        #sample = np.transpose(sample, (1, 2, 0))
        plt.imshow(sample)

    return fig


mb_size = 36
image_width=32
X_dim = image_width*image_width
channels = 3
z_dim = 100
lr = 3e-4
BETA = 0.5
BETA2 = 1e-3
NUM_CLASSES = 10


X_ = tf.placeholder(tf.float32, shape=[mb_size, image_width, image_width, channels])
#X_lab = tf.placeholder(tf.float32, shape=[mb_size, channels, image_width, image_width])
Y_ = tf.placeholder(tf.float32, shape=[None, 10])
z_ = tf.placeholder(tf.float32, shape=[mb_size, z_dim])
y_prob = tf.placeholder(tf.float32, shape=[10])
# z_prime1 = tf.random_uniform(shape=[mb_size, z_dim], minval=-1, maxval=1)
# z_prime2 = tf.random_uniform(shape=[mb_size, z_dim], minval=-1, maxval=1)
trnow = tf.placeholder(tf.bool) #is training now?

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * tf.abs(x)

def convlayer(layer, filter_size, stride, filters, bias=True, nonlinearity=lrelu):
    return tf.layers.conv2d(inputs=layer, filters=filters, kernel_size=[filter_size,filter_size], padding='same', activation=nonlinearity, strides=(stride,stride), use_bias=bias)

#transpoose convolution (deconvolution)
def convlayer_t(layer, filter_size, stride, filters, bias=True, nonlinearity=lrelu):
    return tf.layers.conv2d_transpose(inputs=layer, filters=filters, kernel_size=[filter_size,filter_size], padding='same', activation=nonlinearity, strides=(stride,stride), use_bias=bias)

#transpoose convolution (deconvolution)
def bn_convlayer_t(layer, filter_size, stride, filters, bias=True, nonlinearity=lrelu):
    return tf.contrib.layers.batch_norm(tf.layers.conv2d_transpose(inputs=layer, filters=filters, kernel_size=[filter_size,filter_size], padding='same', strides=(stride,stride), use_bias=bias), is_training=trnow, fused=True, activation_fn=nonlinearity)

def bn_convlayer(layer, filter_size, stride, filters, bias=True, nonlinearity=lrelu):
    return tf.contrib.layers.batch_norm(tf.layers.conv2d(inputs=layer, filters=filters, kernel_size=[filter_size,filter_size], padding='same', strides=(stride,stride), use_bias=bias), is_training=trnow, fused=True, activation_fn=nonlinearity)

def dense(layer, units, bias=True, nonlinearity=lrelu):
    return tf.layers.dense(inputs=layer, units=units, activation=nonlinearity, use_bias=bias)

def bn_dense(layer, units, bias=True, nonlinearity=lrelu):
    return tf.contrib.layers.batch_norm(tf.layers.dense(inputs=layer, units=units, use_bias=bias), is_training=trnow, fused=True,  activation_fn=nonlinearity)

def dropout_layer(layer, rate):
    return tf.layers.dropout(inputs=layer, rate=rate, training=trnow)

def global_pool(layer, old_width):
    return tf.squeeze(tf.layers.average_pooling2d(layer, [old_width, old_width], [old_width, old_width]))


def G(z, reuse=False):
    with tf.variable_scope('G_', reuse=reuse):
        h = bn_dense(z, 4*4*512, bias=False)
        h = tf.reshape(h, (mb_size, 4, 4, 512))
        h = bn_convlayer_t(h, 5, 2, 256, bias=False)
        h = bn_convlayer_t(h, 5, 2, 128, bias=False)
        h = tf.layers.conv2d_transpose(inputs=h, filters=channels, kernel_size=[5,5], padding='same', activation=tf.nn.tanh, strides=(2,2), use_bias=True)
        return h


def D(X, reuse=False): #discriminator
    with tf.variable_scope('D_', reuse=reuse):
        h_X = dropout_layer(X, 0.2)
        h_X = convlayer(h_X, 3, 1, 64)
        h_X = convlayer(h_X, 3, 1, 64)
        h_X = convlayer(h_X, 3, 2, 64)
        h_X = dropout_layer(h_X, 0.5)
        h_X = convlayer(h_X, 3, 1, 128)
        h_X = convlayer(h_X, 3, 1, 128)
        h_X = convlayer(h_X, 3, 2, 128)
        h_X = dropout_layer(h_X, 0.5)
        h_X = convlayer(h_X, 3, 1, 256)
        h_X = convlayer(h_X, 1, 1, 128)
        h_X = convlayer(h_X, 1, 1, 64)
        h_X = global_pool(h_X, 8)

        h = tf.layers.dense(h_X, NUM_CLASSES)

        return h, h_X

def proportion_compute(label_input):

    batch_size = len(label_input)
    prob_result = label_input.sum(axis = 0)/batch_size
    return  prob_result

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])



g_net = G(z_)
d_net_real, d_net_real_feat = D(X_)
d_net_fake, d_net_fake_feat = D(g_net, reuse=True)


l_real = tf.reduce_logsumexp(d_net_real, axis=1)
l_fake = tf.reduce_logsumexp(d_net_fake, axis=1)





theta_G = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G_')
theta_D = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D_')

D_loss_prop =  tf.reduce_mean(-tf.reduce_sum(y_prob*(tf.log(tf.reduce_mean(tf.nn.softmax(d_net_real), [0]) + 1e-7))))
D_loss_unl = -tf.reduce_mean(l_real) + tf.reduce_mean(tf.nn.softplus(l_real)) + tf.reduce_mean(tf.nn.softplus(l_fake))



D_loss = D_loss_unl+ D_loss_prop

G_loss = tf.reduce_mean(tf.square(tf.reduce_mean(d_net_real_feat, axis=0)-tf.reduce_mean(d_net_fake_feat, axis=0)))

if args.pull_away:
    #pull-away term
    feat_norm = d_net_fake_feat / tf.norm(d_net_fake_feat, axis=1, keep_dims=True)
    G_pt = tf.tensordot(feat_norm, feat_norm, axes=[[1],[1]])
    G_pt = tf.reduce_mean(G_pt)
    print(G_pt.get_shape())

    G_loss += G_pt

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=BETA)
    D_solver = (opt.minimize(D_loss, var_list=theta_D))

    G_solver = (opt.minimize(G_loss, var_list=theta_G))


error_op = 1.0 - tf.reduce_mean(tf.cast(tf.equal(tf.argmax(d_net_real, axis=1), tf.argmax(Y_,axis=1)), tf.float32))



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))

def compute_error():
    err = 0

    nmb = testx.shape[0] // mb_size

    for mb in range(nmb):
        err += sess.run(error_op, feed_dict={X_: testx[mb*mb_size:(mb+1)*mb_size], Y_: testy[mb*mb_size:(mb+1)*mb_size], trnow:False})

    err /= nmb
    return err

#os.makedirs(args.images_dir)
#os.makedirs(args.checkpoint_dir)

disp_z = sample_z(mb_size, z_dim)

for epoch in range(3000):
    nmb = trainx.shape[0] // mb_size
    timer = time.time()



    for mb in range(nmb):
        X_mb = trainx[mb*mb_size:(mb+1)*mb_size]
        Y_mb = trainy[mb*mb_size:(mb+1)*mb_size]
        y_prob_mb =  proportion_compute(Y_mb)

        z_mb = sample_z(mb_size, z_dim)

        _, D_loss_result, D_loss_prop_result = sess.run(
            [D_solver, D_loss, D_loss_prop], feed_dict={X_: X_mb, z_: z_mb, y_prob: y_prob_mb, trnow:True}
        )

        _, G_loss_result = sess.run(
            [G_solver, G_loss], feed_dict={X_: X_mb, z_: z_mb, trnow:True}
        )

        #if 100*mb // nmb > lp:
            #lp = 100*mb // nmb
            #print('.', end='', flush=True)


    #samples = sess.run(g_net, feed_dict={z_: disp_z, trnow:False})
    #print samples[0,:,:,0]

    #fig = plot(samples)
    #plt.savefig(os.path.join(args.images_dir,'{}.png'.format(str(epoch).zfill(3))), bbox_inches='tight')
    #plt.close(fig)

    err = compute_error()

    f = open('test_gan_36.txt', 'w')
    f.write(str(err) + '\n')
    f.close()

    saver.save(sess=sess,save_path=os.path.join(args.checkpoint_dir,'checkpoint'), global_step=mb)

    print('')
    print('e: {}\ttime: {:.6}\tD_loss: {:.4}\tD_loss_prop: {:.4}\tG_loss: {:.4}\terr:{:.4}'
          .format(epoch, time.time()-timer, D_loss_result, D_loss_prop_result, G_loss_result, err))
