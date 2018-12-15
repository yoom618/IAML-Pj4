import tensorflow as tf
import pickle
import numpy as np
from dataloader import *
from vae_musicvae import *
from tensorflow.python.platform import gfile
from time import strftime, localtime, time

############ hyperparameters ###############

# is_train_mode, checkpoint_path = True, 'checkpoint'
is_train_mode, checkpoint_path = False, 'checkpoint/run-1211-1642-0050/'
midi_size, pianoroll_path = 128, 'pianoroll.pkl'
midi_len = 256
midi_dim = midi_size * midi_len

learning_rate = 0.1
num_epochs = 500
batch_size = 72
keep_prob = 0.7

latent_dim = 8
encode_hidden_size = [32]
decode_hidden_size = [32]
is_hierarchical = True  # 인코더가 hierarchical인지 아닌지 여부
encoder_level_length = [16,4,4]  # is_hierarchial이 True일 때만 적용됨
decoder_level_length = [16,16]
disable_AR = True  # disable auto-regression

free_bits = 0  # Bits to exclude from KL loss per dimension.
max_beta = 10.0  # Maximum KL cost weight, or cost if not annealing.
beta_rate = 0.0  # Exponential rate at which to anneal KL cost
sampling_schedule = 'inverse_sigmoid'  # constant, exponential, inverse_sigmoid
sampling_rate = 1000.0  # Interpretation is based on `sampling_schedule`.'

use_cudnn = False  # Uses faster CudnnLSTM to train. For GPU only.
residual_encoder = False  # Use residual connections in encoder.
residual_decoder = False  # Use residual connections in decoder.

### sample generation
num_samples = 10
randomness = 1.0

#############################################


# fixed
data_path = 'data/'
sample_path = 'sample/'
if not gfile.Exists(sample_path):
    os.mkdir(sample_path)

# True if you want to train, False if you already trained your model
### TODO : IMPORTANT !!! Please change it to False when you submit your code
### TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : checkpoint/run-0925-0348


if not gfile.Exists(checkpoint_path):
    os.mkdir(checkpoint_path)

# if there is no pianoroll file, make it
if gfile.Exists(pianoroll_path):
    print('Load pianoroll file :', pianoroll_path)
    with open(pianoroll_path, 'rb') as f:
        pianoroll_list = pickle.load(f)
else:
    print('Make pianoroll file :', pianoroll_path)
    pianoroll_list = get_pianoroll_list(data_path)
    with open(pianoroll_path, 'wb') as f:
        pickle.dump(pianoroll_list, f)

a = np.argmax(pianoroll_list, axis=2)
switch = np.concatenate([np.concatenate([a[:,16*2:16*14],a[:,16*2:16*6]], axis=1),
                np.concatenate([a[:,16*6:16*14],a[:,16*2:16*10]], axis=1),
                np.concatenate([a[:,16*10:16*14],a[:,16*2:16*14]], axis=1)],axis=0)
pianoroll_list = np.eye(128)[switch]
print(pianoroll_list.shape)

# split train and test
split = int(pianoroll_list.shape[0]*0.8)
# build dataset
dataset = tf.data.Dataset.from_tensor_slices(pianoroll_list)
train_dataset = dataset.take(split)
test_dataset = dataset.skip(split)

train_dataset = train_dataset.shuffle(buffer_size=split)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
train_dataset = train_dataset.prefetch(batch_size)
train_dataset = train_dataset.repeat(1)

test_dataset = test_dataset.shuffle(buffer_size=pianoroll_list.shape[0] - split)
test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
test_dataset = test_dataset.prefetch(batch_size)
test_dataset = test_dataset.repeat(1)

iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_init = iter.make_initializer(train_dataset)
test_init = iter.make_initializer(test_dataset)

# batch of features, batch of labels
X = iter.get_next()
is_training = tf.placeholder(tf.bool)
dropout_keep_prob = tf.placeholder(tf.float32)

# TODO : build your model here
global_step = tf.Variable(0, trainable=False, name='global_step')

hparams=tf.contrib.training.HParams(
    max_seq_len=midi_len,  # Maximum sequence length. Others will be truncated.
    z_size=latent_dim,  # Size of latent vector z.
    enc_rnn_size=encode_hidden_size,
    dec_rnn_size=decode_hidden_size,
    free_bits=free_bits,  # Bits to exclude from KL loss per dimension.
    max_beta=max_beta,  # Maximum KL cost weight, or cost if not annealing.
    beta_rate=beta_rate,  # Exponential rate at which to anneal KL cost.
    batch_size=batch_size,  # Minibatch size.
    grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
    clip_mode='global_norm',  # value or global_norm.
    # If clip_mode=global_norm and global_norm is greater than this value,
    # the gradient will be clipped to 0, effectively ignoring the step.
    grad_norm_clip_to_zero=10000,
    dropout_keep_prob=dropout_keep_prob,  # Probability for dropout keep.
    learning_rate=learning_rate,  # Learning rate.
    decay_rate=0.999,  # Learning rate decay per minibatch.
    min_learning_rate=0.00001,  # Minimum learning rate.
    conditional=False,
    sampling_schedule=sampling_schedule,  # constant, exponential, inverse_sigmoid
    sampling_rate=sampling_rate,  # Interpretation is based on `sampling_schedule`.'
    use_cudnn=use_cudnn,  # Uses faster CudnnLSTM to train. For GPU only.
    residual_encoder=residual_encoder,  # Use residual connections in encoder.
    residual_decoder=residual_decoder  # Use residual connections in decoder.

)


seq_length = tf.fill([tf.shape(X)[0]], midi_len)
if not is_hierarchical:
    model = MusicVAE(BidirectionalLstmEncoder(),
                 HierarchicalLstmDecoder(CategoricalLstmDecoder(),
                                         level_lengths=decoder_level_length, disable_autoregression=disable_AR))
else:
    model = MusicVAE(HierarchicalLstmEncoder(BidirectionalLstmEncoder, encoder_level_length),
                 HierarchicalLstmDecoder(CategoricalLstmDecoder(),
                                         level_lengths=decoder_level_length, disable_autoregression=disable_AR))
model.build(hparams, output_depth=midi_size, is_training=is_training)
train_metric, optimizer, train_loss = model.train(X, X, seq_length)
grads, var_list = zip(*optimizer.compute_gradients(model.loss))
global_norm = tf.global_norm(grads)
tf.summary.scalar('global_norm', global_norm)
if hparams.clip_mode == 'value':
    g = hparams.grad_clip
    clipped_grads = [tf.clip_by_value(grad, -g, g) for grad in grads]
elif hparams.clip_mode == 'global_norm':
    clipped_grads = tf.cond(
        global_norm < hparams.grad_norm_clip_to_zero,
        lambda: tf.clip_by_global_norm(  # pylint:disable=g-long-lambda
            grads, hparams.grad_clip, use_norm=global_norm)[0],
        lambda: [tf.zeros(tf.shape(g)) for g in grads])
train = optimizer.apply_gradients(zip(clipped_grads, var_list), global_step=model.global_step,name='train_step')

test_metric, test_loss = model.eval(X, X, seq_length)

# train and evaluate
with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=20)
    time_path = '/run-%02d%02d-%02d%02d/' % tuple(localtime(time()))[1:5]
    if is_train_mode:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print('1. Start training')
        train_total_batch = int(split / batch_size)
        test_total_batch = int((pianoroll_list.shape[0] - split) / batch_size)

        for epoch in range(num_epochs):
            sess.run(train_init)
            acc = 0.0
            print('------------------- epoch:', epoch, ' -------------------')
            for step in range(train_total_batch):
                _, c, m = sess.run([train, train_loss, train_metric], feed_dict={is_training: True, dropout_keep_prob: keep_prob})
                acc += m['metrics/accuracy'][0] / train_total_batch
            print('Step: %5d, Loss: %4.3f, r_loss: %4.3f, kl_losss: %4.3f, kl_bits: %4.3f, kl_beta: %.2f '
                      % (sess.run(model.global_step), c['loss'], np.average(c['losses/r_loss'], axis=0),
                         np.average(c['losses/kl_loss'], axis=0), np.average(c['losses/kl_bits'], axis=0),
                         c['losses/kl_beta']), flush=True)
            print('Training Accuracy: %.4f' % acc)

            sess.run(test_init)
            loss, r_loss, kl_loss, kl_bits, kl_beta = 0.0, 0.0, 0.0, 0.0, 0.0
            for _ in range(test_total_batch):
                m, c = sess.run([test_metric, test_loss], feed_dict={is_training: False, dropout_keep_prob: 1.0})
                loss += c['loss'] / test_total_batch
                r_loss += np.average(c['losses/r_loss'], axis=0) / test_total_batch
                kl_loss += np.average(c['losses/kl_loss'], axis=0) / test_total_batch
                kl_bits += np.average(c['losses/kl_bits'], axis=0) / test_total_batch
                kl_beta += c['losses/kl_beta'] / test_total_batch
            acc_index = [k for k in m.keys() if ('/metrics/accuracy' in k and 'segment/' in k)]
            print('\nTest Loss: %4.3f, r_loss: %4.3f, kl_losss: %4.3f, kl_bits: %4.3f, kl_beta: %.2f '
                  % (loss, r_loss, kl_loss, kl_bits, kl_beta), flush=True)
            print('Test Accuracy: %f\n' % np.average([m[i][1] for i in acc_index], axis=0))

            if (epoch+1) % 50 == 0:
                # save checkpoint
                final_path = checkpoint_path + time_path[:-1] + '-%04d/' % (epoch+1)
                if not gfile.Exists(final_path):
                    gfile.MakeDirs(final_path)
                saver.save(sess, final_path, global_step=global_step)
                print('\nEpoch %d Model saved in file : %s' % ((epoch+1), final_path))

                sampling, _ = model.sample(n=num_samples, max_length=midi_len, temperature=randomness)
                samples = sess.run(sampling, feed_dict={is_training: False, dropout_keep_prob: 1.0})
                sample_path_ = sample_path[:-1] + time_path[-11:-1] + '-%04d/' % (epoch+1)
                for i, sample in enumerate(samples):
                    print(i)
                    pianoroll = []
                    for timestep, probs in enumerate(sample):
                        pianoroll.append(np.random.multinomial(1, probs))
                    pianoroll = np.array(pianoroll)

                    if pianoroll_path == 'pianoroll_compact.pkl':
                        pianoroll = np.pad(pianoroll[:,1:], ((0, 0), (55,127-88)), 'constant')

                    if not gfile.Exists(sample_path_):
                        gfile.MakeDirs(sample_path_)
                    pianoroll_to_midi(pianoroll, os.path.join(sample_path_, '%d.mid' % (i + 1)))
                print('\nEpoch %d Model\'s %d samples saved in  : %s\n' % ((epoch+1), num_samples, sample_path_), flush=True)

        print('\n2. Training finished!')
        #
        # # save checkpoint
        # time_path = '/run-%02d%02d-%02d%02d/' % tuple(localtime(time()))[1:5]
        # final_path = checkpoint_path + time_path
        # if not gfile.Exists(final_path):
        #     gfile.MakeDirs(final_path)
        # saver.save(sess, final_path, global_step=global_step)
        # print('\n3. Model saved in file : %s' % final_path)
        #
        # sampling, _ = model.sample(n=num_samples, max_length=midi_len, temperature=randomness)
        # samples = sess.run(sampling, feed_dict={is_training: False})
        # sample_path = sample_path[:-1] + time_path[-11:]
        # for i, sample in enumerate(samples):
        #     print(i)
        #     pianoroll = []
        #     for timestep, probs in enumerate(sample):
        #         pianoroll.append(np.random.multinomial(1, probs))
        #     if not gfile.Exists(sample_path):
        #         os.mkdir(sample_path)
        #     pianoroll_to_midi(np.array(pianoroll), os.path.join(sample_path, '%d.mid' % (i + 1)))
        # print('\n4. %d samples saved in  : %s' % (num_samples, sample_path), flush=True)


    else:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Load model from : %s' % checkpoint_path)
            # saver.restore(sess, ckpt.model_checkpoint_path)

        print('1. Checkout Test Accuracy')
        test_total_batch = int((pianoroll_list.shape[0] - split) / batch_size)
        sess.run(test_init)
        loss, r_loss, kl_loss, kl_bits, kl_beta = 0.0, 0.0, 0.0, 0.0, 0.0
        for _ in range(test_total_batch):
            m, c = sess.run([test_metric, test_loss], feed_dict={dropout_keep_prob: 1.0, is_training: False})
            loss += c['loss'] / test_total_batch
            r_loss += np.average(c['losses/r_loss'], axis=0) / test_total_batch
            kl_loss += np.average(c['losses/kl_loss'], axis=0) / test_total_batch
            kl_bits += np.average(c['losses/kl_bits'], axis=0) / test_total_batch
            kl_beta += c['losses/kl_beta'] / test_total_batch
        acc_index = [k for k in m.keys() if ('/metrics/accuracy' in k and 'segment/' in k)]
        print('\nTest Loss: %4.3f, r_loss: %4.3f, kl_losss: %4.3f, kl_bits: %4.3f, kl_beta: %.2f '
              % (loss, r_loss, kl_loss, kl_bits, kl_beta), flush=True)
        print('Test Accuracy: %f\n' % np.average([m[i][1] for i in acc_index], axis=0))


        sampling, _ = model.sample(n=num_samples, max_length=midi_len, temperature=randomness)
        samples = sess.run(sampling, feed_dict={is_training: False, dropout_keep_prob: 1.0})
        sample_path = sample_path[:-1] + checkpoint_path[-11:]

        for i, sample in enumerate(samples):
            print(i)
            pianoroll = []
            for timestep, probs in enumerate(sample):
                pianoroll.append(np.random.multinomial(1, probs))
            if not gfile.Exists(sample_path):
                os.mkdir(sample_path)
            pianoroll_to_midi(np.array(pianoroll), os.path.join(sample_path, '%d.mid' % (i + 1)))
        print('\n2. %d samples saved in  : %s' % (num_samples, sample_path), flush=True)

