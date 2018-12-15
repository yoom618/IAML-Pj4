import tensorflow as tf
import pickle
from dataloader import *
from vae_vanilla import *
from tensorflow.python.platform import gfile
from time import strftime, localtime, time

# hyperparameters
# TODO : declare additional hyperparameters

# Parameters
learning_rate = 0.001
num_epochs = 10
batch_size = 32
num_samples = 10

# Network Parameters
midi_dim = 256*128
hidden_dim = 1024
latent_dim = 32


# fixed
data_path = 'data/'
sample_path = 'sample/'
if not gfile.Exists(sample_path):
    os.mkdir(sample_path)
pianoroll_path = 'pianoroll.pkl'

# True if you want to train, False if you already trained your model
### TODO : IMPORTANT !!! Please change it to False when you submit your code
### TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : checkpoint/run-0925-0348

# is_train_mode, checkpoint_path = True, 'checkpoint'
is_train_mode, checkpoint_path = False, 'checkpoint/run-1205-1822/'

if not gfile.Exists(checkpoint_path):
    os.mkdir(checkpoint_path)


# if there is no pianoroll file, make it
if gfile.Exists(pianoroll_path):
    print('Load pianoroll file')
    with open(pianoroll_path, 'rb') as f:
        pianoroll_list = pickle.load(f)
else:
    print('Make pianoroll file')
    pianoroll_list = get_pianoroll_list(data_path)
    with open(pianoroll_path, 'wb') as f:
        pickle.dump(pianoroll_list, f)

# split train and test
split = int(len(pianoroll_list)*0.8)

# build dataset
dataset = tf.data.Dataset.from_tensor_slices(pianoroll_list)
train_dataset = dataset.take(split)
test_dataset = dataset.skip(split)

train_dataset = train_dataset.shuffle(buffer_size=split)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(batch_size)
train_dataset = train_dataset.repeat(num_epochs)

test_dataset = test_dataset.shuffle(buffer_size=len(pianoroll_list) - split)
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(batch_size)
test_dataset = test_dataset.repeat(1)

iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_init = iter.make_initializer(train_dataset)
test_init = iter.make_initializer(test_dataset)

# batch of features, batch of labels
X = iter.get_next()


# TODO : build your model here
global_step = tf.Variable(0, trainable=False, name='global_step')
X = tf.reshape(X, [-1, 256*128])

X_reconstructed, mu, sig = VAE(X, data_dim=midi_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
loss = VAE_loss(X_reconstructed, X, mu, sig)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

X = tf.reshape(X, [-1, 256, 128])
X_reconstructed = tf.reshape(X_reconstructed, [-1, 256, 128])
infer = tf.argmax(X_reconstructed, axis=2)
answer = tf.argmax(X, axis=2)

# calculate accuracy
correct_prediction = tf.equal(infer, answer)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# train and evaluate
with tf.Session() as sess:
    saver = tf.train.Saver()

    if is_train_mode:
        sess.run(tf.global_variables_initializer())

        print('Start training')
        train_total_batch = int(split / batch_size)
        total_epoch = 0
        sess.run(train_init)
        for epoch in range(num_epochs):
            # TODO: do train
            print('------------------- epoch:', epoch, ' -------------------')
            for _ in range(train_total_batch):
                c, _, acc = sess.run([loss, optimizer, accuracy])
                print('Step: %5d, ' % sess.run(global_step), ' Cost: %.4f ' % c,
                      ' Accuracy: %.4f ' % acc)

        print('Training finished!')

        # TODO : do accuracy test
        test_total_batch = int((len(pianoroll_list) - split) / batch_size)
        sess.run(test_init)
        acc = 0.0
        for _ in range(test_total_batch):
            acc += sess.run(accuracy)

        print('Test accuracy: %.4f' % (acc/test_total_batch))

        # save checkpoint
        final_path = checkpoint_path + '/run-%02d%02d-%02d%02d/' % tuple(localtime(time()))[1:5]
        if not gfile.Exists(final_path):
            gfile.MakeDirs(final_path)
        saver.save(sess, final_path, global_step=global_step)
        print('Model saved in file : %s' % final_path)

    else:
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Load model from : %s' % checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # TODO : do sampling
        noise_input = tf.placeholder(tf.float64, shape=[None, latent_dim])
        decode_noise = Decoder(noise_input)
        decode_noise = tf.reshape(decode_noise, [-1, 256, 128])
        sampling_outputs = tf.nn.softmax(decode_noise, axis=2)

        noise = np.random.normal(size=(num_samples, latent_dim))
        samples = sess.run(sampling_outputs, feed_dict={noise_input: noise})

        for i, sample in enumerate(samples):
            print(i)
            pianoroll = []
            for timestep, probs in enumerate(sample):
                pianoroll.append(np.random.multinomial(1, probs))

            pianoroll_to_midi(np.array(pianoroll), os.path.join(sample_path, '%d.mid' % (i + 1)))
