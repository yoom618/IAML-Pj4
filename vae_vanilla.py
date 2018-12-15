import tensorflow as tf


def Encoder(x, hidden_dim=1024, latent_dim=32):

    with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
        fc = tf.layers.dense(x, hidden_dim, activation=tf.tanh)
        z_before = tf.layers.dense(fc, latent_dim * 2)

    return z_before


def Decoder(z, data_dim=256*128, hidden_dim=1024):

    with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
        fc = tf.layers.dense(z, hidden_dim, activation=tf.tanh)
        output = tf.layers.dense(fc, data_dim)

    return output


def VAE(x, data_dim=256*128, hidden_dim=1024, latent_dim=32):

    z_before = Encoder(x, hidden_dim=hidden_dim, latent_dim=latent_dim)
    mu = z_before[:, :latent_dim]
    sig = z_before[:, latent_dim:]

    eps = tf.random_normal(tf.shape(mu), dtype=tf.float64, mean=0., stddev=1.0,
                           name='epsilon')
    z = mu + eps * (tf.exp(sig / 2))

    x_reconstructed = Decoder(z, data_dim=data_dim, hidden_dim=hidden_dim)

    return x_reconstructed, mu, sig


def VAE_loss(x_reconstructed, x_target, mu, sig):

    reconst_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_reconstructed,
                                                           labels=x_target)
    reconst_loss = tf.reduce_sum(reconst_loss, 1)

    kld_loss = 1 + sig - tf.square(mu) - tf.exp(sig)
    kld_loss = -0.5 * tf.reduce_sum(kld_loss, 1)

    return tf.reduce_mean(reconst_loss + kld_loss)


