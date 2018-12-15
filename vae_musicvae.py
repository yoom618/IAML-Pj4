from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import core as layers_core
from tensorflow.python.util import nest

import lstm_utils

ds = tfp.distributions


class MusicVAE(object):
  """Music Variational Autoencoder."""

  def __init__(self, encoder, decoder):
    """Initializer for a MusicVAE model.

    Args:
      encoder: A BaseEncoder implementation class to use.
      decoder: A BaseDecoder implementation class to use.
    """
    self._encoder = encoder
    self._decoder = decoder


  def build(self, hparams, output_depth, is_training):
    """Builds encoder and decoder.

    Must be called within a graph.

    Args:
      hparams: An HParams object containing model hyperparameters. See
          `get_default_hparams` below for required values.
      output_depth: Size of final output dimension.
      is_training: Whether or not the model will be used for training.
    """
    tf.logging.info('Building MusicVAE model with %s, %s, and hparams:\n%s',
                    self.encoder.__class__.__name__,
                    self.decoder.__class__.__name__,
                    hparams.values())
    self.global_step = tf.train.get_or_create_global_step()
    self._hparams = hparams
    self._encoder.build(hparams, is_training)
    self._decoder.build(hparams, output_depth, is_training)

  @property
  def encoder(self):
    return self._encoder

  @property
  def decoder(self):
    return self._decoder

  @property
  def hparams(self):
    return self._hparams

  def encode(self, sequence, sequence_length, control_sequence=None):
    """Encodes input sequences into a MultivariateNormalDiag distribution.

    Args:
      sequence: A Tensor with shape `[num_sequences, max_length, input_depth]`
          containing the sequences to encode.
      sequence_length: The length of each sequence in the `sequence` Tensor.
      control_sequence: (Optional) A Tensor with shape
          `[num_sequences, max_length, control_depth]` containing control
          sequences on which to condition. These will be concatenated depthwise
          to the input sequences.

    Returns:
      A tfp.distributions.MultivariateNormalDiag representing the posterior
      distribution for each sequence.
    """
    hparams = self.hparams
    z_size = hparams.z_size

    sequence = tf.to_float(sequence)
    if control_sequence is not None:
      control_sequence = tf.to_float(control_sequence)
      sequence = tf.concat([sequence, control_sequence], axis=-1)
    encoder_output = self.encoder.encode(sequence, sequence_length)

    mu = tf.layers.dense(
        encoder_output,
        z_size,
        name='encoder/mu',
        kernel_initializer=tf.random_normal_initializer(stddev=0.001), reuse=tf.AUTO_REUSE)
    sigma = tf.layers.dense(
        encoder_output,
        z_size,
        activation=tf.nn.softplus,
        name='encoder/sigma',
        kernel_initializer=tf.random_normal_initializer(stddev=0.001), reuse=tf.AUTO_REUSE)

    return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

  def _compute_model_loss(
      self, input_sequence, output_sequence, sequence_length, control_sequence):
    """Builds a model with loss for train/eval."""
    hparams = self.hparams
    batch_size = hparams.batch_size
    # batch_size = tf.shape(input_sequence)[0]
    input_sequence = tf.to_float(input_sequence)
    output_sequence = tf.to_float(output_sequence)

    max_seq_len = tf.minimum(tf.shape(output_sequence)[1], hparams.max_seq_len)

    input_sequence = input_sequence[:, :max_seq_len]

    if control_sequence is not None:
      control_depth = control_sequence.shape[-1]
      control_sequence = tf.to_float(control_sequence)
      control_sequence = control_sequence[:, :max_seq_len]
      # Shouldn't be necessary, but the slice loses shape information when
      # control depth is zero.
      control_sequence.set_shape([batch_size, None, control_depth])

    # The target/expected outputs.
    x_target = output_sequence[:, :max_seq_len]
    # Inputs to be fed to decoder, including zero padding for the initial input.
    x_input = tf.pad(output_sequence[:, :max_seq_len - 1],
                     [(0, 0), (1, 0), (0, 0)])
    # x_input = output_sequence[:, :max_seq_len]
    x_length = tf.minimum(sequence_length, max_seq_len)

    # Either encode to get `z`, or do unconditional, decoder-only.
    if hparams.z_size:  # vae mode:
      q_z = self.encode(input_sequence, x_length, control_sequence)
      z = q_z.sample()

      # Prior distribution.
      p_z = ds.MultivariateNormalDiag(
          loc=[0.] * hparams.z_size, scale_diag=[1.] * hparams.z_size)

      # KL Divergence (nats)
      kl_div = ds.kl_divergence(q_z, p_z)

      # Concatenate the Z vectors to the inputs at each time step.
    else:  # unconditional, decoder-only generation
      kl_div = tf.zeros([batch_size, 1], dtype=tf.float32)
      z = None

    r_loss, metric_map, r_output = self.decoder.reconstruction_loss(x_input, x_target, x_length, z, control_sequence)

    free_nats = hparams.free_bits * tf.log(2.0)
    kl_cost = tf.maximum(kl_div - free_nats, 0)

    beta = ((1.0 - tf.pow(hparams.beta_rate, tf.to_float(self.global_step)))
            * hparams.max_beta)
    self.loss = tf.reduce_mean(r_loss) + beta * tf.reduce_mean(kl_cost)

    scalars_to_summarize = {
        'loss': self.loss,
        'losses/r_loss': r_loss,
        'losses/kl_loss': kl_cost,
        'losses/kl_bits': kl_div / tf.log(2.0),
        'losses/kl_beta': beta,
    }
    return metric_map, scalars_to_summarize, r_output

  def train(self, input_sequence, output_sequence, sequence_length,
            control_sequence=None):
    """Train on the given sequences, returning an optimizer.

    Args:
      input_sequence: The sequence to be fed to the encoder.
      output_sequence: The sequence expected from the decoder.
      sequence_length: The length of the given sequences (which must be
          identical).
      control_sequence: (Optional) sequence on which to condition. This will be
          concatenated depthwise to the model inputs for both encoding and
          decoding.

    Returns:
      optimizer: A tf.train.Optimizer.
    """

    metric_map, scalars_to_summarize, _ = self._compute_model_loss(
        input_sequence, output_sequence, sequence_length, control_sequence)

    hparams = self.hparams
    lr = ((hparams.learning_rate - hparams.min_learning_rate) *
          tf.pow(hparams.decay_rate, tf.to_float(self.global_step)) +
          hparams.min_learning_rate)

    optimizer = tf.train.AdamOptimizer(lr)

    tf.summary.scalar('learning_rate', lr)
    for n, t in scalars_to_summarize.items():
      tf.summary.scalar(n, tf.reduce_mean(t))

    return metric_map, optimizer, scalars_to_summarize

  def eval(self, input_sequence, output_sequence, sequence_length,
           control_sequence=None):
    """Evaluate on the given sequences, returning metric update ops.

    Args:
      input_sequence: The sequence to be fed to the encoder.
      output_sequence: The sequence expected from the decoder.
      sequence_length: The length of the given sequences (which must be
        identical).
      control_sequence: (Optional) sequence on which to condition the decoder.

    Returns:
      metric_update_ops: tf.metrics update ops.
    """
    metric_map, scalars_to_summarize, reconst_output = self._compute_model_loss(
        input_sequence, output_sequence, sequence_length, control_sequence)

    # for n, t in scalars_to_summarize.items():
    #     metric_map[n] = tf.metrics.mean(t)
    # #
    # metrics_to_values, metrics_to_updates = (
    #     tf.contrib.metrics.aggregate_metric_map(metric_map))
    #
    # for metric_name, metric_value in metrics_to_values.items():
    #   tf.summary.scalar(metric_name, metric_value)

    return metric_map, scalars_to_summarize

  def sample(self, n, max_length=None, z=None, c_input=None, **kwargs):
    """Sample with an optional conditional embedding `z`."""
    if z is not None and z.shape[0].value != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0].value, n))

    if self.hparams.z_size and z is None:
      tf.logging.warning(
          'Sampling from conditional model without `z`. Using random `z`.')
      normal_shape = [n, self.hparams.z_size]
      normal_dist = tfp.distributions.Normal(
          loc=tf.zeros(normal_shape), scale=tf.ones(normal_shape))
      z = normal_dist.sample()

    return self.decoder.sample(n, max_length, z, c_input, **kwargs)


def flatten_maybe_padded_sequences(maybe_padded_sequences, lengths=None):
  """Flattens the batch of sequences, removing padding (if applicable).
  Args:
    maybe_padded_sequences: A tensor of possibly padded sequences to flatten,
        sized `[N, M, ...]` where M = max(lengths).
    lengths: Optional length of each sequence, sized `[N]`. If None, assumes no
        padding.
  Returns:
     flatten_maybe_padded_sequences: The flattened sequence tensor, sized
         `[sum(lengths), ...]`.
  """
  def flatten_unpadded_sequences():
    # The sequences are equal length, so we should just flatten over the first
    # two dimensions.
    return tf.reshape(maybe_padded_sequences,
                      [-1] + maybe_padded_sequences.shape.as_list()[2:])

  if lengths is None:
    return flatten_unpadded_sequences()

  def flatten_padded_sequences():
    indices = tf.where(tf.sequence_mask(lengths))
    return tf.gather_nd(maybe_padded_sequences, indices)

  return tf.cond(
      tf.equal(tf.reduce_min(lengths), tf.shape(maybe_padded_sequences)[1]),
      flatten_unpadded_sequences,
      flatten_padded_sequences)


### ENCODERS
class BaseEncoder(object):
  """Abstract encoder class.

    Implementations must define the following abstract methods:
     -`build`
     -`encode`
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def output_depth(self):
    """Returns the size of the output final dimension."""
    pass

  @abc.abstractmethod
  def build(self, hparams, is_training=True):
    """Builder method for BaseEncoder.

    Args:
      hparams: An HParams object containing model hyperparameters.
      is_training: Whether or not the model is being used for training.
    """
    pass

  @abc.abstractmethod
  def encode(self, sequence, sequence_length):
    """Encodes input sequences into a precursors for latent code `z`.

    Args:
       sequence: Batch of sequences to encode.
       sequence_length: Length of sequences in input batch.

    Returns:
       outputs: Raw outputs to parameterize the prior distribution in
          MusicVae.encode, sized `[batch_size, N]`.
    """
    pass

class BidirectionalLstmEncoder(BaseEncoder):
  """Bidirectional LSTM Encoder."""

  @property
  def output_depth(self):
    if self._use_cudnn:
      return self._cells[0][-1].num_units + self._cells[1][-1].num_units
    return self._cells[0][-1].output_size + self._cells[1][-1].output_size

  def build(self, hparams, is_training=True, name_or_scope='encoder'):
    if hparams.use_cudnn and hparams.residual_decoder:
      raise ValueError('Residual connections not supported in cuDNN.')

    self._is_training = is_training
    self._name_or_scope = name_or_scope
    self._use_cudnn = hparams.use_cudnn

    tf.logging.info('\nEncoder Cells (bidirectional):\n'
                    '  units: %s\n',
                    hparams.enc_rnn_size)

    if isinstance(name_or_scope, tf.VariableScope):
      name = name_or_scope.name
      reuse = name_or_scope.reuse
    else:
      name = name_or_scope
      reuse = None

    cells_fw = []
    cells_bw = []
    for i, layer_size in enumerate(hparams.enc_rnn_size):
      if self._use_cudnn:
        cells_fw.append(lstm_utils.cudnn_lstm_layer(
            [layer_size], hparams.dropout_keep_prob, self._is_training,
            name_or_scope=tf.VariableScope(
                reuse,
                name + '/cell_%d/bidirectional_rnn/fw' % i)))
        cells_bw.append(lstm_utils.cudnn_lstm_layer(
            [layer_size], hparams.dropout_keep_prob, self._is_training,
            name_or_scope=tf.VariableScope(
                reuse,
                name + '/cell_%d/bidirectional_rnn/bw' % i)))
      else:
        cells_fw.append(
            lstm_utils.rnn_cell(
                [layer_size], hparams.dropout_keep_prob,
                hparams.residual_encoder, self._is_training))
        cells_bw.append(
            lstm_utils.rnn_cell(
                [layer_size], hparams.dropout_keep_prob,
                hparams.residual_encoder, self._is_training))

    self._cells = (cells_fw, cells_bw)

  def encode(self, sequence, sequence_length):
    cells_fw, cells_bw = self._cells
    if self._use_cudnn:
      # Implements stacked bidirectional LSTM for variable-length sequences,
      # which are not supported by the CudnnLSTM layer.
      inputs_fw = tf.transpose(sequence, [1, 0, 2])
      for lstm_fw, lstm_bw in zip(cells_fw, cells_bw):
        outputs_fw, _ = lstm_fw(inputs_fw, training=self._is_training)
        inputs_bw = tf.reverse_sequence(
            inputs_fw, sequence_length, seq_axis=0, batch_axis=1)
        outputs_bw, _ = lstm_bw(inputs_bw, training=self._is_training)
        outputs_bw = tf.reverse_sequence(
            outputs_bw, sequence_length, seq_axis=0, batch_axis=1)

        inputs_fw = tf.concat([outputs_fw, outputs_bw], axis=2)

      last_h_fw = lstm_utils.get_final(outputs_fw, sequence_length)
      # outputs_bw has already been reversed, so we can take the first element.
      last_h_bw = outputs_bw[0]

    else:
      _, states_fw, states_bw = rnn.stack_bidirectional_dynamic_rnn(
          cells_fw,
          cells_bw,
          sequence,
          sequence_length=sequence_length,
          time_major=False,
          dtype=tf.float32,
          scope=self._name_or_scope)
      # Note we access the outputs (h) from the states since the backward
      # ouputs are reversed to the input order in the returned outputs.
      last_h_fw = states_fw[-1][-1].h
      last_h_bw = states_bw[-1][-1].h

    return tf.concat([last_h_fw, last_h_bw], 1)

class HierarchicalLstmEncoder(BaseEncoder):
  """Hierarchical LSTM encoder wrapper.

  Input sequences will be split into segments based on the first value of
  `level_lengths` and encoded. At subsequent levels, the embeddings will be
  grouped based on `level_lengths` and encoded until a single embedding is
  produced.

  See the `encode` method for details on the expected arrangement the sequence
  tensors.

  Args:
    core_encoder_cls: A single BaseEncoder class to use for each level of the
      hierarchy.
    level_lengths: A list of the (maximum) lengths of the segments at each
      level of the hierarchy. The product must equal `hparams.max_seq_len`.
  """

  def __init__(self, core_encoder_cls, level_lengths):
    self._core_encoder_cls = core_encoder_cls
    self._level_lengths = level_lengths

  @property
  def output_depth(self):
    return self._hierarchical_encoders[-1][1].output_depth

  @property
  def level_lengths(self):
    return list(self._level_lengths)

  def level(self, l):
    """Returns the BaseEncoder at level `l`."""
    return self._hierarchical_encoders[l][1]

  def build(self, hparams, is_training=True):
    self._total_length = hparams.max_seq_len
    if self._total_length != np.prod(self._level_lengths):
      raise ValueError(
          'The product of the HierarchicalLstmEncoder level lengths (%d) must '
          'equal the padded input sequence length (%d).' % (
              np.prod(self._level_lengths), self._total_length))
    tf.logging.info('\nHierarchical Encoder:\n'
                    '  input length: %d\n'
                    '  level lengths: %s\n',
                    self._total_length,
                    self._level_lengths)
    self._hierarchical_encoders = []
    num_splits = np.prod(self._level_lengths)
    for i, l in enumerate(self._level_lengths):
      num_splits //= l
      tf.logging.info('Level %d splits: %d', i, num_splits)
      h_encoder = self._core_encoder_cls()
      h_encoder.build(
          hparams, is_training,
          name_or_scope=tf.VariableScope(
              tf.AUTO_REUSE, 'encoder/hierarchical_level_%d' % i))
      self._hierarchical_encoders.append((num_splits, h_encoder))
    self.n_batch = hparams.batch_size

  def encode(self, sequence, sequence_length):
    """Hierarchically encodes the input sequences, returning a single embedding.

    Each sequence should be padded per-segment. For example, a sequence with
    three segments [1, 2, 3], [4, 5], [6, 7, 8 ,9] and a `max_seq_len` of 12
    should be input as `sequence = [1, 2, 3, 0, 4, 5, 0, 0, 6, 7, 8, 9]` with
    `sequence_length = [3, 2, 4]`.

    Args:
      sequence: A batch of (padded) sequences, sized
        `[batch_size, max_seq_len, input_depth]`.
      sequence_length: A batch of sequence lengths. May be sized
        `[batch_size, level_lengths[0]]` or `[batch_size]`. If the latter,
        each length must either equal `max_seq_len` or 0. In this case, the
        segment lengths are assumed to be constant and the total length will be
        evenly divided amongst the segments.

    Returns:
      embedding: A batch of embeddings, sized `[batch_size, N]`.
    """
    batch_size = sequence.shape[0].value
    # if batch_size is None:
    #     batch_size = self.n_batch
    sequence_length = lstm_utils.maybe_split_sequence_lengths(
        sequence_length, np.prod(self._level_lengths[1:]),
        self._total_length)

    for level, (num_splits, h_encoder) in enumerate(
        self._hierarchical_encoders):
      split_seqs = tf.split(sequence, num_splits, axis=1)
      # In the first level, we use the input `sequence_lengths`. After that,
      # we use the full embedding sequences.
      sequence_length = (
          sequence_length if level == 0 else
          tf.fill([batch_size, num_splits], split_seqs[0].shape[1]))
      split_lengths = tf.unstack(sequence_length, axis=1)
      embeddings = [
          h_encoder.encode(s, l) for s, l in zip(split_seqs, split_lengths)]
      sequence = tf.stack(embeddings, axis=1)

    with tf.control_dependencies([tf.assert_equal(tf.shape(sequence)[1], 1)]):
      return sequence[:, 0]

### DECODERS
class BaseDecoder(object):
  """Abstract decoder class.

  Implementations must define the following abstract methods:
     -`build`
     -`reconstruction_loss`
     -`sample`
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def build(self, hparams, output_depth, is_training=True):
    """Builder method for BaseDecoder.

    Args:
      hparams: An HParams object containing model hyperparameters.
      output_depth: Size of final output dimension.
      is_training: Whether or not the model is being used for training.
    """
    pass

  @abc.abstractmethod
  def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                          c_input=None):
    """Reconstruction loss calculation.

    Args:
      x_input: Batch of decoder input sequences for teacher forcing, sized
          `[batch_size, max(x_length), output_depth]`.
      x_target: Batch of expected output sequences to compute loss against,
          sized `[batch_size, max(x_length), output_depth]`.
      x_length: Length of input/output sequences, sized `[batch_size]`.
      z: (Optional) Latent vectors. Required if model is conditional. Sized
          `[n, z_size]`.
      c_input: (Optional) Batch of control sequences, sized
          `[batch_size, max(x_length), control_depth]`. Required if conditioning
          on control sequences.

    Returns:
      r_loss: The reconstruction loss for each sequence in the batch.
      metric_map: Map from metric name to tf.metrics return values for logging.
    """
    pass

  @abc.abstractmethod
  def sample(self, n, max_length=None, z=None, c_input=None):
    """Sample from decoder with an optional conditional latent vector `z`.

    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) Scalar maximum sample length to return. Required if
        data representation does not include end tokens.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.

    Returns:
      samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
    """
    pass

class BaseLstmDecoder(BaseDecoder):
  """Abstract LSTM Decoder class.

  Implementations must define the following abstract methods:
      -`_sample`
      -`_flat_reconstruction_loss`
  """

  def build(self, hparams, output_depth, is_training=False):
    if hparams.use_cudnn and hparams.residual_decoder:
      raise ValueError('Residual connections not supported in cuDNN.')

    self._is_training = is_training
    tf.logging.info('\nDecoder Cells:\n'
                    '  units: %s\n',
                    hparams.dec_rnn_size)

    # self._sampling_probability = lstm_utils.get_sampling_probability(
    #     hparams, self._is_training)
    self._sampling_probability = lstm_utils.get_sampling_probability(
        hparams)
    self._output_depth = output_depth
    self._output_layer = layers_core.Dense(
        output_depth, name='output_projection')
    self._dec_cell = lstm_utils.rnn_cell(
        hparams.dec_rnn_size, hparams.dropout_keep_prob,
        hparams.residual_decoder, self._is_training)
    self._cudnn_dec_lstm = lstm_utils.cudnn_lstm_layer(
        hparams.dec_rnn_size, hparams.dropout_keep_prob, self._is_training,
        name_or_scope='decoder') if hparams.use_cudnn else None
    # self.n_batch = hparams.batch_size

  @property
  def state_size(self):
    return self._dec_cell.state_size

  @abc.abstractmethod
  def _sample(self, rnn_output, temperature):
    """Core sampling method for a single time step.

    Args:
      rnn_output: The output from a single timestep of the RNN, sized
          `[batch_size, rnn_output_size]`.
      temperature: A scalar float specifying a sampling temperature.
    Returns:
      A batch of samples from the model.
    """
    pass

  @abc.abstractmethod
  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    """Core loss calculation method for flattened outputs.

    Args:
      flat_x_target: The flattened ground truth vectors, sized
        `[sum(x_length), self._output_depth]`.
      flat_rnn_output: The flattened output from all timeputs of the RNN,
        sized `[sum(x_length), rnn_output_size]`.
    Returns:
      r_loss: The unreduced reconstruction losses, sized `[sum(x_length)]`.
      metric_map: A map of metric names to tuples, each of which contain the
        pair of (value_tensor, update_op) from a tf.metrics streaming metric.
    """
    pass

  def _decode(self, z, helper, input_shape, max_length=None):
    """Decodes the given batch of latent vectors vectors, which may be 0-length.

    Args:
      z: Batch of latent vectors, sized `[batch_size, z_size]`, where `z_size`
        may be 0 for unconditioned decoding.
      helper: A seq2seq.Helper to use. If a TrainingHelper is passed and a
        CudnnLSTM has previously been defined, it will be used instead.
      input_shape: The shape of each model input vector passed to the decoder.
      max_length: (Optional) The maximum iterations to decode.

    Returns:
      results: The LstmDecodeResults.
    """
    initial_state = lstm_utils.initial_cell_state_from_embedding(
        self._dec_cell, z, name='decoder/z_to_initial_state')

    # CudnnLSTM does not support sampling so it can only replace TrainingHelper.
    if  self._cudnn_dec_lstm and type(helper) is seq2seq.TrainingHelper:  # pylint:disable=unidiomatic-typecheck
      rnn_output, _ = self._cudnn_dec_lstm(
          tf.transpose(helper.inputs, [1, 0, 2]),
          initial_state=lstm_utils.state_tuples_to_cudnn_lstm_state(
              initial_state),
          training=self._is_training)
      with tf.variable_scope('decoder'):
        rnn_output = self._output_layer(rnn_output)

      results = lstm_utils.LstmDecodeResults(
          rnn_input=helper.inputs[:, :, :self._output_depth],
          rnn_output=tf.transpose(rnn_output, [1, 0, 2]),
          samples=tf.zeros([tf.shape(z)[0], 0]), ###########################################
          # TODO(adarob): Pass the final state when it is valid (fixed-length).
          final_state=None,
          final_sequence_lengths=helper.sequence_length)
    else:
      if self._cudnn_dec_lstm:
        tf.logging.warning(
            'CudnnLSTM does not support sampling. Using `dynamic_decode` '
            'instead.')
      decoder = lstm_utils.Seq2SeqLstmDecoder(
          self._dec_cell,
          helper,
          initial_state=initial_state,
          input_shape=input_shape,
          output_layer=self._output_layer)
      final_output, final_state, final_lengths = seq2seq.dynamic_decode(
          decoder,
          maximum_iterations=max_length,
          swap_memory=True,
          scope='decoder')
      results = lstm_utils.LstmDecodeResults(
          rnn_input=final_output.rnn_input[:, :, :self._output_depth],
          rnn_output=final_output.rnn_output,
          samples=final_output.sample_id,
          final_state=final_state,
          final_sequence_lengths=final_lengths)

    return results

  def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                          c_input=None):
    """Reconstruction loss calculation.

    Args:
      x_input: Batch of decoder input sequences for teacher forcing, sized
        `[batch_size, max(x_length), output_depth]`.
      x_target: Batch of expected output sequences to compute loss against,
        sized `[batch_size, max(x_length), output_depth]`.
      x_length: Length of input/output sequences, sized `[batch_size]`.
      z: (Optional) Latent vectors. Required if model is conditional. Sized
        `[n, z_size]`.
      c_input: (Optional) Batch of control sequences, sized
          `[batch_size, max(x_length), control_depth]`. Required if conditioning
          on control sequences.

    Returns:
      r_loss: The reconstruction loss for each sequence in the batch.
      metric_map: Map from metric name to tf.metrics return values for logging.
      decode_results: The LstmDecodeResults.
    """

    # batch_size = tf.shape(x_input)[0] ########################
    batch_size = x_input.shape[0].value
    # if batch_size == None:
    #     batch_size = self.n_batch

    has_z = z is not None
    z = tf.zeros([batch_size, 0]) if z is None else z
    repeated_z = tf.tile(
        tf.expand_dims(z, axis=1), [1, tf.shape(x_input)[1], 1])

    has_control = c_input is not None
    if c_input is None:
      c_input = tf.zeros([batch_size, tf.shape(x_input)[1], 0])

    sampling_probability_static = tensor_util.constant_value(
        self._sampling_probability)
    if sampling_probability_static == 0.0:
      # Use teacher forcing.
      x_input = tf.concat([x_input, repeated_z, c_input], axis=2)
      helper = seq2seq.TrainingHelper(x_input, x_length)
    else:
      # Use scheduled sampling.
      if has_z or has_control:
        auxiliary_inputs = tf.zeros([batch_size, tf.shape(x_input)[1], 0])
        if has_z:
          auxiliary_inputs = tf.concat([auxiliary_inputs, repeated_z], axis=2)
        if has_control:
          auxiliary_inputs = tf.concat([auxiliary_inputs, c_input], axis=2)
      else:
        auxiliary_inputs = None
      helper = seq2seq.ScheduledOutputTrainingHelper(
          inputs=x_input,
          sequence_length=x_length,
          auxiliary_inputs=auxiliary_inputs,
          sampling_probability=self._sampling_probability,
          next_inputs_fn=self._sample)

    decode_results = self._decode(
        z, helper=helper, input_shape=helper.inputs.shape[2:])
    flat_x_target = flatten_maybe_padded_sequences(x_target, x_length)
    flat_rnn_output = flatten_maybe_padded_sequences(
        decode_results.rnn_output, x_length)
    r_loss, metric_map = self._flat_reconstruction_loss(
        flat_x_target, flat_rnn_output)

    # Sum loss over sequences.
    cum_x_len = tf.concat([(0,), tf.cumsum(x_length)], axis=0)

    # r_loss = tf.map_fn(lambda i: tf.reduce_sum(r_loss[cum_x_len[i]:cum_x_len[i + 1]]), tf.range(batch_size))
    r_losses = []
    for i in range(batch_size):
      b, e = cum_x_len[i], cum_x_len[i + 1]
      r_losses.append(tf.reduce_sum(r_loss[b:e]))
    r_loss = tf.stack(r_losses)

    return r_loss, metric_map, decode_results

  def sample(self, n, max_length=None, z=None, c_input=None, temperature=1.0,
             start_inputs=None, end_fn=None):
    """Sample from decoder with an optional conditional latent vector `z`.

    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) Scalar maximum sample length to return. Required if
        data representation does not include end tokens.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
      temperature: (Optional) The softmax temperature to use when sampling, if
        applicable.
      start_inputs: (Optional) Initial inputs to use for batch.
        Sized `[n, output_depth]`.
      end_fn: (Optional) A callable that takes a batch of samples (sized
        `[n, output_depth]` and emits a `bool` vector
        shaped `[batch_size]` indicating whether each sample is an end token.
    Returns:
      samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
      final_state: The final states of the decoder.
    Raises:
      ValueError: If `z` is provided and its first dimension does not equal `n`.
    """
    if z is not None and z.shape[0].value != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0].value, n))

    # Use a dummy Z in unconditional case.
    z = tf.zeros((n, 0), tf.float32) if z is None else z

    if c_input is not None:
      # Tile control sequence across samples.
      c_input = tf.tile(tf.expand_dims(c_input, 1), [1, n, 1])

    # If not given, start with zeros.
    start_inputs = start_inputs if start_inputs is not None else tf.zeros(
        [n, self._output_depth], dtype=tf.float32)
    # In the conditional case, also concatenate the Z.
    start_inputs = tf.concat([start_inputs, z], axis=-1)
    if c_input is not None:
      start_inputs = tf.concat([start_inputs, c_input[0]], axis=-1)
    initialize_fn = lambda: (tf.zeros([n], tf.bool), start_inputs)

    sample_fn = lambda time, outputs, state: self._sample(outputs, temperature)
    end_fn = end_fn or (lambda x: False)

    def next_inputs_fn(time, outputs, state, sample_ids):
      del outputs
      finished = end_fn(sample_ids)
      next_inputs = tf.concat([sample_ids, z], axis=-1)
      if c_input is not None:
        next_inputs = tf.concat([next_inputs, c_input[time]], axis=-1)
      return (finished, next_inputs, state)

    sampler = seq2seq.CustomHelper(
        initialize_fn=initialize_fn, sample_fn=sample_fn,
        next_inputs_fn=next_inputs_fn, sample_ids_shape=[self._output_depth],
        sample_ids_dtype=tf.float32)

    decode_results = self._decode(
        z, helper=sampler, input_shape=start_inputs.shape[1:],
        max_length=max_length)

    return decode_results.samples, decode_results

class CategoricalLstmDecoder(BaseLstmDecoder):
  """LSTM decoder with single categorical output."""

  def _flat_reconstruction_loss(self, flat_x_target, flat_rnn_output):
    flat_logits = flat_rnn_output
    flat_truth = tf.argmax(flat_x_target, axis=1)
    flat_predictions = tf.argmax(flat_logits, axis=1)
    r_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=flat_x_target, logits=flat_logits)

    metric_map = {
        'metrics/accuracy':
            tf.metrics.accuracy(flat_truth, flat_predictions),
        'metrics/mean_per_class_accuracy':
            tf.metrics.mean_per_class_accuracy(
                flat_truth, flat_predictions, flat_x_target.shape[-1].value),
    }
    return r_loss, metric_map

  def _sample(self, rnn_output, temperature=1.0):
    sampler = tfp.distributions.OneHotCategorical(
        logits=rnn_output / temperature, dtype=tf.float32)
    return sampler.sample()

  def sample(self, n, max_length=None, z=None, c_input=None, temperature=None,
             start_inputs=None, beam_width=None, end_token=None):
    """Overrides BaseLstmDecoder `sample` method to add optional beam search.

    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) Scalar maximum sample length to return. Required if
        data representation does not include end tokens.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
      temperature: (Optional) The softmax temperature to use when not doing beam
        search. Defaults to 1.0. Ignored when `beam_width` is provided.
      start_inputs: (Optional) Initial inputs to use for batch.
        Sized `[n, output_depth]`.
      beam_width: (Optional) Width of beam to use for beam search. Beam search
        is disabled if not provided.
      end_token: (Optional) Scalar token signaling the end of the sequence to
        use for early stopping.
    Returns:
      samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
      final_state: The final states of the decoder.
    Raises:
      ValueError: If `z` is provided and its first dimension does not equal `n`,
        or if `c_input` is provided under beam search.
    """
    if beam_width is None:
      end_fn = (None if end_token is None else
                lambda x: tf.equal(tf.argmax(x, axis=-1), end_token))
      return super(CategoricalLstmDecoder, self).sample(
          n, max_length, z, c_input, temperature, start_inputs, end_fn)

    # TODO(iansimon): Support conditioning in beam search decoder, which may be
    # awkward as there's no helper.
    if c_input is not None:
      raise ValueError('Control sequence unsupported in beam search.')

    # If `end_token` is not given, use an impossible value.
    end_token = self._output_depth if end_token is None else end_token
    if z is not None and z.shape[0].value != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0].value, n))

    if temperature is not None:
      tf.logging.warning('`temperature` is ignored when using beam search.')
    # Use a dummy Z in unconditional case.
    z = tf.zeros((n, 0), tf.float32) if z is None else z

    # If not given, start with dummy `-1` token and replace with zero vectors in
    # `embedding_fn`.
    start_tokens = (
        tf.argmax(start_inputs, axis=-1, output_type=tf.int32)
        if start_inputs is not None else
        -1 * tf.ones([n], dtype=tf.int32))

    initial_state = lstm_utils.initial_cell_state_from_embedding(
        self._dec_cell, z, name='decoder/z_to_initial_state')
    beam_initial_state = seq2seq.tile_batch(
        initial_state, multiplier=beam_width)

    # Tile `z` across beams.
    beam_z = tf.tile(tf.expand_dims(z, 1), [1, beam_width, 1])

    def embedding_fn(tokens):
      # If tokens are the start_tokens (negative), replace with zero vectors.
      next_inputs = tf.cond(
          tf.less(tokens[0, 0], 0),
          lambda: tf.zeros([n, beam_width, self._output_depth]),
          lambda: tf.one_hot(tokens, self._output_depth))

      # Concatenate `z` to next inputs.
      next_inputs = tf.concat([next_inputs, beam_z], axis=-1)
      return next_inputs

    decoder = seq2seq.BeamSearchDecoder(
        self._dec_cell,
        embedding_fn,
        start_tokens,
        end_token,
        beam_initial_state,
        beam_width,
        output_layer=self._output_layer,
        length_penalty_weight=0.0)

    final_output, final_state, final_lengths = seq2seq.dynamic_decode(
        decoder,
        maximum_iterations=max_length,
        swap_memory=True,
        scope='decoder')

    samples = tf.one_hot(final_output.predicted_ids[:, :, 0],
                         self._output_depth)
    # Rebuild the input by combining the inital input with the sampled output.
    initial_inputs = (
        tf.zeros([n, 1, self._output_depth]) if start_inputs is None else
        tf.expand_dims(start_inputs, axis=1))
    rnn_input = tf.concat([initial_inputs, samples[:, :-1]], axis=1)

    results = lstm_utils.LstmDecodeResults(
        rnn_input=rnn_input,
        rnn_output=None,
        samples=samples,
        final_state=nest.map_structure(
            lambda x: x[:, 0], final_state.cell_state),
        final_sequence_lengths=final_lengths[:, 0])
    return samples, results

class HierarchicalLstmDecoder(BaseDecoder):
  """Hierarchical LSTM decoder."""

  def __init__(self,
               core_decoder,
               level_lengths,
               disable_autoregression=False,
               hierarchical_encoder=None):
    """Initializer for HierarchicalLstmDecoder.

    Hierarchicaly decodes a sequence across time.

    Each sequence is padded per-segment. For example, a sequence with
    three segments [1, 2, 3], [4, 5], [6, 7, 8 ,9] and a `max_seq_len` of 12
    is represented as `sequence = [1, 2, 3, 0, 4, 5, 0, 0, 6, 7, 8, 9]` with
    `sequence_length = [3, 2, 4]`.

    `z` initializes the first level LSTM to produce embeddings used to
    initialize the states of LSTMs at subsequent levels. The lowest-level
    embeddings are then passed to the given `core_decoder` to generate the
    final outputs.

    This decoder has 3 modes for what is used as the inputs to the LSTMs
    (excluding those in the core decoder):
      Autoregressive: (default) The inputs to the level `l` decoder are the
        final states of the level `l+1` decoder.
      Non-autoregressive: (`disable_autoregression=True`) The inputs to the
        hierarchical decoders are 0's.
      Re-encoder: (`hierarchical_encoder` provided) The inputs to the level `l`
        decoder are re-encoded outputs of level `l+1`, using the given encoder's
        matching level.

    Args:
      core_decoder: The BaseDecoder implementation to use at the output level.
      level_lengths: A list of the number of outputs of each level of the
        hierarchy. The final level is the (padded) maximum length. The product
        of the lengths must equal `hparams.max_seq_len`.
      disable_autoregression: Whether to disable the autoregression within the
        hierarchy. May also be a collection of levels on which to disable.
      hierarchical_encoder: (Optional) A HierarchicalLstmEncoder instance to use
        for re-encoding the decoder outputs at each level for use as inputs to
        the next level up in the hierarchy, instead of the final decoder state.
        The encoder level output lengths (except for the final single-output
        level) should be the reverse of `level_output_lengths`.

    Raises:
      ValueError: If `hierarchical_encoder` is given but has incompatible level
        lengths.
    """
    if disable_autoregression is True:
      disable_autoregression = range(len(level_lengths))
    elif disable_autoregression is False:
      disable_autoregression = []
    if (hierarchical_encoder and
        (tuple(hierarchical_encoder.level_lengths[-1::-1]) !=
         tuple(level_lengths))):
      raise ValueError(
          'Incompatible hierarchical encoder level output lengths: ',
          hierarchical_encoder.level_lengths, level_lengths)

    self._core_decoder = core_decoder
    self._level_lengths = level_lengths
    self._disable_autoregression = disable_autoregression
    self._hierarchical_encoder = hierarchical_encoder

  def build(self, hparams, output_depth, is_training):
    self.hparams = hparams
    self._output_depth = output_depth
    self._total_length = hparams.max_seq_len
    if self._total_length != np.prod(self._level_lengths):
      raise ValueError(
          'The product of the HierarchicalLstmDecoder level lengths (%d) must '
          'equal the padded input sequence length (%d).' % (
              np.prod(self._level_lengths), self._total_length))
    tf.logging.info('\nHierarchical Decoder:\n'
                    '  input length: %d\n'
                    '  level output lengths: %s\n',
                    self._total_length,
                    self._level_lengths)

    self._hier_cells = [
        lstm_utils.rnn_cell(
            hparams.dec_rnn_size,
            dropout_keep_prob=hparams.dropout_keep_prob,
            residual=hparams.residual_decoder,
            is_training=is_training)
        for _ in range(len(self._level_lengths))]

    with tf.variable_scope('core_decoder', reuse=tf.AUTO_REUSE):
      self._core_decoder.build(hparams, output_depth, is_training)

  @property
  def state_size(self):
    return self._core_decoder.state_size

  def _merge_decode_results(self, decode_results):
    """Merge across time."""
    assert decode_results
    time_axis = 1
    zipped_results = lstm_utils.LstmDecodeResults(*zip(*decode_results))
    return lstm_utils.LstmDecodeResults(
        rnn_output=(None if zipped_results.rnn_output[0] is None else
                    tf.concat(zipped_results.rnn_output, axis=time_axis)),
        rnn_input=(None if zipped_results.rnn_input[0] is None else
                   tf.concat(zipped_results.rnn_input, axis=time_axis)),
        samples=tf.concat(zipped_results.samples, axis=time_axis),
        final_state=zipped_results.final_state[-1],
        final_sequence_lengths=tf.stack(
            zipped_results.final_sequence_lengths, axis=time_axis))

  def _hierarchical_decode(self, z, base_decode_fn):
    """Depth first decoding from `z`, passing final embeddings to base fn."""
    batch_size = tf.shape(z)[0]
    # Subtract 1 for the core decoder level.
    num_levels = len(self._level_lengths) - 1
    # batch_size = self.hparams.batch_size

    def recursive_decode(initial_input, path=None):
      """Recursive hierarchical decode function."""
      path = path or []
      level = len(path)

      if level == num_levels:
        with tf.variable_scope('core_decoder', reuse=tf.AUTO_REUSE):
          return base_decode_fn(initial_input, path)

      scope = tf.VariableScope(
          tf.AUTO_REUSE, 'decoder/hierarchical_level_%d' % level)
      num_steps = self._level_lengths[level]
      with tf.variable_scope(scope):
        state = lstm_utils.initial_cell_state_from_embedding(
            self._hier_cells[level], initial_input, name='initial_state')
      if level not in self._disable_autoregression:
        # The initial input should be the same size as the tensors returned by
        # next level.
        if self._hierarchical_encoder:
          input_size = self._hierarchical_encoder.level(0).output_depth
        elif level == num_levels - 1:
          input_size = sum(nest.flatten(self._core_decoder.state_size))
        else:
          input_size = sum(nest.flatten(self._hier_cells[level + 1].state_size))
        next_input = tf.zeros([batch_size, input_size])
      lower_level_embeddings = []
      for i in range(num_steps):
        if level in self._disable_autoregression:
          next_input = tf.zeros([batch_size, 1])
        else:
          next_input = tf.concat([next_input, initial_input], axis=1)
        with tf.variable_scope(scope):
          output, state = self._hier_cells[level](next_input, state, scope)
        next_input = recursive_decode(output, path + [i])
        lower_level_embeddings.append(next_input)
      if self._hierarchical_encoder:
        # Return the encoding of the outputs using the appropriate level of the
        # hierarchical encoder.
        enc_level = num_levels - level
        return self._hierarchical_encoder.level(enc_level).encode(
            sequence=tf.stack(lower_level_embeddings, axis=1),
            sequence_length=tf.fill([batch_size], num_steps))
      else:
        # Return the final state.
        return tf.concat(nest.flatten(state), axis=-1)

    return recursive_decode(z)

  def _reshape_to_hierarchy(self, t):
    """Reshapes `t` so that its initial dimensions match the hierarchy."""
    # Exclude the final, core decoder length
    t_shape = t.shape
    t_rank = t.shape.ndims
    # print('t rank : ',t_rank)
    batch_size = t_shape[0]
    if batch_size == None:
        batch_size = self.hparams.batch_size
    if t_rank == 3:
        hier_shape = [batch_size, *self._level_lengths[:-1], -1, *t_shape[2:]]
    elif t_rank == 2:
        hier_shape = [batch_size, *self._level_lengths[:-1]]
    else:
        # We only expect rank-2 for lengths and rank-3 for sequences.
        raise ValueError('Unexpected shape for tensor: %s' % t)

    hier_t = tf.reshape(t, hier_shape)
    # Move the batch dimension to after the hierarchical dimensions.
    num_levels = len(self._level_lengths[:-1])
    perm = list(range(len(hier_shape)))
    perm.insert(num_levels, perm.pop(0))
    return tf.transpose(hier_t, perm)

  def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                          c_input=None):
    """Reconstruction loss calculation.

    Args:
      x_input: Batch of decoder input sequences of concatenated segments for
        teacher forcing, sized `[batch_size, max_seq_len, output_depth]`.
      x_target: Batch of expected output sequences to compute loss against,
        sized `[batch_size, max_seq_len, output_depth]`.
      x_length: Length of input/output sequences, sized
        `[batch_size, level_lengths[0]]` or `[batch_size]`. If the latter,
        each length must either equal `max_seq_len` or 0. In this case, the
        segment lengths are assumed to be constant and the total length will be
        evenly divided amongst the segments.
      z: (Optional) Latent vectors. Required if model is conditional. Sized
        `[n, z_size]`.
      c_input: (Optional) Batch of control sequences, sized
        `[batch_size, max_seq_len, control_depth]`. Required if conditioning on
        control sequences.

    Returns:
      r_loss: The reconstruction loss for each sequence in the batch.
      metric_map: Map from metric name to tf.metrics return values for logging.
      decode_results: The LstmDecodeResults.

    Raises:
      ValueError: If `c_input` is provided in re-encoder mode.
    """
    if self._hierarchical_encoder and c_input is not None:
      raise ValueError(
          'Re-encoder mode unsupported when conditioning on controls.')

    batch_size = x_input.shape[0].value
    # if batch_size == None:
    #     batch_size = self.hparams.batch_size
    x_length = lstm_utils.maybe_split_sequence_lengths(
        x_length, np.prod(self._level_lengths[:-1]), self._total_length)

    hier_input = self._reshape_to_hierarchy(x_input)
    hier_target = self._reshape_to_hierarchy(x_target)
    hier_length = self._reshape_to_hierarchy(x_length)
    hier_control = (self._reshape_to_hierarchy(c_input)
                    if c_input is not None else None)

    loss_outputs = []

    def base_train_fn(embedding, hier_index):
      """Base function for training hierarchical decoder."""
      split_size = self._level_lengths[-1]
      split_input = hier_input[hier_index]
      split_target = hier_target[hier_index]
      split_length = hier_length[hier_index]
      split_control = (hier_control[hier_index]
                       if hier_control is not None else None)

      res = self._core_decoder.reconstruction_loss(
          split_input, split_target, split_length, embedding, split_control)
      loss_outputs.append(res)
      decode_results = res[-1]

      if self._hierarchical_encoder:
        # Get the approximate "sample" from the model.
        # Start with the inputs the RNN saw (excluding the start token).
        samples = decode_results.rnn_input[:, 1:]
        # Pad to be the max length.
        samples = tf.pad(
            samples,
            [(0, 0), (0, split_size - tf.shape(samples)[1]), (0, 0)])
        samples.set_shape([batch_size, split_size, self._output_depth])
        # Set the final value based on the target, since the scheduled sampling
        # helper does not sample the final value.
        samples = lstm_utils.set_final(
            samples,
            split_length,
            lstm_utils.get_final(split_target, split_length, time_major=False),
            time_major=False)
        # Return the re-encoded sample.
        return self._hierarchical_encoder.level(0).encode(
            sequence=samples,
            sequence_length=split_length)
      elif self._disable_autoregression:
        return None
      else:
        return tf.concat(nest.flatten(decode_results.final_state), axis=-1)

    z = tf.zeros([batch_size, 0]) if z is None else z
    self._hierarchical_decode(z, base_train_fn)

    # Accumulate the split sequence losses.
    r_losses, metric_maps, decode_results = zip(*loss_outputs)

    # Merge the metric maps by passing through renamed values and taking the
    # mean across the splits.
    merged_metric_map = {}
    for metric_name in metric_maps[0]:
      metric_values = []
      for i, m in enumerate(metric_maps):
        merged_metric_map['segment/%03d/%s' % (i, metric_name)] = m[metric_name]
        metric_values.append(m[metric_name][0])
      merged_metric_map[metric_name] = (
          tf.reduce_mean(metric_values), tf.no_op())

    return (tf.reduce_sum(r_losses, axis=0),
            merged_metric_map,
            self._merge_decode_results(decode_results))

  def sample(self, n, max_length=None, z=None, c_input=None,
             **core_sampler_kwargs):
    """Sample from decoder with an optional conditional latent vector `z`.

    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) maximum total length of samples. If given, must
        match `hparams.max_seq_len`.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.
      **core_sampler_kwargs: (Optional) Additional keyword arguments to pass to
        core sampler.
    Returns:
      samples: Sampled sequences with concenated, possibly padded segments.
         Sized `[n, max_length, output_depth]`.
      decoder_results: The merged LstmDecodeResults from sampling.
    Raises:
      ValueError: If `z` is provided and its first dimension does not equal `n`,
        or if `c_input` is provided in re-encoder mode.
    """
    if z is not None and z.shape[0].value != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0].value, n))
    z = tf.zeros([n, 0]) if z is None else z

    if self._hierarchical_encoder and c_input is not None:
      raise ValueError(
          'Re-encoder mode unsupported when conditioning on controls.')

    if max_length is not None:
      with tf.control_dependencies([
          tf.assert_equal(
              max_length, self._total_length,
              message='`max_length` must equal `hparams.max_seq_len` if given.')
      ]):
        max_length = tf.identity(max_length)

    if c_input is not None:
      # Reshape control sequence to hierarchy.
      c_input = tf.squeeze(
          self._reshape_to_hierarchy(tf.expand_dims(c_input, 0)),
          axis=len(self._level_lengths) - 1)

    core_max_length = self._level_lengths[-1]
    all_samples = []
    all_decode_results = []

    def base_sample_fn(embedding, hier_index):
      """Base function for sampling hierarchical decoder."""
      samples, decode_results = self._core_decoder.sample(
          n,
          max_length=core_max_length,
          z=embedding,
          c_input=c_input[hier_index] if c_input is not None else None,
          start_inputs=all_samples[-1][:, -1] if all_samples else None,
          **core_sampler_kwargs)
      all_samples.append(samples)
      all_decode_results.append(decode_results)
      if self._hierarchical_encoder:
        return self._hierarchical_encoder.level(0).encode(
            samples,
            decode_results.final_sequence_lengths)
      else:
        return tf.concat(nest.flatten(decode_results.final_state), axis=-1)

    # Populate `all_sample_ids`.
    self._hierarchical_decode(z, base_sample_fn)

    all_samples = tf.concat(
        [tf.pad(s, [(0, 0), (0, core_max_length - tf.shape(s)[1]), (0, 0)])
         for s in all_samples],
        axis=1)
    return all_samples, self._merge_decode_results(all_decode_results)



