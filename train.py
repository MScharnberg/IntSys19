"""IntSys19

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/MScharnberg/IntSys19/blob/master/Train.ipynb

# Training

Intelligent Systems

---

[@mats.scharnberg](mailto:mats.scharnberg@study.hs-duesseldorf.de)

[@christoph.schneider](mailto:christoph.schneider@study.hs-duesseldorf.de)

[@tobias.vossen](mailto:tobias.vossen@study.hs-duesseldorf.de)

## Contents

* Setup
* Data
* Model
* Deployment

## Setup
"""

# Showcase this Notebook?
_SHOWCASE = True #@param ["True", "False"] {type:"raw"}

"""### Requirements"""

import datetime
import os
from timeit import default_timer as timer

import numpy as np
print('NumPy version:', np.__version__)

from matplotlib import pyplot as plt

from tensorboard import version
print('TensorBoard version:', version.VERSION)
from tensorboard.plugins.hparams import api as hp

import tensorflow as tf
print('TensorFlow version:', tf.__version__)
from tensorflow import keras
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, ReLU

import tensorflow_addons as tfa
print('TensorFlow Addons version:', tfa.__version__)
import tensorflow_datasets as tfds
print('TensorFlow Datasets version:', tfds.__version__)

"""### Utils"""

def check_env():
  """Check hardware accelerator for training"""

  if not tf.test.gpu_device_name():
    print('WARNING: Running on CPU. Training may take a while...\n')
  else:
    print('WARNING: Running on GPU. Ressources may be temporarly blocked...\n')

def write_config(parameter, metrics):
    """Write config

    Args:
        parameter: Dict
        metrics: Dict
    """

    with tf.summary.create_file_writer('./logs/hparams').as_default():
        hp.hparams_config(
            hparams=list(parameter.values()),
            metrics=list(metrics.values()),
        )

def get_metrics():
    """Metrics
    
    Returns:
        metrics: Dict
    """

    metrics = {
        'loss/fake_gen' : hp.Metric('loss/fake_gen', display_name='Fake_Gen', description='Loss of generator'),
        'loss/rec_gen' : hp.Metric('loss/rec_gen', display_name='Rec_Gen', description='Loss of generator'),
        'loss/gen' : hp.Metric('loss/gen', display_name='Gen', description='Loss of generator'),
        'loss/fake_disc' : hp.Metric('loss/fake_disc', display_name='FakeDisc', description='Loss of discriminator'),
        'loss/real_disc' : hp.Metric('loss/real_disc', display_name='RealDisc', description='Loss of discriminator'),
        'loss/disc' : hp.Metric('loss/disc', display_name='Disc', description='Loss of discriminator'),
        'eval/kld' : hp.Metric('eval/kld', display_name='KLD', description='Kullback–Leibler divergence'),
        'eval/mse' : hp.Metric('eval/mse', display_name='MSE', description='Mean-squared error'),
    }
    return metrics

def get_metrics_dict(metrics):

  metrics_list = []
  for i in range(6):
    metrics_list.append(tf.keras.metrics.Mean())

  metrics_list.append(tf.keras.metrics.KLDivergence())
  metrics_list.append(tf.keras.metrics.MeanSquaredError())
  metrics_dict = dict(zip(list(metrics.keys()), metrics_list))

  return metrics_dict

def get_parameter_dict(parameter):

  parameter_list = []
  for key, value in parameter.items():
    for parameter_value in value.domain.values:
        parameter_list.append(parameter_value)
        break

  parameter_dict = dict(zip(list(parameter.keys()), parameter_list))
  
  return parameter_dict

"""## Data

*   Constants
*   Choose dataset
*   Load dataset
*   Explore dataset
*   Preprocess dataset
*   Visualize dataset

### Constants
"""

_BS = 64 #@param {type:"slider", min:32, max:128, step:32}
_DIM = 28 #@param ["28"] {type:"raw"}
_DS = 'mnist'
_INPUT = 196 # Noise input
_SHAPE = (_DIM, _DIM, 1) # Image shape
_SIZE = '[:100%]' # Complete dataset size

if _SHOWCASE:
  _SIZE = '[:20%]' # Decrease dataset size for showcase

"""### Choose dataset"""

datasets = tfds.list_builders()
print('TF datasets available:', len(datasets))

"""### Load dataset"""

def load_dataset(dataset):
  """Load dataset by means of TFDS (TensorFlow Datasets)
  
  Args:
    dataset: String
  
  Returns:
    train: tf.data.Dataset
  """

  (train, _), info = tfds.load(dataset,
                            split=['train'+_SIZE, 'test'+_SIZE],
                            as_supervised=True,
                            with_info=True)

  print('Description:', info.description)
  print('Source:', info.homepage)
  print('Total number of training examples:', info.splits['train'].num_examples)
  return train

train_dataset = load_dataset(_DS)

"""### Explore dataset"""

def balance(title, *argv):
  """Balance of datasets
  
  Args:
    title: List of Strings
  """

  fig = plt.figure(figsize=(16, 8))

  for num, dataset in enumerate(argv):
    size = tf.data.experimental.cardinality(dataset)

    for image, label in dataset.batch(size).take(-1):
      y, idx, count = tf.unique_with_counts(label)
      fig.add_subplot(1, len(argv), num+1)
      plt.pie(count, autopct='%1.1f%%')
      plt.legend(labels=y.numpy())
      plt.title(title[num] + ' label distribution of ' + str(size.numpy()) + ' examples')
      
  plt.show()

balance(['Train'], train_dataset)

"""### Preprocess dataset"""

def normalize(image, label):
  """Normalize dataset 
  
  Normalize:
    Cast -> Normalize

  Args:
    image: tf.Tensor as Integer
    label: tf.Tensor as Integer
  
  Returns:
    image: tf.Tensor as Float
    label: tf.Tensor as Float
    noise: tf.Tensor as Float
  """

  image = tf.cast(image, tf.float32)
  image = (image - 127.5) / 127.5 # -1...1
  
  label = tf.cast(label, tf.float32) # (1)
  label = tf.expand_dims(label, -1)
  
  noise = tf.random.normal((_INPUT, ))

  return image, label, noise

def preprocess(dataset, shuffle=True, batch=True, prefetch=False):
  """Preprocess dataset

  Preprocess: 
    Normalize -> Shuffle -> Batch -> Prefetch
  
  Args:
    dataset: tf.data.Dataset
    shuffle: boolean
    batch: boolean
    prefetch: boolean
  
  Returns:
    dataset: tf.data.Dataset
  """

  dataset = dataset.map(normalize)
  if shuffle: dataset = dataset.shuffle(tf.data.experimental.cardinality(dataset))
  if batch: dataset = dataset.batch(_BS, drop_remainder=True)
  if prefetch: dataset = dataset.prefetch(1)
  
  return dataset

train_dataset = preprocess(train_dataset)

"""### Visualize dataset"""

def visualize(dataset):
  """Visualize dataset
  
  Args:
    dataset: tf.data.Dataset
  """

  dataset = dataset.unbatch()
  fig = plt.figure(figsize=(8, 8))
  i = 0
  for image, label, noise in dataset.take(16):
    fig.add_subplot(4, 4, i+1)
    plt.imshow(tf.squeeze(image), cmap='gray')
    plt.title(int(label.numpy()))
    plt.axis('off')
    i = i+1

  plt.suptitle('Real data instances')
  plt.show()

visualize(train_dataset)

"""## Model

*   Parameters
*   Define layers
*   Define model
*   Explore model
*   Compile model
*   Train model
*   Evaluate model
"""

_EPOCHS = 100 #@param {type:"slider", min:10, max:100, step:10}
_DEPTH = 5 # Model depth

if _SHOWCASE:
  _EPOCHS = 10  # Decrease training epochs for showcase

"""### Parameters"""

def get_parameter():
  """Setup parameter
  
  Returns:
    Dict 
  """

  parameter = {
    'model/act' : hp.HParam('model/act', hp.Discrete(['relu', 'lrelu']), display_name='Activation', description='Layer activation'),
    'model/init' : hp.HParam('model/init', hp.Discrete(['normal', 'xavier']), display_name='Initialization', description='Weight initialization'),
    'model/lrd' : hp.HParam('model/lrd', hp.Discrete([1e-4]), display_name='LR', description='Discriminator learning rate'),
    'model/lrg' : hp.HParam('model/lrg', hp.Discrete([2e-4]), display_name='LR', description='Generator learning rate'),
    'model/opt' : hp.HParam('model/opt', hp.Discrete(['adam', 'sgd']), display_name='Optimizer', description='Optimizer algorithm'),
    'model/norm' : hp.HParam('model/norm', hp.Discrete(['batch', 'group']), display_name='Normalization', description='Layer normalization'),
  }
  return parameter

"""### Define layers"""

def get_activation(activation='relu'):
  """Activation layer
  
  Args:
    avtivation: String

  Returns:
    keras.layers.Layer
  """
  
  if activation == 'lrelu': return keras.layers.LeakyReLU()
  else: return keras.layers.ReLU()

def get_normalization(normalization='batch'):
  """Normalization layer
  
  Args:
    normalization: String

  Returns:
    keras.layers.Layer
  """

  if normalization == 'group': return tfa.layers.GroupNormalization()
  else: return BatchNormalization()

def get_initializer(initializer='normal'):
  """Initializer object
  
  Args:
    initializer: String

  Returns:
    keras.initializers.Initializer
  """
  
  if initializer == 'xavier': return keras.initializers.GlorotNormal()
  else: return keras.initializers.RandomNormal()

def plot_initializer(*args):
  """Visualize initializer"""

  fig = plt.figure(figsize=(8*len(args), 8))

  i = 1
  for arg in args:

    model = keras.Sequential(Dense(_INPUT, kernel_initializer=get_initializer(arg), input_shape=(1, )))
    fig.add_subplot(1, len(args), i)
    plt.hist(model.layers[0].get_weights()[0][0], bins=32)
    plt.title(arg)
    i += 1

  plt.suptitle('Initializer')
  plt.show()

plot_initializer('normal', 'xavier')

"""### Define model

![GAN](https://github.com/MScharnberg/IntSys19/blob/master/gan.png?raw=1)

Fake data distribution $p_{noise}$

Real data distribution $p_{data}$

---

Generator $G$

Discriminator $D$

---

Noise $z \sim p_{noise}$

Original $x \sim p_{data}$

Copy $x' = G(z)$

Classification $c = D(x \lor x')$
"""

def get_generator(activation='relu', initializer='normal', normalization='batch'):
  """Get Generator

  Args:
    parameter: Dict

  Returns:
    keras.Model
  """
  
  first = keras.Input(shape=(_INPUT, ), name='Input')
  layer = first

  layer = Reshape((7, 7, 4))(layer)
    
  for i in range(_DEPTH):
    if i % 2 == 0:
      layer = Conv2DTranspose(8*2**i, (3, 3), 
        padding='same', use_bias=False,
        kernel_initializer=get_initializer(initializer))(layer)

    elif i % 2 == 1:
      layer = Conv2DTranspose(8*2**i, (3, 3), 
        padding='same', strides=2, use_bias=False,
        kernel_initializer=get_initializer(initializer))(layer)

    layer = get_normalization(normalization)(layer)
    layer = get_activation(activation)(layer)

  last = Conv2DTranspose(1, (3, 3), 
    strides=1, padding='same', use_bias=False,
    activation='tanh')(layer)

  return keras.Model(inputs=first, outputs=last, name='Generator')

def get_discriminator(activation='relu', initializer='normal', normalization='batch'):
  """Get Discriminator

  Args:
    parameter: Dict
             
  Returns:
    keras.Model
  """

  first = keras.Input(shape=_SHAPE, name='Input')
  layer = first
   
  for i in range(_DEPTH):
    if i % 2 == 0:
      layer = Conv2D(8*2**i, (3, 3), 
        padding='same', use_bias=False,
        kernel_initializer=get_initializer(initializer))(layer)

    if i % 2 == 1:
      layer = Conv2D(8*2**i, (3, 3), 
        strides=2, padding='same', use_bias=False,
        kernel_initializer=get_initializer(initializer))(layer)

    layer = get_normalization(normalization)(layer)
    layer = get_activation(activation)(layer)

  last = Conv2D(1, (3, 3), padding='same')(layer)
  return keras.Model(inputs=first, outputs=last, name='Discriminator')

"""### Explore model"""

generator = get_generator()
generator.summary()

discriminator = get_discriminator()
discriminator.summary()

def explore_model():
  """Explore model
  
  Pipeline: Generator -> Discriminator
    Noise -> Generator -> Image -> Discriminator -> Classification
  """

  # Generator
  noise = tf.random.normal((1, _INPUT))
  image = generator(noise, training=False)

  # Discriminator
  classification = discriminator(image, training=False)
  classification = tf.squeeze(classification) # (1)
  image = tf.squeeze(image) # (28, 28)

  fig = plt.figure(figsize=(8, 4))

  ax1 = fig.add_subplot(1, 2, 1)
  plt.hist(noise, bins=32)
  ax1.title.set_text('Noise')

  ax2 = fig.add_subplot(1, 2, 2)
  ax2.imshow(image, cmap='gray')
  ax2.set_axis_off()
  ax2.title.set_text('Output image')

  plt.suptitle('Generator')
  plt.show()

  fig = plt.figure(figsize=(8, 4))

  ax1 = fig.add_subplot(1, 2, 1)
  ax1.imshow(image, cmap='gray')
  ax1.set_axis_off()
  ax1.title.set_text('Input image')

  ax2 = fig.add_subplot(1, 2, 2)
  img = ax2.imshow(classification, vmin=-20, vmax=20, cmap='RdBu_r')
  cbar = fig.colorbar(img)
  ax2.title.set_text('Output classification')

  plt.suptitle('Discriminator')
  plt.show()

explore_model()

"""### Compile model

* Loss
   * Generator loss
   * Discriminator loss
* Optimizer
  * [Adam](https://arxiv.org/abs/1412.6980) (Adaptive Moment Estimation)
  * [SGD](#) (Stochastic Gradient Descent)
"""

def generator_loss(fake_output, real_images, fake_images):
  """Generator loss
  
  Args:
    fake_output: tf.Tensor as Float
    real_images: tf.Tensor as Float
    fake_images: tf.Tensor as Float

  Returns:
    generator_loss: tf.Tensor as Float
    reconstruction_loss: tf.Tensor as Float
  """

  fake_loss = loss_fn(tf.ones_like(fake_output), fake_output)
  reconstruction_loss = tf.reduce_mean(tf.abs(real_images - fake_images)) # L1

  return fake_loss, reconstruction_loss, fake_loss + reconstruction_loss

def discriminator_loss(real_output, fake_output):
  """Discriminator loss
  
  Args:
    real_output: tf.Tensor as Float
    fake_output: tf.Tensor as Float

  Returns:
    discriminator_loss: tf.Tensor as Float
    loss_quotient: tf.Tensor as Float
  """

  real_loss = loss_fn(tf.ones_like(real_output), real_output)
  fake_loss = loss_fn(tf.zeros_like(fake_output), fake_output)

  return real_loss, fake_loss, real_loss + fake_loss

def optimizer(optimizer='adam', learning_rate=1e-4):
  """Optimizer algorithm
  
  Args:
    algorithm: String
    learning_rate: Float
  
  Returns:
    keras.optimizers.Optimizer
  """
  
  if optimizer == 'sgd': return tf.keras.optimizers.SGD()
  else: return tf.keras.optimizers.Adam()

"""### Train model

#### Train step
"""

def train_step(real_images, noise, metrics):
  """Train step
  
  Step:
    Generate fake images -> Update evaluation metrics -> 
    Classify fake/real images -> Compute losses -> 
    Update loss metrics -> Compute gradients -> Apply gradients

  Args:
    real_images: tf.Tensor as Float
    noise: tf.Tensor as Float
    metrics: Dict
  """
  
  with tf.GradientTape(persistent=True) as tape:

    fake_images = generator(noise, training=True)
    
    real_output = discriminator(real_images, training=True)
    fake_output = discriminator(fake_images, training=True)
    metrics['eval/kld'].update_state(real_images, fake_images)
    metrics['eval/mse'].update_state(real_images, fake_images)

    fake_gen, rec_gen, gen = generator_loss(fake_output, real_images, fake_images)
    metrics['loss/fake_gen'].update_state(fake_gen)
    metrics['loss/rec_gen'].update_state(rec_gen)
    metrics['loss/gen'].update_state(gen)

    real_disc, fake_disc, disc = discriminator_loss(real_output, fake_output)
    metrics['loss/real_disc'].update_state(real_disc)
    metrics['loss/fake_disc'].update_state(fake_disc)
    metrics['loss/disc'].update_state(disc)

  gradients_of_generator = tape.gradient(gen, generator.trainable_variables)
  gradients_of_discriminator = tape.gradient(disc, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

"""#### Train run"""

def train_run(dataset, parameter, metrics, checkpoint, manager):
  """Train run
  
  Train:
    Iterate over epochs -> Iterate over batches -> Train step per batch ->
    Save (and reset) metrics

  Args:
    dataset: tf.data.Dataset
    parameter: Dict
    metrics: Dict
  """

  checkpoint.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print('\nINFO: Restored from {}\n...'.format(manager.latest_checkpoint))
  else:
    print('\nINFO: Initializing training from scratch...\n')

  start = timer()

  for epoch in range(1, _EPOCHS + 1): # Iterate over epochs

    epoch_start = timer()
    output = [epoch, _EPOCHS] 

    for batch in dataset: # Iterate over batches
      train_step(batch[0], batch[2], metrics)

    if (epoch - 1) % 3 == 0:
      fig = plt.figure()
      fake_images = generator(tf.random.normal((4, _INPUT)), training=False)

      for i in range(4):
        fig.add_subplot(1, 4, i+1)
        plt.imshow(tf.squeeze(fake_images[i, ...]), cmap='gray')
        plt.axis('off')

      plt.show()

      print('\t|\tGenerator\t|\tDiscriminator\t|Evaluation\t|Time')
      print('Epoch\t|Fake\tRec\tSum\t|Real\tFake\tSum\t|KLD\tMSE\t|ELA\tETA')

    tf.summary.image('generator/image', generator(tf.random.normal((1, _INPUT)), training=False), step=epoch)

    for key, metric in metrics.items(): # Save scalar metrics
      tf.summary.scalar(key, metric.result(), step=epoch)
      output.append(metric.result())
      metric.reset_states()

    output.append((timer() - start) / 60)
    output.append(((timer() - epoch_start) * (_EPOCHS - epoch)) / 60)
    print('%i/%i\t|%.3f\t%.3f\t%.3f\t|%.3f\t%.3f\t%.3f\t|%.2f\t%.2f\t|%.1f m\t%.1f m' % tuple(output))

    if epoch % 10 == 0: # Save training every 10 epochs
      print('\nINFO: Saved checkpoint after %i epochs as %s\n' % 
           (epoch, manager.save()))

"""#### Train"""

def coldstart():
  check_env()
  if _SHOWCASE:
    print('INFO: Training in showcase mode with reduced dataset size and training epochs...')
  # %rm -rf logs/ # Remove training logs
  # %rm -rf ckpt/ # Remove training logs

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs

coldstart()

parameter = get_parameter()
parameter_dict = get_parameter_dict(parameter)

metrics = get_metrics()
metrics_dict = get_metrics_dict(metrics)

write_config(parameter, metrics)

logdir = './logs'

run = 1
for act in parameter['model/act'].domain.values: # Iterate over layer activations
  for init in parameter['model/init'].domain.values: # Iterate over weigth initializers
    for lrd in parameter['model/lrd'].domain.values: # Iterate over optimizer learning rate
      for lrg in parameter['model/lrg'].domain.values: # Iterate over optimizer learning rate
        for norm in parameter['model/norm'].domain.values: # Iterate over layer normalization
          for opt in parameter['model/opt'].domain.values: # Iterate over optimizer algorithms

            # Write
            parameter_dict['model/act'] = act
            parameter_dict['model/init'] = init
            parameter_dict['model/lrd'] = lrd
            parameter_dict['model/lrg'] = lrg              
            parameter_dict['model/norm'] = norm
            parameter_dict['model/opt'] = opt

            print('\n+++ Run', run, '+++\n')
            print('Act\t|Init\t|LRD\t|LRG\t|Opt\t|norm')
            print('%s\t|%s\t|%.4f\t|%.4f\t|%s\t|%s\n' % tuple(parameter_dict.values()))

            rundir = logdir + '/' + str(run)
            writer = tf.summary.create_file_writer(rundir)

            generator = get_generator(activation=act, initializer=init, normalization=norm)
            discriminator = get_discriminator(activation=act, initializer=init, normalization=norm)

            loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

            discriminator_optimizer = optimizer(optimizer=opt, learning_rate=lrd)
            generator_optimizer = optimizer(optimizer=opt, learning_rate=lrg)

            ckptdir = './ckpt/'
            checkpoint = tf.train.Checkpoint(step=tf.Variable(1), 
                                             generator_optimizer=generator_optimizer,
                                             discriminator_optimizer=discriminator_optimizer,
                                             generator=generator,
                                             discriminator=discriminator)
                      
            manager = tf.train.CheckpointManager(checkpoint, ckptdir, max_to_keep=3)   

            with writer.as_default():
              hp.hparams(parameter_dict)
              train_run(train_dataset, parameter_dict, metrics_dict, checkpoint, manager)

            run += 1

"""### Evaluate model"""

fig = plt.figure(figsize=(16, 32))

for i in range(10):

  # Copy
  noise = tf.random.normal((16, _INPUT, ))
  fake_images = generator(noise, training=False)
  fake_output = tf.reduce_mean(discriminator(fake_images, training=False), [1, 2, 3])

  # Original
  real_images = tf.zeros((1, 28, 28, 1), dtype=float)
  for image, label, noise in train_dataset.unbatch().take(-1):
      if label == i:
        real_images = tf.concat([real_images, tf.expand_dims(image, 0)], 0)
        if len(real_images) >= 17: 
          break
          
  real_images = real_images[1:, :, :]
  real_output = tf.reduce_mean(discriminator(real_images, training=False), [1, 2, 3])

  ax1 = fig.add_subplot(10, 3, 3*i + 1)
  ax1.set_axis_off()
  ax1.imshow(tf.squeeze(fake_images[0]), cmap='gray')

  ax2 = fig.add_subplot(10, 3, 3*i + 2)
  ax2.hist([tf.squeeze(fake_output), tf.squeeze(real_output)])
  ax2.axes.set_xticks([-1.5, 0, 1.5])
  ax2.axes.set_xticklabels(['Copy', 'Unsure', 'Original'])
  ax2.legend(['Fake', 'Real'])

  ax3 = fig.add_subplot(10, 3, 3*i + 3)
  ax3.set_axis_off()
  ax3.imshow(tf.squeeze(real_images[0], -1), cmap='gray')

plt.suptitle('Discriminator classification')
plt.show()

"""## Deployment

*   Use model
*   Export model
*   Export metrics

### Use model
"""

def generate_multiple():
  """Generate multiple digits

  Args:
    num: Integer
  """

  fig = plt.figure(figsize=(8, 8))

  for i in range(16):
    fig.add_subplot(4, 4, i+1)
    noise = tf.random.normal((1, _INPUT))
    image = generator(noise, training=False)
    plt.imshow(tf.squeeze(image), cmap='gray')
    plt.axis('off')

  plt.suptitle('Fake data instances')
  plt.show()

generate_multiple()

"""### Export model"""

export = False #@param ["False", "True"] {type:"raw"}
if export: # Export model
  keras.models.save_model(generator, './generator.h5')
  keras.models.save_model(discriminator, './discriminator.h5')

plot = False #@param ["False", "True"] {type:"raw"}
if plot:
  keras.utils.plot_model(generator, to_file='generator.png', show_shapes=True)
  keras.utils.plot_model(discriminator, to_file='discriminator.png', show_shapes=True)