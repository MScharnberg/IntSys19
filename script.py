"""# IntSys19

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
* Exit strategy

## Setup

### Magic
"""

"""### Libraries"""

import datetime
import os
from timeit import default_timer as timer

import numpy as np
from matplotlib import pyplot as plt
from tensorboard import version
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

print('Keras version:', keras.__version__)
print('NumPy version:', np.__version__)
print('TensorBoard version:', version.VERSION)
print('TensorFlow version:', tf.__version__)
print('TensorFlow Datasets version:', tfds.__version__)

"""### Utils"""

def check_env():
  """Check hardware accelerator for training"""

  if not tf.test.gpu_device_name():
    print('WARNING: Running on CPU. Training may take a while...\n')
  else:
    print('WARNING: Running on GPU. Ressources may be temporarly blocked...\n')

def get_logs():
  """Setup logging"""

  logdir = os.path.join('logs', datetime.datetime.now().strftime('%Y-%m-%d'))
  print('INFO: Logs will be written to', logdir, '\n')
  return tf.summary.create_file_writer(logdir)

def get_metrics():
  """Setup metrics"""

  metrics = {
    'loss/generator' : keras.metrics.Mean(),
    'loss/discriminator' : keras.metrics.Mean(),
    'loss/realVSfake' : keras.metrics.Mean(),
    'eval/kld' : keras.metrics.KLDivergence(),
    'eval/mse' : keras.metrics.MeanSquaredError()
  }
  print('INFO:', len(metrics), 'metrics to measure including...')
  for key in metrics.keys():
    print(key)
  return metrics

"""## Data

*   Choose dataset
*   Load dataset
*   Explore dataset
*   Preprocess dataset
*   Visualize dataset
"""

# Global data parameter
_BS = 32 #@param {type:"slider", min:32, max:128, step:32}
_DIM = 28 #@param ["28"] {type:"raw"}
_SHAPE = (_DIM, _DIM, 1) # Image shape

# Local data parameter
ds = 'mnist' # String identifier for dataset

_DATA = {
    'Batch size' : _BS,
    'Image dimension' : _DIM,
    'Image shape' : _SHAPE
}

"""### Choose dataset"""

# To do

"""### Load dataset"""

def load_dataset(ds):
  """Load dataset by means of TFDS (TensorFlow Datasets)
  
  Args:
    dataset: str
  
  Returns:
    train: tf.data.Dataset
    test: tf.data.Dataset
  """

  (train, test) = tfds.load(ds,
                            split=['train[:10%]', 'test[:10%]'],
                            shuffle_files=True,
                            as_supervised=True)
  
  return train, test

train, test = load_dataset(ds)

"""### Explore dataset"""

# To do

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
    label: tf.Tensor as Integer
  """

  image = tf.cast(image, tf.float32) # 0...255
  image = (image - 127.5) / 127.5 # -1...1
  return image, label

def preprocess(ds, shuffle=True, batch=True):
  """Preprocess dataset

  Preprocess: 
    Normalize -> Shuffle -> Batch
  
  Args:
    ds: tf.data.Dataset
    shuffle: boolean
    batch: boolean
  
  Returns:
    ds: tf.data.Dataset
  """

  ds = ds.map(normalize)
  if shuffle: ds = ds.shuffle(shuffle_size)
  if batch: ds = ds.batch(_BS, drop_remainder=True)
  return ds

train = preprocess(train, shuffle=False)
test = preprocess(test, shuffle=False)

"""### Visualize dataset"""

def visualize(ds):
  """Visualize dataset
  
  Args:
    ds: tf.data.Dataset
  
  Returns:
    fig: plt.figure
  """

  ds = ds.unbatch()
  fig = plt.figure(figsize=(16, 16))
  i = 0
  for example in ds.take(36):
    fig.add_subplot(6, 6, i+1)
    plt.imshow(tf.squeeze(example[0], -1), cmap='gray')
    plt.axis('off')
    i = i+1
  plt.show()
  return fig

visualize(train)

"""## Model

*   Choose model
*   Compile model
*   Train model
*   Evaluate model
"""

# Global model parameter
_EPOCHS = 40 #@param {type:"slider", min:10, max:100, step:10}
_LR = 1e-3 # Learning rate

_MODEL = {
    'Epochs' : _EPOCHS,
    'Learning rate' : _LR
}

"""### Choose model"""

def get_generator():
  """ Generator as faking model
  
  Architecture: Encoder-Decoder
    Input -> Dense -> Reshape -> Convolution -> Normalization -> Activation ->
    Inverse Convolution
              
  Returns:
    keras.Model
  """

  first = keras.Input(shape=(196, ))
  layer = keras.layers.Dense(196)(first)
  layer = keras.layers.Reshape((7, 7, 4))(layer)

  for i in range(2):
    layer = keras.layers.Conv2DTranspose(8*2**i, (3, 3), strides=2, padding='same')(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.LeakyReLU()(layer)
  
  last = keras.layers.Conv2DTranspose(1, (3, 3), strides=1, padding='same', activation='tanh')(layer)

  return keras.Model(inputs=first, outputs=last, name='Generator')

def get_discriminator():
  """ Discriminator as expertise model
  
  Architecture: Encoder
    Input -> Convolution -> Normalization -> Activation -> Flatten -> Dense
              
  Returns:
    keras.Model
  """

  first = keras.Input(shape=_SHAPE)
  layer = first

  for i in reversed(range(2)):
    layer = keras.layers.Conv2D(8*2**i, (3, 3), strides=2, padding='same')(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.LeakyReLU()(layer)

  layer = keras.layers.Flatten()(layer)
  layer = keras.layers.Dense(98)(layer)
  last = keras.layers.Dense(1)(layer)

  return keras.Model(inputs=first, outputs=last, name='Discriminator')

generator = get_generator()
discriminator = get_discriminator()

"""### Explore model"""

def predict(num):
  
  noise = tf.random.normal([num, 196]) # (num, 196)
  prediction = generator(noise, training=False) # (num, 28, 28, 1)
  prediction = tf.squeeze(prediction, 0) # (28, 28, 1)
  return noise, tf.squeeze(prediction, -1)

def explore_model(gen=True, disc=True):

  noise, prediction = predict(1)

  if gen:
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.hist(noise)
    ax1.title.set_text('Input (Noise)')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.imshow(prediction, cmap='gray')
    plt.axis('off')
    ax2.title.set_text('Output (Digit)')
    plt.suptitle('Generator')
    plt.show()

  if disc:
    classification = discriminator(tf.expand_dims(tf.expand_dims(prediction, 0), -1), training=False)
    idx = tf.squeeze(classification, 0)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow(prediction, cmap='gray')
    plt.axis('off')
    ax1.title.set_text('Input (Digit)')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot([idx, idx], [0, 1], 'r-', label='Classification')
    plt.legend()
    ax2.title.set_text('Output (Classification)')
    ax2.axes.set_xticks([-1, 0, 1])
    ax2.axes.set_xticklabels(['Copy', 'Unsure', 'Original'])
    plt.suptitle('Discriminator')
    plt.show()

explore_model()

generator.summary()

discriminator.summary()

"""### Compile model"""

def objective():
  """Binary crossentropy objective function
  
  Returns:
    keras.losses 
  """

  return keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
  """Generator loss
  
  Returns:
    generator_loss: tf.Tensor as Float
  """

  generator_loss = loss_fn(tf.ones_like(fake_output), fake_output)
  return generator_loss

def discriminator_loss(real_output, fake_output):
  """Discriminator loss
  
  Returns:
    discriminator_loss: tf.Tensor as Float
    loss_quotient: tf.Tensor as Float
  """

  real_loss = loss_fn(tf.ones_like(real_output), real_output)
  fake_loss = loss_fn(tf.zeros_like(fake_output), fake_output)
  discriminator_loss = real_loss + fake_loss
  loss_quotient = real_loss / fake_loss
  return discriminator_loss, loss_quotient

def optimizer(learning_rate):
  """Adam optimizer with exponential decay, as learning rate decreases over time
  
  Args:
    learning_rate: tf.float32
  
  Returns:
    keras.optimizer
  """
  
  schedule = keras.optimizers.schedules.ExponentialDecay(
      learning_rate,
      decay_steps=10,
      decay_rate=0.96)

  return tf.keras.optimizers.Adam(schedule)

loss_fn = objective()

"""### Train model"""

def train_step(images, noise, metrics):
  """Train step
  
  Step:
    Generate fake images -> Update evaluation metrics -> Let Discriminator
    classify -> Compute losses -> Update loss metrics -> Compute gradients ->
    Apply gradients

  Args:
    images: tf.Tensor as Float
      Real data distribution
    noise: tf.Tensor as Float
      Imaginary data distribution
  """

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

    generated_images = generator(noise, training=True)

    metrics['eval/kld'].update_state(images, generated_images)
    metrics['eval/mse'].update_state(images, generated_images)
    
    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    metrics['loss/generator'].update_state(gen_loss)
    disc_loss, realVSfake = discriminator_loss(real_output, fake_output)
    metrics['loss/discriminator'].update_state(disc_loss)
    metrics['loss/realVSfake'].update_state(realVSfake)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  optimizer(learning_rate=_LR).apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  optimizer(learning_rate=_LR).apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train_model(dataset):
  """Train model
  
  Train:
    Iterate over epochs -> Iterate over batches -> Train step per batch ->
    Save (and reset) metrics

  Args:
    images: tf.data.Dataset
  """
  
  check_env()
  logging = get_logs()
  metrics = get_metrics()
  start = timer()

  print('\nEpoch\tGEN\tDIS\tRvsF\tKLD\tMSE\tElapsed\tETA')

  for epoch in range(1, _EPOCHS+1): # Iterate over epochs

    epoch_start = timer()
    output = [epoch, _EPOCHS] 

    for batch in dataset: # Iterate over batches
      image_batch = batch[0]
      noise = tf.random.normal([_BS, 196])
      train_step(image_batch, noise, metrics)

    with logging.as_default(): # Save (and reset) metrics
      tf.summary.histogram('input/noise', noise[-1], step=epoch)
      tf.summary.image('output/digit', generator(tf.expand_dims(noise[-1], 0), training=False), step=epoch)
      tf.summary.scalar('optimizer/lr', optimizer(learning_rate=_LR).lr(epoch), step=epoch)

      for key, metric in metrics.items(): # Loss metrics
        tf.summary.scalar(key, metric.result(), step=epoch)
        output.append(metric.result().numpy())
        metric.reset_states()

    output.append((timer()-start)/60)
    output.append(((timer()-epoch_start) * (_EPOCHS-epoch))/60)
    print('%i/%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.1f m\t%.1f m' % tuple(output))

train_model(train)

"""### Evaluate model"""

"""## Deployment

*   Use model
*   Export model
*   Export metadata
"""

# Global deployment parameter

_DEPLOYMENT = {
}

"""### Use model"""

def generate_digits():
  """ Generate digits

  Predict:
    Prediction -> Visualization 

  Returns:
    plt.fig
  """

  fig = plt.figure(figsize=(16, 16))

  for i in range(36): # Generate 36 digits
    fig.add_subplot(6, 6, i+1)
    _, prediction = predict(1)
    plt.imshow(prediction, cmap='gray')
    plt.axis('off')

  plt.show()

generate_digits()

"""### Export model"""

export = False #@param ["False", "True"] {type:"raw"}
if export: # Export model
  model.save('./generator.h5')

"""### Export metadata"""

def print_parameter(parameter, domain):
  print(domain, 'domain:')
  for key, value in parameter.items(): 
    print(key, '=', value)
  print('')

print_parameter(_DATA, 'Data')
print_parameter(_MODEL, 'Model')
print_parameter(_DEPLOYMENT, 'Deployment')