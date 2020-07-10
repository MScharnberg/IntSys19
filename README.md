# IntSys19

## Contents

* [Setup](https://github.com/MScharnberg/IntSys19/tree/documentation#setup)
  * Getting started
  * Requirements
* [Data](https://github.com/MScharnberg/IntSys19/tree/documentation#data)
  * Choose dataset 
  * Load dataset
  * Explore dataset
  * Preprocess dataset
  * Visualize dataset
* [Model](https://github.com/MScharnberg/IntSys19/tree/documentation#model)
  * Choose model 
  * Compile model
  * Explore model
  * Train model
  * Evaluate model
* [Deployment](https://github.com/MScharnberg/IntSys19/tree/documentation#deployment)
  * Use model
  * Export model
  * Export metadata

## Setup

### Getting started

Option 1 (RECOMMENDED): Open [Notebook](./Notebook.ipynb) in [CoLab](https://colab.research.google.com/)

Option 2: Install requirements via `pip install -r requirements.txt` and run [Script](./script.py) on local [Jupyter Notebook](https://jupyter.org/) Server

### Requirements

| Package             | Version |
|:--------------------|--------:|
| NumPy               | 1.18    |
| TensorFlow          | 2.2.0   |
| TensorFlow Datasets | 2.1.0   |
| TensorBoard         | 2.2.0   |

## Data

### Choose dataset 

* [MNIST](http://yann.lecun.com/exdb/mnist/) (Modified National Institute of Standards and Technology)
* [HSD Sans]()
  * Work in Progress

### Load dataset

### Explore dataset

![Label distribution](./img/label.png)

Label distribution

### Preprocess dataset

### Visualize dataset

![Real data](./img/real.png)

Real data instances

## Model

### Choose model 

* [GAN](https://arxiv.org/abs/1406.2661) (Generative Adversarial Network)
  * Generator as generative model
    * Map imaginary to real data distribution
    * Copy digits as close to original ones as regression problem
  * Discriminator as discriminative model
    * Distinguish imaginary from real data distribution
    * Differentiate between a digit's copy and original as classification problem
* [CGAN]() (Conditional Generative Adversarial Network)
  * Work in Progress

![GAN](./img/gan.png)

### Explore model

![Generator](./img/generator_grouped.svg)

Generator architecture (visualized by [Net2Vis](https://arxiv.org/abs/1902.04394))

![Discriminator](./img/discriminator_grouped.svg)

Discriminator architecture (visualized by [Net2Vis](https://arxiv.org/abs/1902.04394))

### Compile model

* Loss function
  * Discriminator
  * Generator
  * Discriminator + Generator = GAN

* Loss metrics
  * Generator
  * Discriminator
  * Real vs. Fake
  
* Optimizer
  * [Adam](https://arxiv.org/abs/1412.6980) (Adaptive Moment Estimation)

### Train model

### Evaluate model

* Evaluation metrics
  * MSE (Mean-squared error)
  * KLD (Kullback–Leibler divergence)

## Deployment 

### Use model

![Fake data](./img/fake.png)

Fake data instances

### Export model

### Export metadata
