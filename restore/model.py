#! /usr/bin/env python3
# Copyright (C) 2018 Eugene Palovcak
# University of California, San Francisco
#
# Module containing neural network models and training utility functions
# Includes a vanilla U-net and wide-activation U-net
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
import tensorflow as tf

from keras.layers import Add
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.models import Model
from keras.layers import Lambda

from keras.models import load_model
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.utils.io_utils import HDF5Matrix
from keras.utils import Sequence

from external.weightnorm import AdamWithWeightnorm

def unet(kernel_size=3, filters=24, expansion=2, layers=5, acti='relu'):
    """ Nearly classical U-net model for dense 'array-to-array' predictions.

    The main difference between this model and the original U-net proposed
    in Ronneberger et al. is that convolutional operations are replaced
    with memory-efficient separable convolutions.

    Because the network may eventually be applied to very large images,
    we generally make design decisions that favor memory efficiency, 
    especially if the performance is equivalent or better. 

    Input layer:
    Takes a tensor of size (b, None, None, 1), where b is the batch size.
    So long as input dimensions are divisble by 2**num_layers, the layer
    reshaping will work. This lets us train on small image patches but 
    predict much larger images that wouldn't fit in memory. 

    Encoder layers:
    In each layer of the encoder, two separable convolutions are performed.
    The first maintains the dimension of the input.
    The second uses a stride of 2, down-sampling the feature maps. 
    The feature depth is increased according to the 'expansion' parameter
    The feature maps are also stored for later in the list 'skips' 

    Decoder layers:
    In each layer of the decoder, the feature maps are first upsampled.
    Then, another separable convolution is performed. 
    Finally, the congruent feature maps from the encoder are merged
    by concatenating them with those of the decoder and reducing the 
    dimensionality with a  1x1 convolution.  

    Output layers:
    Once we're back to full-resolution feature maps, a series of 
    1x1 convolutions are used like a multi-layer perceptron to compute
    the value of each denoised pixel.

    See:
    Chollet. "Xception: Deep Learning with Depthwise Separable Convolutions"
    """

    k = kernel_size
    f = filters
    l = layers
    e = expansion

    def sepconv(n,f,k,s=1):
        """ Depthwise separable convolution layer """
        n = SeparableConv2D(f, k, padding='same', activation=acti, strides=s)(n)
        return n

    def skipreduce(n,s,f):
        """ Concatenate two feature tensors, n and s then 
        reduce the feature depth to f with a 1x1 convolution """
        n = Concatenate()([n,s])
        n = Conv2D(f,(1,1), activation=acti)(n)
        return n

    def subpixel_upsample(n,f):
        """ Upsample the feature maps with the depth-to-space operation.
        This operation is also sometimes called 'subpixel convolution' """
        n = Conv2D(f*4, 1, activation=acti)(n)
        n = Lambda(lambda x: tf.depth_to_space(x, 2))(n)
        return n

    # Input layer
    input_layer = Input(shape=(None,None,1))
    n = input_layer
    skips = [n]

    # Encoder layers
    for i in range(l):
        n = sepconv(n, int(f*(e**i)), k)
        n = sepconv(n, int(f*(e**i)), k, 2)
        skips.append(n)

    n = sepconv(n, int(f*(e**l)), k)
    n = sepconv(n, int(f*(e**l)), k)

    # Decoder layers
    for i in reversed(range(l)):
        n = subpixel_upsample(n, int(f*(e**i)))
        n = sepconv(n, int(f*(e**i)), 2)
        n = skipreduce(n, skips[i],  int(f*(e**i)))

    # Output layers
    n = Conv2D(f,1, activation='relu')(n)
    n = Conv2D(f,1, activation='relu')(n)
    output_layer = Conv2D(1,1)(n)

    return Model(inputs=input_layer, outputs=output_layer)


def waunet(layers=3, blocks_per_layer=4, expansion=4):
    """ WAUNET = wide-activation U-net 
    
    Each block of convolutional layers in the U-net is replaced with a 
    small 'wide-activation super-resolution' network. 

    Wide activation layers take linear combinations of feature maps to 
    increase their depth before applying a non-linearity and convolution.
    The thought is that these expanded linear combinations allow more
    information to pass through the layers without getting crushed by the
    non-linearity. This is important for dense image prediction tasks
    where we can't afford to lose information. These tasks include 
    super-resolution and also denoising. For the same reason, each layer
    also uses residual connections and the entire network maintains
    skip connections between encoder and decoder branches.

    Encoder layers:
    Each layer is a mini WDSR. Downsampling is performed with max pooling.
    Feature maps are saved for skip connections in the decoder.

    Decoder layers:
    Each layer is a mini WDSR. Upsampling is performed with depth-to-space
    Skip feature maps from the encoder are merged here. Convolutions in the
    decoder use even-sized kernels to avoid checkerboard artefacts.

    See: 
    Yu et al.
    "Wide Activation for Efficient and Accurate Image Super-Resolution"
    Odena et al.
    "Deconvolution and checkerboard artefacts"    
    """
    kernel_size = k = 3
    num_filters = f = 32
    e = expansion
    
    def wa_block(n, f, e, k, strides=None):
        """ Wide-activation convolutional block """
        lin = 0.8

        n_in = n
        n = Conv2D(f * e, 1, padding='same')(n)
        n = Activation('relu')(n)
        n = Conv2D( int(f*lin), 1, padding='same')(n)
        n = Conv2D(f, k, padding='same')(n)
        n = Add()([n_in, n])

        if strides:
            n = Conv2D(f, 3, padding='same', strides=strides)(n)
            n = Activation('relu')(n)

        return n

    def skipreduce(n,s,f):
        """ Concatenate two feature tensors, n and s then 
        reduce the feature depth to f with a 1x1 convolution """
        n = Concatenate()([n,s])
        n = Conv2D(f, 1, padding='same')(n)
        n = Activation('relu')(n)
        return n

    def expand_and_upsample(n, f, e, k):
        """ Wide-activation block followed by depth-to-space upsampling """
        n = Conv2D(f*4, 1, padding='same')(n)
        n = Activation('relu')(n)
        n = wa_block(n, f*4, e, k)
        n = Lambda(lambda x: tf.depth_to_space(x, 2))(n)
        return n

    # Input layer
    input_layer = Input(shape=(None, None, 1))
    n = input_layer
    skips = []
    n = Conv2D(f, k, padding='same')(n)

    # Encoder layers
    for l in range(layers):
        j = MaxPooling2D(2)(n)
        for b in range(blocks_per_layer-1):
            n = wa_block(n, f, e, k)
        n = wa_block(n, f, e, k, strides=2)
        n = Add()([n,j])
        skips.append(n)

    # Decoder layers
    decoder_kernel_size = dk = 4
    for l in reversed(range(layers)):
        j = expand_and_upsample(n,f,e,dk)
        n = skipreduce(n, skips[l], f)
        for b in range(blocks_per_layer-1):
            n = wa_block(n, f, e, dk)
        n = expand_and_upsample(n, f, e, dk)
        n = Add()([n,j])

    # Output layers
    n = Conv2D(f, 1)(n)
    n = Activation('relu')(n)
    n = Conv2D(1, 1)(n)

    output_layer = n
    return Model(input_layer, output_layer)

def get_model(learning_rate, model='waunet'):
    """ Initialize a new neural network model 
    Uses the Adam optimizer with weight normalization. """
    opt = AdamWithWeightnorm(lr = learning_rate, 
              beta_1=0.9, beta_2=0.99, epsilon=1e-8)

    if model=='waunet':
         nn = waunet()
    elif model=='unet':
         nn = unet()
    else:
         raise Exception("{0} is not recognized".format(model))

    nn.compile(optimizer=opt, loss='mse')
    return nn

def load_trained_model(model_file):
    """ Load pre-trained Keras model with custom objects """
    nn = load_model(model_file, compile=compile,
             custom_objects={'tf':tf, 
                 'AdamWithWeightnorm':AdamWithWeightnorm})
    return nn

class SampleGenerator(Sequence):
    """ Turns HDF5 training data file into a generator that yields
    inputs and outputs. Compatible with 'fit_generator' in Keras """

    def __init__(self, training_data, batch_size=10):
        self.even = HDF5Matrix(training_data, 'even')
        self.odd = HDF5Matrix(training_data, 'odd')
        self.batch_size = batch_size
        self.n_data, self.n_x, self.n_y, self.n_channels = self.even.shape

    def __len__(self):
        return len(self.even) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = np.random.randint(0,self.n_data, self.batch_size)

        norm = lambda x: (x-x.mean())/x.std()
        inputs = np.array([norm(self.even[b]) for b in batch_idx])
        outputs = np.array([norm(self.odd[b]) for b in batch_idx])

        return inputs, outputs

class Schedule:
    """ Simple schedule callback for decreasing the learning rate """

    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125

def get_callbacks(model_directory, model_prefix, number_of_epochs, 
                  learning_rate, tensorboard_directory=None):
    """ Generates a list of callbacks """
    l = LearningRateScheduler(
            schedule=Schedule(number_of_epochs, learning_rate))
    m = ModelCheckpoint(
            str(model_directory) + "/{0}".format(model_prefix)
            + "_{epoch:02d}_{loss:0.3f}.hdf5", verbose=1, 
            monitor='loss', save_best_only=False, save_weights_only=False)

    callbacks = [l,m]
    if tensorboard_directory:
        callbacks.append(TensorBoard(log_dir=tensorboard_directory))

    return callbacks

def main():
    model = waunet()
    model.summary()

if __name__ == '__main__':
    main()


