#! /usr/bin/env python3
# Copyright (C) 2018 Eugene Palovcak
# University of California, San Francisco
#
# Program for training a convolutional neural network to denoise images 
# from cryogenic electron microscopy (cryo-EM).
# See README and help text for usage information.
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

from h5py import File
import mrcfile
import argparse
import sys
import os
from tqdm import tqdm

import numpy as np
from numpy.fft import rfft2
from numpy.fft import irfft2

from pyem import star
from pyem import ctf
from restore.utils import load_star
from restore.utils import load_mic
from restore.utils import bin_mic
from restore.utils import get_patches
from restore.utils import get_mic_freqs
from restore.utils import normalize
from restore.utils import fourier_crop

from restore.model import get_model
from restore.model import load_trained_model

def main(args):
    """Main function for training a denoising CNN"""

    # Training data is stored in an HDF file.
    # If a STAR file is given, the training data will be created
    # If an HDF file is given, the training data will be loaded
    if args.training_mics:
        cutoff_frequency = 1./args.max_resolution
        training_data = args.training_filename
        generate_training_data(args.training_mics, cutoff_frequency, 
                               training_data, args.even_odd_suffix,
                               phaseflip=args.phaseflip)

    elif args.training_data:
        training_data = args.training_data

    else:
        raise Exception(
            "Neither training micrographs or training_data were provided!")

    # Initialize a neural network model for training
    # OR if a pre-trained model is provided, load that instead
    learning_rate = args.learning_rate
    num_epochs = args.number_of_epochs
    epoch_length = args.batches_per_epoch
    batch_size = args.batch_size
    
    if args.initial_model:
        nn = load_trained_model(args.initial_model)
    else:
        nn = get_model(learning_rate)

    

    return 


def generate_training_data(training_mics, cutoff, training_data, suffixes,
                           window=192, phaseflip=True):
    """ Generate the training data given micrographs and their CTF information

    Keyword arguments:
    training_mics -- Micrograph STAR file with CTF information for each image
    cutoff -- Spatial frequency for Fourier cropping an image
    training_data -- Filename for the HDF file that is created 

    It is presumed that all images have the same shape and pixel size. 
    By default, phase-flipping is performed to correct for the CTF.
    """

    star_file = load_star(training_mics)
    apix = star.calculate_apix(star_file)
    n_mics = len(star_file)

    dset_file = File(training_data, "w")
    dset_shape, n_patches, mic_freqs, mic_angles = get_dset_shape(
                                                       star_file, window, 
                                                       apix, cutoff)

    even_dset = dset_file.create_dataset("even", dset_shape, dtype="float32")
    odd_dset = dset_file.create_dataset("odd", dset_shape, dtype="float32")

    orig,even,odd = suffixes.split(",")
    if len(suffixes.split(",")) != 3:
        raise Exception("Improperly formatted suffixes for even/odd mics!")

    for i, metadata in tqdm(star_file.iterrows(), 
                            desc="Pre-processing", total=n_mics):
        
        mic_file = metadata[star.Relion.MICROGRAPH_NAME]
        even_file = mic_file.replace(orig, even)
        odd_file = mic_file.replace(orig, odd)

        mic_even_patches, apix_bin = process(metadata, cutoff, window, 
                                             even_file, mic_freqs, mic_angles,
                                             phaseflip=phaseflip)

        mic_odd_patches, apix_bin = process(metadata, cutoff, window, 
                                            odd_file, mic_freqs, mic_angles,
                                            phaseflip=phaseflip)

        even_dset[i*n_patches: (i+1)*n_patches] = mic_even_patches
        odd_dset[i*n_patches: (i+1)*n_patches] = mic_odd_patches
        

    even_dset.attrs['apix']=apix_bin
    even_dset.attrs['phaseflip']=phaseflip

    odd_dset.attrs['apix']=apix_bin
    odd_dset.attrs['phaseflip']=phaseflip

    dset_file.close()
    return 


def get_dset_shape(star_file, window, apix, cutoff_frequency):
    """Calculate the expected shape of the training dataset.
    Returns the shape of the dataset, the number of patches per micrograph,
    and the unbinned spatial frequency and angle arrays so they don't need 
    to be recalculated in later steps"""

    first_mic = load_mic(star_file[star.Relion.MICROGRAPH_NAME][0])
    mic_bin = bin_mic(first_mic, apix, cutoff_frequency)

    s,a = get_mic_freqs(first_mic, apix, angles=True)
    n_patches = len(get_patches(mic_bin, window))
    n_mics = len(star_file)

    return (n_patches*n_mics, window, window, 1), n_patches, s, a


def process(metadata, cutoff, window, mic_file, freqs, angles, 
            bandpass=True, hp=.005, phaseflip=True):
    """ Process a training micrograph.

    The following steps are performed:
    (1) The micrograph is loaded, Fourier transformed and Fourier cropped
    (2) A bandpass filter is applied with pass-band from cutoff to 1/200A
    (3) The FT is multiplied by the sign of the CTF (phase-flipping)
        This is a crude form of CTF correction. 
    (4) The inverse FT is applied to return to real-space
    (5) The binned, filtered image is divided into patches, which are
        normalized (Z-score normalization) and returned
    """

    mic = load_mic(mic_file)
    mic_ft = rfft2(mic)

    mic_ft_bin = fourier_crop(mic_ft, freqs, cutoff)
    freqs_bin = fourier_crop(freqs, freqs, cutoff)
    angs_bin = fourier_crop(angles, freqs, cutoff)

    apix_bin = 0.5/freqs_bin[0,-1]

    if bandpass:
        bp_filt = ( (1. - 1./(1.+(freqs_bin/hp)**10)) 
                   + 1./(1.+(freqs_bin/cutoff)**10)/2.)
        mic_ft_bin *= bp_filt

    if phaseflip:
        m = metadata
        try:
            phase_shift = m[star.relion.PHASESHIFT]
        except:
            phase_shift = 0.
 
        ctf_img = ctf.eval_ctf(freqs_bin, angs_bin,
                               m[star.Relion.DEFOCUSU],
                               m[star.Relion.DEFOCUSV],
                               m[star.Relion.DEFOCUSANGLE],
                               phase_shift,
                               m[star.Relion.VOLTAGE],
                               m[star.Relion.AC],
                               m[star.Relion.CS], 
                               bf = 0, lp=2*apix_bin)

        mic_ft_bin *= np.sign(ctf_img)
    
    mic = irfft2(mic_ft_bin).real.astype('float32')
    patches = [normalize(p) for p in get_patches(mic, window)]
    n_patches = len(patches)

    return np.array(patches).reshape((n_patches, window, window, 1)), apix_bin
 

if __name__=="__main__":
    parser = argparse.ArgumentParser(
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments: 
    # User must provide STAR file of training mics (--training_mics, -m)
    # OR training data HDF file (--trainind_data, -t)
    parser.add_argument("--training_mics", "-m", type=str, default=None,
                        help="STAR file with micrographs and CTF information")

    parser.add_argument("--training_data", "-t", type=str, default=None,
                        help="HDF file containing processed training data")

    # Optional arguments
    parser.add_argument("--even_odd_suffix", "-s", type=str, 
                        default="DW,EVN,ODD",
                        help="A comma-separated series of three suffixes.  \
                              The first is a suffix in the training micrographs name. \
                              The second is the suffix of the 'even' sums. \
                              The third is the suffix of the 'odd' sums. \
                                                                             \
                              If MotionCor2 is used to generate even/odd sums,\
                              the default should be sufficient.")

    parser.add_argument("--max_resolution", "-r", type=float, default=3.0, 
                        help="Max resolution to consider in training (angstroms). \
                              Determines the extent of Fourier binning.")

    parser.add_argument("--training_filename", "-f", type=str, default="training_data.hdf",
                        help="Name for the newly generated training data file.")

    parser.add_argument("--initial_model", "-i", type=str, default=None,
                        help="Initialize training with this pre-trained model")

    parser.add_argument("--batch_size", "-b", type=float, default=10,
                        help="Number of training examples used per training batch.")

    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4,
                        help="Initial learning rate for training the neural network")

    parser.add_argument("--number_of_epochs", type=int, default=200,
                        help="Number of training epochs to perform. \
                              Model checkpoints are produced after every epoch.")

    parser.add_argument("--batches_per_epoch", type=int, default=500,
                        help="Number of training batches per epoch")

    parser.add_argument("--model_prefix", "-x", type=str, default="model", 
                        help="Prefix for model files containing the structure and \
                              weights of the neural network.")

    parser.add_argument("--phaseflip", dest="phaseflip", action="store_true",
                        help="Correct the CTF of the training images by phase-flipping")

    parser.add_argument("--dont_phaseflip", dest="phaseflip", action="store_false",
                        help="Don't phase-flip the training images.") 

    parser.set_defaults(phaseflip=True) 

    sys.exit(main(parser.parse_args()))
