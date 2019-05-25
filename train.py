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

from pyem import star
from pyem import ctf
from restore.utils import load_star
from restore.utils import load_mic
from restore.utils import bin_mic
from restore.utils import get_patches


def main(args):
    """Main function for training a denoising CNN"""

    # Training data is stored in an HDF file.
    # If a STAR file is given, the training data will be created
    # If an HDF file is given, the training data will be loaded
    if args.training_mics:
        cutoff_frequency = 1./args.max_resolution
        training_data = args.training_filename
        generate_training_data(
            args.training_mics, cutoff_frequency, training_data)

    elif args.training_data:
        training_data = args.training_data

    else:
        raise Exception("Neither training micrographs or training_data were provided!")

    return


def generate_training_data(training_mics, cutoff_frequency, training_data):
    """ Generate the training data given micrographs and their CTF information

    Keyword arguments:
    training_mics -- Micrograph STAR file with CTF information for each image
    cutoff_frequency -- Spatial frequency for Fourier cropping an image
    training_data -- Filename for the HDF file that is created 
    """

    star_file = load_star(training_mics)
    apix = star.calculate_apix(star_file)
    n_mics = len(star_file)
    window = 192

    dset_file = File(training_data, "w")
    dset_shape, n_patches = get_dset_shape(star_file, window, 
                                           apix, cutoff_frequency)

    print(dset_shape)

    return 


def get_dset_shape(star_file, window, apix, cutoff_frequency):
    """Calculate the expected shape of the training dataset
    Returns the shape and the number of patches per micrograph"""

    first_mic_name = star_file[star.Relion.MICROGRAPH_NAME][0]
    mic_bin = bin_mic(load_mic(first_mic_name), apix, cutoff_frequency)
    n_patches = len(get_patches(mic_bin, window))
    n_mics = len(star_file)

    return (n_patches*n_mics, window, window, 1), n_patches


if __name__=="__main__":
    parser = argparse.ArgumentParser(
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--training_mics", "-m", type=str, default=None,
                        help="STAR file with micrographs and CTF information")

    parser.add_argument("--even_odd_suffix", "-s", type=str, 
                        default="DW,EVN,ODD",
                        help="A comma-separated series of three suffixes.  \
                              The first is a suffix in the training micrographs name. \
                              The second is the suffix of the 'even' sums. \
                              The third is the suffix of the 'odd' sums. \
                                                                             \
                              If MotionCor2 is used to generate even/odd sums,\
                              the default should be sufficient.")

    parser.add_argument("--training_data", "-t", type=str, default=None, 
                        help="HDF file containing processed training data")  

    parser.add_argument("--max_resolution", "-r", type=float, default=3.0, 
                        help="Max resolution to consider in training (angstroms). \
                              Determines the extent of Fourier binning.")

    parser.add_argument("--training_filename", "-f", type=str, default="training_data.hdf",
                        help="Name for the newly generated training data file.")

    parser.add_argument("--initial_model", "-i", type=str, default=None,
                        help="Initialize training with this pre-trained model")

    parser.add_argument("--batch_size", "-b", type=float, default=12,
                        help="Number of training examples used per training batch.")

    parser.add_argument("--model_prefix", "-x", type=str, default="model", 
                        help="Prefix for model files containing the structure and \
                              weights of the neural network.")
  
    sys.exit(main(parser.parse_args()))
