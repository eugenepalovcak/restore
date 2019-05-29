#! /usr/bin/env python3
# Copyright (C) 2018 Eugene Palovcak
# University of California, San Francisco
#
# Program denoising cryo-EM images with a trained convolutional network 
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
import sys
import argparse
from tqdm import tqdm

import numpy as np
from numpy.fft import rfft2, irfft2

from pyem import star
from pyem import ctf
from restore.utils import load_star
from restore.utils import load_mic
from restore.utils import save_mic
from restore.utils import bin_mic
from restore.utils import get_mic_freqs
from restore.utils import normalize
from restore.utils import fourier_crop
from restore.utils import next32

from restore.model import load_trained_model

def main(args):
    """ Main denoising CNN function """

    # Load STAR file and neural network
    star_file = load_star(args.input_micrographs)
    num_mics = len(star_file)
    apix = star.calculate_apix(star_file)
    cutoff_frequency = 1./args.max_resolution    
    nn = load_trained_model(args.model)
    suffix = args.output_suffix
    phaseflip=args.phaseflip
    merge_with_raw=args.merge_with_raw
 
    # Main denoising loop
    for i, metadata in tqdm(star_file.iterrows(),
                            desc="Denoising", total=num_mics):

        mic_file = metadata[star.Relion.MICROGRAPH_NAME]

        # Pre-calculate frequency and angle arrays
        if not i:
            first_mic = load_mic(mic_file)
            freqs, angles = get_mic_freqs(first_mic, apix, angles=True)    
        
        new_mic = process(nn, mic_file, metadata, freqs, angles, apix, 
                          cutoff_frequency, phaseflip=phaseflip, 
                          merge_with_raw=merge_with_raw)

        new_mic_file = mic_file.replace(".mrc", "{0}.mrc".format(suffix))
        save_mic(new_mic, new_mic_file)

    return

def process(nn, mic_file, metadata, freqs, angles, apix, cutoff, 
            hp=.005, phaseflip=True, merge_with_raw=True):
    """ Denoise a cryoEM image 
 
    The following steps are performed:
    (1) The micrograph is loaded, phaseflipped, and Fourier cropped
    (2) A bandpass filter is applied with pass-band from cutoff to 1/200A
    (3) The inverse FT is calculated to return to real-space
    (4) The micrograph is padded do a dimension divisible by 32
    (5) The padded is passed through the CNN to denoise then unpadded
    (6) The micrograph is upsampled and optionally merged with the high
        frequency image in the phase-flipped, non-denoised image.  
    """

    # Load the micrograph and phase-flip to correct the CTF
    mic = load_mic(mic_file)
    mic_ft = rfft2(mic) 

    if phaseflip:
        m = metadata
        try:
            phase_shift = m[star.relion.PHASESHIFT]
        except:
            phase_shift = 0.

        ctf_img = ctf.eval_ctf(freqs, angles,
                               m[star.Relion.DEFOCUSU],
                               m[star.Relion.DEFOCUSV],
                               m[star.Relion.DEFOCUSANGLE],
                               phase_shift,
                               m[star.Relion.VOLTAGE],
                               m[star.Relion.AC],
                               m[star.Relion.CS],
                               lp=2*apix)

        mic_ft *= np.sign(ctf_img)

    # Fourier crop the micrograph and bandpass filter
    mic_ft_bin = fourier_crop(mic_ft, freqs, cutoff)
    freqs_bin = fourier_crop(freqs, freqs, cutoff) 

    bp_filter = ((1. - 1./(1.+ (freqs_bin/ hp)**(10))) 
                     + 1./(1.+ (freqs_bin/ cutoff )**(10)))/2.
    mic_bin = normalize(irfft2(mic_ft_bin).real)

    # Pad the image so the dimension is divisible by 32 (2**5)
    n_x, n_y = mic_bin.shape
    x_p, y_p = (next32(n_x) - n_x)//2, (next32(n_y) - n_y)//2
    mic_bin = np.pad(mic_bin, ((x_p, x_p), (y_p,y_p)), mode='mean')
    n_x2, n_y2 = mic_bin.shape

    # Denoise the image and unpad
    denoised = nn.predict(mic_bin.reshape((1,n_x2,n_y2,1)))
    denoised = denoised.reshape((n_x2,n_y2))
    
    new_mic = denoised
    return new_mic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_micrographs", "-i", type=str, default=None,
                        help="STAR file with micrographs to restore")

    parser.add_argument("--model", "-m", type=str, default=None, 
                        help="Trained neural network model")

    parser.add_argument("--max_resolution", "-r", type=float, default=3.0,
                        help="Max resolution to consider in training (angstroms). \
                              Determines the extent of Fourier binning. Should be \
                              consistent with the resolution of the training data.")

    parser.add_argument("--output_suffix", "-s", type=str, default="_denoised", 
                        help="Suffix added to denoised image output")

    parser.add_argument("--merge_with_raw", dest='merge_with_raw', action='store_true', 
                        help="Merge the low-resolution denoised image with \
                              the uncorrected raw high-resolution image.")

    parser.add_argument("--dont_merge_with_raw", dest='merge_with_raw', action='store_false',
                        help="Return only the denoised image, upsampled to \
                              match the dimensions of the input image")

    parser.set_defaults(merge_with_raw=True)

    parser.add_argument("--phaseflip", dest="phaseflip", action="store_true",
                        help="Correct the CTF by phase-flipping. Should be consistent \
                              with the training data.")

    parser.add_argument("--dont_phaseflip", dest="phaseflip", action="store_false",
                        help="Don't phase-flip.")

    parser.set_defaults(phaseflip=True)

    sys.exit(main(parser.parse_args()))
