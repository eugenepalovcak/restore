#! /usr/bin/env python
# Copyright (C) 2018 Eugene Palovcak
# University of California, San Francisco
#
# Program for characterizing the SNR and SSNR of denoised images. 
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
import pandas as pd

from pyem import star
from pyem import ctf
from restore.utils import load_star
from restore.utils import load_mic
from restore.utils import save_mic
from restore.utils import bin_mic
from restore.utils import get_mic_freqs
from restore.utils import normalize
from restore.utils import fourier_crop
from restore.utils import fourier_pad_to_shape
from restore.utils import next32
from restore.utils import get_denoised_SNR
from restore.utils import get_denoised_SSNR
from restore.model import load_trained_model


def main(args):
    """ Main SNR-measuring function """

    # Load STAR file and neural network
    star_file = load_star(args.input_micrographs)
    num_mics = len(star_file)
    apix = star.calculate_apix(star_file)
    cutoff_frequency = 1./args.max_resolution
    nn = load_trained_model(args.model)
    orig,even,odd = args.even_odd_suffix.split(",")
    phaseflip=args.phaseflip

    SNR_df = pd.DataFrame(columns=["MicrographName", 
                                   "SNR_raw",
                                   "SNR_denoised", 
                                   "Frequencies", 
                                   "SSNR_raw",
                                   "SSNR_denoised"])

    # Main denoising loop
    for i, metadata in tqdm(star_file.iterrows(),
                            desc="Denoising", total=num_mics):

        mic_file = metadata[star.Relion.MICROGRAPH_NAME]

        # Pre-calculate frequencies and angles
        if not i:
            first_mic = load_mic(mic_file)
            freqs, angles = get_mic_freqs(first_mic, apix, angles=True)

        # Bin and denoise the even and odd micrographs
        even_mic_file = mic_file.replace(orig,even)
        even_raw_bin, even_den_bin = process_snr(
                                         nn, even_mic_file, metadata, 
                                         freqs, angles, apix, 
                                         cutoff_frequency, 
                                         phaseflip=phaseflip)

        odd_mic_file = mic_file.replace(orig,odd)
        odd_raw_bin, odd_den_bin = process_snr(
                                       nn, odd_mic_file, metadata,
                                       freqs, angles, apix,
                                       cutoff_frequency,
                                       phaseflip=phaseflip)

        # Calculate SNR and SSNRs
        old_x_dim,_ = first_mic.shape
        new_x_dim,_ = even_raw_bin.shape

        new_apix = apix * (old_x_dim / new_x_dim)
        SNR_den, SNR_raw = get_denoised_SNR(
                               even_raw_bin, odd_raw_bin,
                               even_den_bin, odd_den_bin)

        frequencies, SSNR_den, SSNR_raw = get_denoised_SSNR(
                                              even_raw_bin, odd_raw_bin,
                                              even_den_bin, odd_den_bin,
                                              new_apix)

        # Save the SNR in the dataframe
        SNR_df.loc[i] = [mic_file, SNR_raw, SNR_den, 
                         frequencies, SSNR_raw, SSNR_den]

    SNR_df.to_pickle(args.output_dataframe)
    return


def process_snr(nn, mic_file, metadata, freqs, angles, apix, cutoff,
            hp=.005, phaseflip=True):
    """ Denoise a cryoEM image for SNR calculation
 
    The following steps are performed:
    (1) The micrograph is loaded, phaseflipped, and Fourier cropped
    (2) A bandpass filter is applied with pass-band from cutoff to 1/200A
    (3) The inverse FT is calculated to return to real-space
    (4) The micrograph is padded do a dimension divisible by 32
    (5) The padded is passed through the CNN to denoise
    (6) The Fourier cropped denoised and raw images are returned
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
    mic_ft_bin *= bp_filter
    mic_bin = normalize(irfft2(mic_ft_bin).real)

    # Pad the image so the dimension is divisible by 32 (2**5)
    n_x, n_y = mic_bin.shape
    x_p, y_p = (next32(n_x) - n_x)//2, (next32(n_y) - n_y)//2
    mic_bin = np.pad(mic_bin, ((x_p, x_p), (y_p,y_p)), mode='mean')
    n_x2, n_y2 = mic_bin.shape

    # Denoise and unpad the image
    denoised = nn.predict(mic_bin.reshape((1,n_x2,n_y2,1)))
    denoised = denoised.reshape((n_x2,n_y2))

    denoised = normalize(denoised[x_p : n_x+x_p, y_p:n_y+y_p])
    raw = normalize(mic_bin[x_p : n_x+x_p, y_p:n_y+y_p])

    return raw, denoised


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--input_micrographs", "-i", type=str, default=None,
                        help="STAR file with micrographs to restore")

    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Trained neural network model")

    parser.add_argument("--max_resolution", "-r", type=float, default=4.5,
                        help="Max resolution to consider in denoising (angstroms). \
                              Should be consistent with the training data")

    parser.add_argument("--output_dataframe", "-o", type=str, default="denoised_SNR.pkl",
                        help="Output dataframe file containing the micrograph name and \
                              an estimate of the raw/denoised SNR and SSNRs.")

    # Optional arguments
    parser.add_argument("--even_odd_suffix", "-s", type=str,
                        default="DW,EVN,ODD",
                        help="A comma-separated series of three suffixes.  \
                              The first is a suffix with the micrograph's name. \
                              The second is the suffix of the 'even' sums. \
                              The third is the suffix of the 'odd' sums. \
                                                                             \
                              If MotionCor2 is used to generate even/odd sums,\
                              the default should be sufficient.")

    parser.add_argument("--phaseflip", dest="phaseflip", action="store_true",
                        help="Correct the CTF of the images by phase-flipping. \
                              Should be consistent with the training data")

    parser.add_argument("--dont_phaseflip", dest="phaseflip", action="store_false",
                        help="Don't phase-flip the images.")

    parser.set_defaults(phaseflip=True)

    sys.exit(main(parser.parse_args()))

