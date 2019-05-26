#! /usr/bin/env python3
# Copyright (C) 2018 Eugene Palovcak
# University of California, San Francisco
#
# Utility functions for the restore program.
# These are mainly for IO and image processing including
# Fourier-based image resizing and Fourier filtering. 
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

import mrcfile
import numpy as np
from itertools import product

from pyem import star
from pyem import ctf

from numpy.fft import rfft2, irfft2
from numpy.fft import rfftfreq, fftfreq


def load_mic(filename):
    """ Load a micrograph (MRC format) into a numpy array""" 
    with mrcfile.open(filename, "r", permissive=True) as mrc:
        mic = mrc.data
    return mic

def load_star(filename):
    """ Load STAR file into a pyem (pandas) data frame"""
    star_file = star.parse_star(filename)
    return star_file

def bin_mic(mic, apix, cutoff, mic_freqs=None, lp=True, bwo=5):
    """ Bins a micrograph by Fourier cropping 
    Optionally applies a Butterworth low-pass filter"""
    if mic_freqs:
        f = mic_freqs
    else:
        f = get_mic_freqs(mic, apix) 

    mic_ft = rfft2(mic)
    if lp:
        mic_ft *=  1./ (1.+ ( f/ cutoff )**(2*bwo))
    mic_bin = irfft2(fourier_crop(mic_ft, f, cutoff)).real

    return mic_bin

def get_patches(img, w=192, overlap=0):
    """Extract patches of size w from an image, optionally with an overlap.
    w is given in pixels. overlap is given as a fraction of w."""
    N_x, N_y = img.shape
    X_cd = np.arange(0,N_x-w,int(w*(1-overlap)), dtype=np.int)
    Y_cd = np.arange(0,N_y-w,int(w*(1-overlap)), dtype=np.int)

    patches = [ img[c[0]:c[0]+w, c[1]:c[1]+w]
                for c in product(X_cd, Y_cd) ]

    return np.array(patches)
  
def normalize(x):
    """ Z-score normalization for array x """
    return (x - x.mean()) / x.std()

def get_mic_freqs(mic, apix, angles=False):
    """Returns array of effective spatial frequencies for a real 2D FFT.
    If angles is True, returns the array of the angles w.r.t. the X-axis
    """
    n_x, n_y = mic.shape
    x,y =  np.meshgrid(rfftfreq(n_y,d=apix), fftfreq(n_x,d=apix))
    s = np.sqrt(x**2 + y**2)

    if angles:
        a = np.arctan2(y,x)
        return s,a
    else:
        return s

def fourier_crop(mic_ft, mic_freqs, cutoff):
    """Extract the portion of the real FT lower than a cutoff frequency"""
    n_x, n_y = mic_ft.shape

    f_h = mic_freqs[0]
    f_v = mic_freqs[:n_x//2,0]

    
    c_h = np.searchsorted(f_h, cutoff)
    c_v = np.searchsorted(f_v, cutoff)

    mic_ft_crop = np.vstack((mic_ft[:c_v, :c_h], 
                             mic_ft[n_x - c_v:, :c_h]))
    return mic_ft_crop


