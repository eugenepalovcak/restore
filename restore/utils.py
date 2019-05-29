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

from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import interp1d

def load_mic(filename):
    """ Load a micrograph (MRC format) into a numpy array""" 
    with mrcfile.open(filename, "r", permissive=True) as mrc:
        mic = mrc.data
    return mic

def save_mic(mic, filename, overwrite=True):
    """ Save a micrograph (numpy array) in an MRC file """
    with mrcfile.new(filename, overwrite=overwrite) as mrc:
        mrc.set_data(mic)
    return

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

def unbin_mic(mic, unbin_shape):
    """ Upsample a micrograph by padding its Fourier transform
    Basically a Fourier 'resize' function. While this works, 
    it's Fourier interpolation, which can causes 'ringing'.

    In practice, we use 'fourier_pad_to_shape' and then process
    the resulting FT to soften the sharp edges and reduce ringing. 
    """

    s_x = len(fftfreq(unbin_shape[0]))
    s_y = len(rfftfreq(unbin_shape[1]))

    mic_ft = rfft2(mic)
    mic_ft = fourier_pad_to_shape(mic_ft, (s_x,s_y))

    return irfft2(mic_ft).real

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

def fourier_pad_to_shape(mic_ft, new_shape):
    """ Pad a Fourier transform with zeros
    This is equivalent to upsampling in real-space """
    
    # Separate input FT into top and bottom quadrants
    n_x, n_y = mic_ft.shape
    top = mic_ft[:n_x//2]
    bottom = mic_ft[n_x//2:]
  
    # Insert quadrants into new, larger FT array
    mic_ft_pad = np.zeros((new_shape), dtype=np.complex64)
    mic_ft_pad[:n_x//2, :n_y] = top
    mic_ft_pad[new_shape[0]-n_x//2:, :n_y] = bottom

    return mic_ft_pad

def next32(n):
    """ Return next integer divisible by 32 """
    while n%32 !=0:
        n+=1
    return n


def get_softmask(freqs, cutoff, width):
    """ Given a frequency array and a cutoff frequency, 
    generates a soft Fourier mask that decays from 1 to 0
    over a band of pixels with specified width. Uses a sine 
    function to smoothly go from 1 to 0.

    Inspired by a trick Daniel Asarnow does in pyem's mask.py
    """

    # Erode mask 'width' pixels then distance transform
    mask = freqs < cutoff
    eroded = binary_erosion(mask, iterations=width, border_value=1)
    dt = distance_transform_edt(~eroded)

    # Generate soft edge (sine wave) interpolator
    x = np.arange(1, width+1)
    y = np.sin(np.linspace(np.pi/2,0, width))
    f = interp1d(x,y, bounds_error=False, fill_value=(1,0))

    # Interpolate the distance transform so it decays smoothly
    softmask = f(dt)

    return softmask


