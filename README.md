# UCSF restore

Restore is a program for denoising cryogenic electron microscopy images with a neural network. 

## Requirements
Restore requires a Linux system with an NVIDIA GPU. 
[MotionCor2](https://msg.ucsf.edu/software) is recommended for generating the training data. 

## Installation for Linux

1. Make sure the [NVIDIA](https://www.nvidia.com/Download/index.aspx?lang=en-us) driver is up-to-date. (Must be >410)
2. [Install Miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) (or Anaconda) if necessary. 
3. Make sure [git](https://git-scm.com/download/linux) is installed. 
4. Run [sh install.sh] to install [pyem](https://github.com/asarnow/pyem) and the restore code. 
