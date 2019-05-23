# UCSF restore

Restore is a program for denoising cryogenic electron microscopy images with a neural network. 

[MotionCor2](https://msg.ucsf.edu/software) is recommended for generating the training data. 

## Installation for Linux

1. Make sure the [NVIDIA](https://www.nvidia.com/Download/index.aspx?lang=en-us) driver is up-to-date. (Must be >410)
2. [Install Miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) (or Anaconda) if necessary. 
3. Download the package and type: `conda env create -f restore.yml`. This make take a few minutes.
