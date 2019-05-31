# UCSF restore

`restore` is a program for denoising cryogenic electron microscopy images with a neural network. 

## Requirements
`restore` requires a Linux system with an NVIDIA GPU. 
[MotionCor2](https://msg.ucsf.edu/software) is recommended for generating the training data. 

## Installation for Linux

Working on a simpler installation. For the time being:

1. Make sure the [NVIDIA](https://www.nvidia.com/Download/index.aspx?lang=en-us) driver is up-to-date. (Must be >410)
2. [Install Miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) (or Anaconda) if necessary. 
3. Make sure [git](https://git-scm.com/download/linux) is installed.
4. Download the package and navigate into the directory:
```bash
git clone https://github.com/eugenepalovcak/restore
cd restore
```
5. Create an conda python environment that contains the required dependencies. This step may take a few minutes.
```bash
conda env create -f restore.yml
conda activate restore
```
6. Now install `mrcfile` (only available with `pip`) and Daniel Asarnow's `pyem`
```bash
git clone https://github.com/asarnow/pyem.git
pip install ./pyem
```
7. Install  `restore` 
```bash
pip install .
```

8. Add the restore binaries to your path via `.bash_profile` or `.bashrc` file:
```bash
```
