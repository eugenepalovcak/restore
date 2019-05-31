# Script for installing restore to the local environment
# This should be replaced with a conda package installation. 


# (1) Create python environment and install dependencies
conda env create -f restore.yml
conda activate restore
pip install mrcfile

# (2) Install UCSF pyem to the local environment
git clone https://github.com/asarnow/pyem.git
pip install ./pyem

# (3) Move up a directory and install 
cd ..
pip install ./restore

