#! /bin/bash

# Create and activate virtual environment
virtualenv -p /usr/bin/python3 ./.detectron
source .detectron/bin/activate

# Install torchvision
python3 -m pip install -U torch torchvision
python3 -m pip install git+https://github.com/facebookresearch/fvcore.git

# Install detectron2
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
python3 -m pip install -e detectron2_repo

python3 -m pip install -r requirements.txt
