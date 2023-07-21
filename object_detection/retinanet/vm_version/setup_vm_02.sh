#!/bin/bash

#to be run from vm

#install python
pyenv install 3.10.6 #install pythonversion
pyenv global 3.10.6 #set this as global version

echo python --version

#move protoc
cd ~/development/assets/protoc-3.20.3-linux-x86_64
sudo mv bin/protoc /usr/local/bin

echo protoc --version

#instatiate protoc
cd ~/development/tensorflow_models/models-master/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .


#install libraries
pip install imageio
pip uninstall pillow
pip install pillow==9.5

#test
cd ~/setup
python sample_test.py