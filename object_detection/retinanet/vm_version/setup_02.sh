#!/bin/bash

#to be run from vm

#install python
pyenv install 3.10.6 #install pythonversion
pyenv global 3.10.6 #set this as global version

echo python --version


#create directories
mkdir ~/development
mkdir ~/development/tensorflow_models
mkdir ~/development/assets