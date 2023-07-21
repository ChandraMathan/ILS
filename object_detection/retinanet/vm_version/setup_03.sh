#!/bin/bash

#to be run from local drive

ssh_key='~/.ssh/ILS-vm2_key.pem'
azure_user='azureuser@20.239.86.44'
source_model=~/Desktop/ILS_infra_packages_scripts/packages/models-master/
dest_model='~/development/tensorflow_models'

source_protoc=~/Desktop/ILS_infra_packages_scripts/packages/protoc-3.20.3-linux-x86_64
dest_protoc='~/development/assets'

source_checkpoint=~/Desktop/ILS_infra_packages_scripts/packages/checkpoint_retinanet
dest_checkpoint='~/development/assets'

echo $ssh_key
#install Object Detection API
scp -i "${ssh_key}" -r "${source_model}" "${azure_user}":"${dest_model}"
scp -i "${ssh_key}" -r "${source_protoc}" "${azure_user}":"${dest_protoc}"
scp -i "${ssh_key}" -r "${source_checkpoint}" "${azure_user}":"${dest_checkpoint}"