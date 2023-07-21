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

source_setup_1=~/Desktop/ILS/object_detection/retinanet/vm_version/setup_vm_toplevel.sh
source_setup_2=~/Desktop/ILS/object_detection/retinanet/vm_version/setup_vm_01.sh
source_setup_3=~/Desktop/ILS/object_detection/retinanet/vm_version/setup_vm_02.sh
dest_setup='~/setup'

source_test=~/Desktop/ILS/object_detection/retinanet/vm_version/sample_test.py
dest_test='~/setup'

echo $ssh_key

#create direcories
ssh "${azure_user}" "mkdir ~/development"
ssh "${azure_user}" "mkdir ~/development/tensorflow_models"
ssh "${azure_user}" "mkdir ~/development/assets"
ssh "${azure_user}" "mkdir ~/setup"

#install Object Detection API
scp -i "${ssh_key}" -r "${source_model}" "${azure_user}":"${dest_model}"
scp -i "${ssh_key}" -r "${source_protoc}" "${azure_user}":"${dest_protoc}"
scp -i "${ssh_key}" -r "${source_checkpoint}" "${azure_user}":"${dest_checkpoint}"
scp -i "${ssh_key}" -r "${source_setup_1}" "${azure_user}":"${dest_setup}"
scp -i "${ssh_key}" -r "${source_setup_2}" "${azure_user}":"${dest_setup}"
scp -i "${ssh_key}" -r "${source_setup_3}" "${azure_user}":"${dest_setup}"

#run bash script on vm for setup

ssh "${azure_user}" "bash chmod +x ~/setup/setup_vm_toplevel.sh"
ssh "${azure_user}" "bash ~/setup/setup_vm_toplevel.sh"
