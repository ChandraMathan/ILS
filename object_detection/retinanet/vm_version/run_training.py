
from retinanet_training import ObjectDetection

""" 
        Params:
        training_image_data_path : location of jpg training images which is path defined from home directory and images to be numbered 100,101,.... in *.jpg extension
        bbox_path: path (defined from home directory) + file name of bounding box in '.npy' format with shape equals to (number of images, ) and each image (e.g bbox[0]) has shape of (num of objects, 4)
        indices_class_list_path: path (defined from home directory) + file name in '.npy' format containing classification (e.g, 1,2,3 ) of objects corresponding to the image and bounding box location defined in bbox_fname
        category_index: its a dictionary of categories being defined (e.g 1,2,3, .. and the name of those)
        num_classes: number of classess defined. type is int
        pipeline_config: this is path (defined from home directory) of config file to be used while training. 
        checkpoint_path: location (defined from home directory) of checkpoint where checkpoint is used for intitiation of weights and for training
        model_export_path: path (defined from home directory) where trained model checkpoint is exported.
        config_export_path: path (defined from home directory) where trained model config file is exported.
        batch_size: size of the batch for training. should be less than the total examples. as int.
        learning_rate: learning rate for training as float
        num_batches: number of epochs which is same as num of batches
        
"""

training_image_data_path = '~/development/ILS/object_detection/data'
bbox_path = '~/development/ILS/object_detection/data/bbox.npy'
indices_class_list_path = '~/development/ILS/object_detection/data/indices_class_list.npy'
category_index_path = '~/development/ILS/object_detection/data/category_index.pickle'
num_classes = 3
pipeline_config_path = '~/development/tensorflow_models/models-master/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
checkpoint_path = '~/development/assets/checkpoint_retinanet/ckpt-0'
model_export_path = '~/development/ILS/object_detection/retinanet/trained_models/checkpoints'
config_export_path = '~/development/ILS/object_detection/retinanet/trained_models/configs'
batch_size = 5
learning_rate = 0.01
num_batches = 100


ObjectDetection(training_image_data_path,bbox_path, indices_class_list_path, category_index_path, num_classes, pipeline_config_path, checkpoint_path , model_export_path, config_export_path,batch_size, learning_rate,num_batches).main()

