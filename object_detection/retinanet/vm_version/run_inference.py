from inference_retinanet import InferenceRetinanet

"""
        Params:

        image_path: test image path
        checkpoint_path: checkpoint to be used for inference
        detected_image_path: image detection 
        category_index_path: its a dictionary of categories being defined (e.g 1,2,3, .. and the name of those)
        pipeline_config: config file to be used for inference
        num_classes: number of classess defined. type is int
        
"""

image_path = '~/development/ILS/object_detection/data'
checkpoint_path = '~/development/ILS/object_detection/retinanet/trained_models/checkpoints'
detected_image_path = '~/development/ILS/object_detection/retinanet/detected_images'
category_index_path = '~/development/ILS/object_detection/data/category_index.pickle'
pipeline_config = '~/development/ILS/object_detection/retinanet/trained_models/configs'
num_classes = 3

InferenceRetinanet(image_path, checkpoint_path, detected_image_path, category_index_path, pipeline_config, num_classes).main()