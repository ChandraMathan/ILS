
import os
import glob
import io
import imageio
from PIL import Image
import numpy as np
from six import BytesIO
from pathlib import Path
import tensorflow as tf
import random
import json
import matplotlib.pyplot as plt
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.meta_architectures import ssd_meta_arch



    
class ObjectDetection:
    """ Training and evaluation of object detection algorithm retinanet"""
    
    def __init__(self, training_image_data_path, bbox_path, indices_class_list_path, category_index, num_classes, pipeline_config, checkpoint_path, model_export_path, config_export_path, detected_image_path):
        
        """ 
        Params:
        training_image_data_path : location of jpg training images which is path defined from home directory
        bbox_path: path (defined from home directory) + file name of bounding box in '.npy' format with shape equals to (number of images, ) and each image (e.g bbox[0]) has shape of (num of objects, 4)
        indices_class_list_path: path (defined from home directory) + file name in '.npy' format containing classification (e.g, 1,2,3 ) of objects corresponding to the image and bounding box location defined in bbox_fname
        category_index: its a dictionary of categories being defined (e.g 1,2,3, .. and the name of those)
        num_classes: number of classess defined. type is int
        pipeline_config: this is path (defined from home directory) of config file to be used while training. 
        checkpoint_path: location (defined from home directory) of checkpoint where checkpoint is used for intitiation of weights and for training
        model_export_path: path (defined from home directory) where trained model checkpoint is exported.
        config_export_path: path (defined from home directory) where trained model config file is exported.
        detected_image_path: path (defined from home directory) where after inference is run, bounding box is drawn and image saved in this folder.

        """

        self.training_image_data_path = os.path.expanduser(training_image_data_path)
        self.bbox_path = os.path.expanduser(bbox_path)
        self.indices_class_path = os.path.expanduser(indices_class_list_path)

        self.category_index = category_index
        self.num_classes =num_classes
        self.pipeline_config = os.path.expanduser(pipeline_config)
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.model_export_path = os.path.expanduser(model_export_path)
        self.config_export_path = os.path.expanduser(config_export_path)

        self.train_images_np = None
        self.bbox = None
        self.indices_class = None
        self.train_image_tensors = []
        self.gt_classes_one_hot_tensors = []
        self.gt_box_tensors = []

        self.detection_model = None
        self.model_config = None

        self.detected_image_path = os.path.expanduser(detected_image_path)


    def load_image_into_numpy_array(self,path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
            path: a file path.

        Returns:
            uint8 numpy array with shape (img_height, img_width, 3)
        """

        img_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(img_data))
        (im_width, im_height) = image.size
        
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def sort_filenames(self):
        
        """ 
            the input which is the filenames are expected to start from 100, 101... 
            the ouput is the list if names of sorted files in ascending order
        """

        #get all the jpeg file names 
        file_names_images = []
        for file in glob.glob(self.training_image_data_path+"/*.jpg"):
            file_name = file.replace(self.training_image_data_path+'/',"")
            file_names_images.append(file_name)

        #sort these in order. note that the naming of file is it should start from '100,101,...'. 
        #this format is validated in 'retinanet_labelImg'
        file_names_sorted = []
        for file in file_names_images:
            file_names_sorted.append(int(file[0:3]))

        file_names_sorted.sort()
        file_names = [str(x)+'.jpg' for x in file_names_sorted]

        return file_names
    
    def np_image(self,file_name):
        """ images are converted to numpy """
        train_images_np = []
        for image_item in file_name:
            image_item_path = self.training_image_data_path+"/"+image_item
            train_images_np.append(self.load_image_into_numpy_array(image_item_path))
        self.train_images_np = train_images_np

        return self.train_images_np
    
    def load_numpy(self,file_path):
        """Loads .npy file"""
        return np.load(file_path, allow_pickle=True)

    def tensor_ground_truth(self):
        """ create tensor of numpy image, class and bbox"""
        # Convert class labels to one-hot; convert everything to tensors.
        for (train_image_np, gt_box_np,indices_class) in zip(
                self.train_images_np, self.bbox,self.indices_class):
            self.train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
                train_image_np, dtype=tf.float32), axis=0))
            self.gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
            zero_indexed_groundtruth_classes = indices_class-1
            self.gt_classes_one_hot_tensors.append(tf.one_hot(
                zero_indexed_groundtruth_classes, self.num_classes))

        print('Done prepping data.')

    def build_model(self):

        tf.keras.backend.clear_session()
        print('Building model and restoring weights for fine-tuning...', flush=True)

        # Load pipeline config and build a detection model.
        #
        # Since we are working off of a COCO architecture which predicts 90
        # class slots by default, we override the `num_classes` field here to be just
        # one (for our new rubber ducky class).
        configs = config_util.get_configs_from_pipeline_file(self.pipeline_config)
        model_config = configs['model']
        model_config.ssd.num_classes = self.num_classes
        model_config.ssd.freeze_batchnorm = True
        detection_model = model_builder.build(
            model_config=model_config, is_training=True)
        
        self.model_config = configs

        
        # Set up object-based checkpoint restore --- RetinaNet has two prediction
        # `heads` --- one for classification, the other for box regression.  We will
        # restore the box regression head but initialize the classification head
        # from scratch (we show the omission below by commenting out the line that
        # we would add if we wanted to restore both heads)
        fake_box_predictor = tf.compat.v2.train.Checkpoint(
            _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
            #_prediction_heads=detection_model._box_predictor._prediction_heads,
            #    (i.e., the classification head that we *will not* restore)
            _box_prediction_head=detection_model._box_predictor._box_prediction_head,
            )
        fake_model = tf.compat.v2.train.Checkpoint(
                _feature_extractor=detection_model._feature_extractor,
                _box_predictor=fake_box_predictor)
        ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
        ckpt.restore(self.checkpoint_path).expect_partial()

        # Run model through a dummy image so that variables are created
        image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
        prediction_dict = detection_model.predict(image, shapes)
        _ = detection_model.postprocess(prediction_dict, shapes)

        self.detection_model = detection_model
        print('Weights restored!')

    
    def get_model_train_step_function(self, batch_size, model, optimizer, vars_to_fine_tune):
        # Set up forward + backward pass for a single train step.
        """Get a tf.function for training step."""

        # Use tf.function for a bit of speed.
        # Comment out the tf.function decorator if you want the inside of the
        # function to run eagerly.
        @tf.function
        def train_step_fn(image_tensors,
                            groundtruth_boxes_list,
                            groundtruth_classes_list):
            """A single training iteration.

            Args:
            image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
                Note that the height and width can vary across images, as they are
                reshaped within this function to be 640x640.
            groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
                tf.float32 representing groundtruth boxes for each image in the batch.
            groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
                with type tf.float32 representing groundtruth boxes for each image in
                the batch.

            Returns:
            A scalar tensor representing the total loss for the input batch.
            """
            shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
            model.provide_groundtruth(
                groundtruth_boxes_list=groundtruth_boxes_list,
                groundtruth_classes_list=groundtruth_classes_list)

            with tf.GradientTape() as tape:
                preprocessed_images = tf.concat(
                    [model.preprocess(image_tensor)[0]
                    for image_tensor in image_tensors], axis=0)
                prediction_dict = model.predict(preprocessed_images, shapes)
                losses_dict = model.loss(prediction_dict, shapes)
                total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
                gradients = tape.gradient(total_loss, vars_to_fine_tune)
                optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
            return total_loss

        return train_step_fn

    def model_training(self):

        tf.keras.backend.set_learning_phase(True)

        # These parameters can be tuned; since our training set has 5 images
        # it doesn't make sense to have a much larger batch size, though we could
        # fit more examples in memory if we wanted to.
        batch_size = 4
        learning_rate = 0.01 #orginally 0.01
        num_batches = 100
        #num_batches = 100

        # Select variables in top layers to fine-tune.
        trainable_variables = self.detection_model.trainable_variables
        to_fine_tune = []
        prefixes_to_train = [
        'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
        'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
        for var in trainable_variables:
            if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
                to_fine_tune.append(var)

        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        train_step_fn = self.get_model_train_step_function(batch_size,
            self.detection_model, optimizer, to_fine_tune)

        print('Start fine-tuning!', flush=True)
        for idx in range(num_batches):
            # Grab keys for a random subset of examples
            all_keys = list(range(len(self.train_images_np)))
            random.shuffle(all_keys)
            example_keys = all_keys[:batch_size]

            # Note that we do not do data augmentation in this demo.  If you want a
            # a fun exercise, we recommend experimenting with random horizontal flipping
            # and random cropping :)
            
            gt_boxes_list = [self.gt_box_tensors[key] for key in example_keys]
            gt_classes_list = [self.gt_classes_one_hot_tensors[key] for key in example_keys]
            image_tensors = [self.train_image_tensors[key] for key in example_keys]

            # Training step (forward pass + backwards pass)
            total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

            if idx % 10 == 0:
                print('batch ' + str(idx) + ' of ' + str(num_batches)
                + ', loss=' +  str(total_loss.numpy()), flush=True)

        print('Done fine-tuning!')

    def save_model(self):
        """ save the model"""

        # Save new pipeline config
        new_pipeline_proto = config_util.create_pipeline_proto_from_configs(self.model_config)
        config_util.save_pipeline_config(new_pipeline_proto, self.config_export_path)

        exported_ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt_manager = tf.train.CheckpointManager(
                exported_ckpt, directory=self.model_export_path, max_to_keep=5)
        
        ckpt_manager.save()
        print('Checkpoint saved!')

    def detect(self, input_tensor):
        """Run detection on an input image.

        Args:
            input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
            Note that height and width can be anything since the image will be
            immediately resized according to the needs of the model within this
            function.

        Returns:
            A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
            and `detection_scores`).
        """
        preprocessed_image, shapes = self.detection_model.preprocess(input_tensor)
        prediction_dict = self.detection_model.predict(preprocessed_image, shapes)
        return self.detection_model.postprocess(prediction_dict, shapes)
    
    def detect_test(self, test_model,input_tensor):
        """Run detection on an input image.

        Args:
            input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
            Note that height and width can be anything since the image will be
            immediately resized according to the needs of the model within this
            function.

        Returns:
            A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
            and `detection_scores`).
        """
        preprocessed_image, shapes = test_model.preprocess(input_tensor)
        prediction_dict = test_model.predict(preprocessed_image, shapes)
        return test_model.postprocess(prediction_dict, shapes)
    
    def viz_images(self, file_names_images):
        """ save visualized images after detection """

        test_images_np = []
        for file_name in file_names_images:
            image_item_path = self.training_image_data_path+"/"+file_name
            test_images_np.append(np.expand_dims(
                self.load_image_into_numpy_array(image_item_path), axis=0))

        
  
        label_id_offset = 1
        for i in range(len(test_images_np)):
            input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
            detections = self.detect(input_tensor)

            image_np = tf.convert_to_tensor(test_images_np[i])
            boxes = detections['detection_boxes']
            classes = tf.dtypes.cast(detections['detection_classes']+label_id_offset, tf.int8)
            scores = detections['detection_scores']
            category_index = self.category_index

            print("shape of image_np:\n", image_np.shape)
            print('\n images_np : \n', image_np)
            print("\n boxes: \n", boxes)
            print("\n classes: \n", classes)
            print("\n scores: \n", scores)

            img_tensor = viz_utils.draw_bounding_boxes_on_image_tensors(

                image_np,
                boxes,
                classes,
                scores,
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.7)
        
            detection_arr = np.squeeze(img_tensor)

            try:
                plt.imshow(detection_arr)
                plt.savefig(self.detected_image_path+"/detected_"+str(i)+'.png')
            except Exception as e:
                print('An error occured while saveing the plot: ', str(e))

    def loaded_model(self,file_names_images):

        tf.keras.backend.clear_session()
        
        configs = config_util.get_configs_from_pipeline_file(self.config_export_path+'/pipeline.config')
        model_config = configs['model']
        test_model = model_builder.build(model_config=model_config, is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=test_model)
        ckpt.restore(self.model_export_path+'/ckpt-1').expect_partial()

        test_images_np = []
        for file_name in file_names_images:
            image_item_path = self.training_image_data_path+"/"+file_name
            test_images_np.append(np.expand_dims(
                self.load_image_into_numpy_array(image_item_path), axis=0))

        label_id_offset = 1
        for i in range(len(test_images_np)):
            input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
            detections = self.detect_test(test_model, input_tensor)

            image_np = tf.convert_to_tensor(test_images_np[i])
            boxes = detections['detection_boxes']
            classes = tf.dtypes.cast(detections['detection_classes']+label_id_offset, tf.int8)
            scores = detections['detection_scores']
            category_index = self.category_index

            img_tensor = viz_utils.draw_bounding_boxes_on_image_tensors(

                image_np,
                boxes,
                classes,
                scores,
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.7)
        
            detection_arr = np.squeeze(img_tensor)

            try:
                plt.imshow(detection_arr)
                plt.savefig(self.detected_image_path+"/detected_loaded_"+str(i+10)+'.png')
            except Exception as e:
                print('An error occured while saveing the plot: ', str(e))


    
    def main(self):

        sorted_file_names = self.sort_filenames() #create list of file names of images in ascending order
        numpy_image = self.np_image(sorted_file_names) #convert images to numpy
        self.bbox = self.load_numpy(self.bbox_path) #load bbox
        self.indices_class = self.load_numpy(self.indices_class_path) #load class indices
        self.tensor_ground_truth() #create tensors for image, bbox, class
        self.build_model()
        self.model_training()
        self.viz_images(sorted_file_names)
        self.save_model()
        self.loaded_model(sorted_file_names)

        print("\n training complete !")
        

# the following few lines need to paramaterized

#define category index
num_classes = 3
input_field_id = 1
dropdown_id = 2
text_id = 3

category_index = {
    input_field_id: {'id': input_field_id, 'name': 'Input Field'},
    dropdown_id:{'id':dropdown_id, 'name':'Drop Down'},
    text_id:{'id':text_id, 'name':'Text'}
    }


ObjectDetection('~/development/ILS/object_detection/data','~/development/ILS/object_detection/data/bbox.npy','~/development/ILS/object_detection/data/indices_class_list.npy', category_index, 3, '~/development/tensorflow_models/models-master/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config', '~/development/assets/checkpoint_retinanet/ckpt-0', '~/development/ILS/object_detection/retinanet/trained_models/checkpoints','~/development/ILS/object_detection/retinanet/trained_models/configs', '~/development/ILS/object_detection/retinanet/detected_images').main()
