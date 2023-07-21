import os
import glob
import numpy as np
import tensorflow as tf
from six import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import argparse
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

class InferenceRetinanet:

    def __init__(self, image_path, checkpoint_path, detected_image_path, category_index_path, pipeline_config, num_classes):

        """
        Params:

        image_path: test image path
        checkpoint_path: checkpoint to be used for inference
        detected_image_path: image detection 
        category_index_path: its a dictionary of categories being defined (e.g 1,2,3, .. and the name of those)
        pipeline_config: config file to be used for inference
        num_classes: number of classess defined. type is int
        
        """
        self.image_path = os.path.expanduser(image_path)
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.detected_image_path = os.path.expanduser(detected_image_path)

        with open(os.path.expanduser(category_index_path), 'rb') as file:
            self.category_index = pickle.load(file)

        self.pipeline_config = os.path.expanduser(pipeline_config)
        self.num_classes =num_classes

        self.images_np = []
        self.model = None

    def get_file_names(self):
        """ get all .jpg files in the path """

        file_names_images = []
        for file in glob.glob(self.image_path+"/*.jpg"):
            file_names_images.append(file)

        return file_names_images
    
    def load_image_into_numpy_array(self, path):
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
    
    def images_to_np(self, file_names_images):

        for file_name in file_names_images:
            self.images_np.append(np.expand_dims(
                self.load_image_into_numpy_array(file_name), axis=0))

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
        preprocessed_image, shapes = self.model.preprocess(input_tensor)
        prediction_dict = self.model.predict(preprocessed_image, shapes)
        return self.model.postprocess(prediction_dict, shapes)
    
    def build_model_weights(self):
        """ build model and restore wights """

        tf.keras.backend.clear_session()

        print('Building model and restoring weights for inference ....', flush=True)

        configs = config_util.get_configs_from_pipeline_file(self.pipeline_config+'/pipeline.config')
        model_config = configs['model']
        test_model = model_builder.build(model_config=model_config, is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=test_model)
        ckpt.restore(self.checkpoint_path+'/ckpt-1').expect_partial()

        self.model = test_model

        print('Weights restored!')

    def bbox_class (self,image_np):
        """ predict boounding box and classes """
        label_id_offset = 1
        input_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
        detections = self.detect(input_tensor)
        boxes = detections['detection_boxes']
        classes = tf.dtypes.cast(detections['detection_classes']+label_id_offset, tf.int8)
        scores = detections['detection_scores']

        return boxes, classes, scores
    
    def viz_images(self, image_np, boxes, classes, scores,img_index):
        """ plot bbox and save images """

        img_tensor = viz_utils.draw_bounding_boxes_on_image_tensors(

                tf.convert_to_tensor(image_np),
                boxes,
                classes,
                scores,
                self.category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.7)
        
        detection_arr = np.squeeze(img_tensor)

        try:
            plt.imshow(detection_arr)
            plt.savefig(self.detected_image_path+"/detected_"+str(img_index)+'.png')
        except Exception as e:
            print('An error occured while saveing the plot: ', str(e))


    def main(self):

        file_names = self.get_file_names()
        self.images_to_np(file_names)
        self.build_model_weights()
        for i in range(len(self.images_np)):
            boxes, classes, scores = self.bbox_class (self.images_np[i])
            self.viz_images(self.images_np[i], boxes, classes, scores,i)
        print("Inference completed !")

def check_args(parser, *args):
    """ Check if the arg values are None """

    # Parse the command-line arguments
    args, unknown = parser.parse_known_args()

    # Iterate over the defined arguments
    for arg in vars(args):
        arg_value = getattr(args, arg)
        if arg_value == None:
            return False
    
    return True


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Inference using Retinanet')

    # Add arguments
    parser.add_argument('-i_img', '--image_path', help = 'Test Image')
    parser.add_argument('-i_ckpt', '--checkpoint_path', help='Checkpoint input file for inference')
    parser.add_argument('-o_ckpt', '--detected_image_path', help='Vizualization: detected objects with bbox in png format')
    parser.add_argument('-i_cat', '--category_index_path', help='Category index dictionary')
    parser.add_argument('-i_cfg', '--pipeline_config', help='Pipeline Config input file for inference')
    parser.add_argument('-c', '--num_classes', help='Number of classes')

    # Parse the command-line arguments
    args = parser.parse_args()

    if check_args(parser, args) == True:
        # Call the main function with the argument values
        inf_ins = InferenceRetinanet(args.image_path, args.checkpoint_path,args.detected_image_path,args.category_index_path,args.pipeline_config,args.num_classes)
        inf_ins.main()


