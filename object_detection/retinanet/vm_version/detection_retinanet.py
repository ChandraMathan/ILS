import os
import glob
import numpy as np
import tensorflow as tf
from six import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

class DetectionRetinanet:

    def __init__(self, image_path, checkpoint_path, detected_image_path, category_index, pipeline_config, num_classes):

        self.image_path = os.path.expanduser(image_path)
        #self.model = tf.saved_model.load(os.path.expanduser(model_path))
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.detected_image_path = os.path.expanduser(detected_image_path)
        self.category_index = category_index
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
            model_config=model_config, is_training=False)

        ckpt = tf.train.Checkpoint(model=detection_model)
        ckpt.restore(tf.train.latest_checkpoint(self.checkpoint_path)).expect_partial()

        self.model = detection_model #.signatures['serving_default']
        print('Weights restored!')

    def viz_images(self):
        """ save visualized images after detection """

        label_id_offset = 1
        for i in range(len(self.images_np)):
            input_tensor = tf.convert_to_tensor(self.images_np[i], dtype=tf.float32)
            detections = self.detect(input_tensor)

            image_np = tf.convert_to_tensor(self.images_np[i])
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
            print(" \n viz : detection_boxes \n", boxes)
            print(" \n viz : scores \n", scores)

            try:
                plt.imshow(detection_arr)
                plt.savefig(self.detected_image_path+"/detected_"+str(i)+'.png')
            except Exception as e:
                print('An error occured while saveing the plot: ', str(e))

    def main(self):

        file_names = self.get_file_names()
        self.images_to_np(file_names)
        self.build_model_weights()
        self.viz_images()


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

DetectionRetinanet('~/development/ILS/object_detection/data', '~/development/ILS/object_detection/retinanet/trained_models/checkpoints', '~/development/ILS/object_detection/retinanet/detected_images', category_index,'~/development/tensorflow_models/models-master/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config',3).main()