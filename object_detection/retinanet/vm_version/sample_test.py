#run this to ensure all the libraries are correctly installed

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
import argparse
import pickle
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.meta_architectures import ssd_meta_arch

print("\n \n \n --------- \n No errors: Installation Sucessfull")