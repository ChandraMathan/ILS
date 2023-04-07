Follow the steps to install object detection api (note only works in windos and linux):

- Clone the tensorflow models repository: https://github.com/tensorflow/models
- using terminal follow these:
  - cd models/research/
  - protoc object_detection/protos/\*.proto --python_out=.
  - cp object_detection/packages/tf2/setup.py .
  - python -m pip install .
