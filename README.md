# ILS: Intelligent Learning System

This project is intended to set foundation to enable algorithms to learn any system presented to it. Start with simple use case of algorithms to be able to navigate webpages given instruction in natural language.

The following is the outline of how project is structured.

object_detection: this folder contains algorithms for detecting objects such as input fields. Note that for training reintanet algorithm to be run on linux or windows machine with object detection API to be set up.

systems: this folder contains mock webpages or other UI or API to be trained

pipeline: this folder contains pipelines. Currently: systems (webpage) ---> object_detection(data)

labelimg: this folder contains script to create bbox using 'labelImg' library

ocr: this folder contains scripts to extract text from the system under test

integration: this folder contains integration scripts
