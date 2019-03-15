# PRODUCTION VERSION
# This code is used to train an object detector based on DLIB.
# The goal is to be able to detect objects within images
# This can be used for face detection, stop sign detection, racing bib detection
# license plate detection, etc.
# It assumes that the xml file has the following form:
#<?xml version="1.0" encoding="UTF-8"?>
#<dataset>
#    <name>dataset containing bounding box labels on images</name>
#    <comment>created by BBTag</comment>
#    <tags>
#        <tag name="JOLabels" color="#032585"/>
#    </tags>
#    <images>
#        <image file="/home/adriansr/Pictures/SprintPhotos_Small/dsc_4136.jpg">
#            <box top="496" left="528" width="251" height="133">
#                <label>JOLabels</label>
#            </box>
#        </image>
#    </images>
#</dataset>

# All that is required is the .xml file, images that the .xml file references, and the name of the target Support Vector Model.
# The DLIB method takes only a few parameters:
#
# USAGE

# python train_detector_Hood.py -xml StopSigns.xml -out output/stop_sign_detector.svm

# import the necessary packages
from __future__ import print_function
import argparse
import dlib
from lxml import etree


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-xml", "--myxmlfile",required=True, help="Path to dataset html file")
ap.add_argument("-out", "--output", required=True, help="Path to the output detector")
args = vars(ap.parse_args())

# grab the default training options for our HOG + Linear SVM detector initialize the
# list of images and bounding boxes used to train the classifier
print("[INFO] gathering images and bounding boxes...")
options = dlib.simple_object_detector_training_options()
images = []
boxes = []

print("[INFO] Parsing XML File")
#doc = etree.parse(args["myxmlfile"])
#MyXML=doc.find('images')


### Note: I cannot use params here. I would first have to extract the features for the
# train the object detector
# Options are found here:  dlib.net/python/#dlib.simple_object_detector_training_options
options.C =5 					# SVM regularization parameter (Higher ==> more strict when finding a separating line)
options.num_threads=2				# Set to Number of Cores on  your machine
options.be_verbose = True			# Print out training information
#options.epsilon=.001				# Stopping Epsilon
options.eps=.001					# eps is used in the C++ file)
options.upsample_limit = 5			# 2 is the defualt. Max number of time it will upsample before throwing an exception
options.u=2					# Upsample Limit (based on DLIB C++ file)
#options.h					# Display help
options.add_left_right_image_flips = False 	# Allow an image to be used twice...once regular and once flipped.
#options.t					# Train and send output to disk
#options.cross-validate = True			# Perform Cross validation on an image dataset and print the results
options.test = True				# Test a trained detector on an image dataset and print the results
#options.folds					# When doing cross-validatation, do <arg> folds (default: 3)
#options.target-size				# Set size of the sliding window to about <arg> pixel
#options.flip					# Add left/right flipped copies of the images into the training

print("[INFO] training detector...")
print("Opening {}".format(args["myxmlfile"]))

detector = dlib.train_simple_object_detector(args["myxmlfile"], args["output"], options)

# Test on same data
print("Training accuracy: {}".format(
dlib.test_simple_object_detector(args["myxmlfile"], args["output"])))

