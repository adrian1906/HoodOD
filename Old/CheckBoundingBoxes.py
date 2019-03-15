# PRODUCTION VERSION
# The purpose of this file is to verify the bounding boxes.

# python CheckBoundingBoxes -xml Filename.xml 

# import the necessary packages
from __future__ import print_function
import argparse
from lxml import etree
import subprocess



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-xml", "--myxmlfile",required=True, help="Path to dataset html file")
args = vars(ap.parse_args())

# Calls the imglab program using the xml file as the argument
# It is assumed that a link to the imglab.sh file resides in a folder 
# that is in the path (ie /usr/share)
CallString="/home/adriansr/Downloads/dlib-19.2/tools/imglab/build$/imglab " + args["myxmlfile"]
print("Call String {}".format(CallString))
subprocess.call([CallString])

