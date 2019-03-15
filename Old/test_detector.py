# USAGE
# python test_detector.py --detector output/stop_sign_detector.svm --testing stop_sign_testing

# import the necessary packages
from __future__ import division # Needed to fix the issue with 3/2=1   division by two integer ==> integer (geez)
from imutils import paths
import argparse
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True, help="Path to trained object detector")
ap.add_argument("-t", "--testing", required=True, help="Path to directory of testing images")
ap.add_argument("-n", "--numofimages", required = True, help="Number of images to test")
ap.add_argument("-time", "--time_ms", required = True, help="Duration to show image (ms)")
args = vars(ap.parse_args())

# load the detector
neww=600
print("Detector: {}".format(args["detector"]))
detector = dlib.simple_object_detector(args["detector"])
# loop over the testing images
#Flag = True
#try:
#    while Flag ==True:
counter =1
for testingPath in paths.list_images(args["testing"]):
  try:	
	if counter <=int(args["numofimages"]):
		image = cv2.imread(testingPath)
        	height,width = image.shape[:2]
        	h_to_w=height/width
        	newh=neww*h_to_w;
		newh=int(newh) # Needed to avoid an error when interpolating
		image=cv2.resize(image,(neww,newh),interpolation = cv2.INTER_CUBIC)
            	#cv2.imshow("Test Image {}".format(counter), image)
            	#cv2.waitKey(0)
		boxes = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
 		# loop over the bounding boxes and draw them
       		for b in boxes:
            		(x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
            		cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 5)
            		#print("BoxSize {}: {}".format([b]))
            		#show the image
            		#image=cv2.resize(image,(neww,newh),interpolation = cv2.INTER_CUBIC)
            	cv2.imshow("Test Image {}".format(counter), image)
		cv2.moveWindow("Test Image {}".format(counter),0,0)
            	cv2.waitKey(int(args["time_ms"]))
		cv2.destroyWindow("Test Image {}".format(counter))
  	else:
		break	
	print("Image: {}".format(counter))
	counter=counter+1
  except:
	print("********* Problem loading the file:. ********")
	print("********* " + testingPath)
	#print("*********   b= {}".format(b))
        print("********************************************")


