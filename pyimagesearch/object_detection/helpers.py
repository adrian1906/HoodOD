# import the necessary packages
import imutils
import cv2

def crop_ct101_bb(image, bb, padding=10, dstSize=(60, 60)):
	# unpack the bounding box, extract the ROI from the image, while taking into account
	# the supplied offset
	(y, h, x, w) = bb # Looks like this is y1,y2,x1,x2
	#print("y,h,x,w ={} {} {} {}".format(y,h,x,w))
	(x, y) = (max(x - padding, 0), max(y - padding, 0))
	roi = image[y:h + padding, x:w + padding]
	#print("ROI: {}".format(roi))
	# resize the ROI to the desired destination size
	# It is important to resize the roi in order to keep the final feature vector the same size	
	roi = cv2.resize(roi, dstSize, interpolation=cv2.INTER_AREA)

	# return the ROI
	return roi

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			#print("X: {}".format(x))
			#print("Y: {}".format(y))
			#print("Window Shape Check: {}".format(image.shape[:2]))
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
