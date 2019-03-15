# import the necessary packages
import helpers

class ObjectDetector:
	def __init__(self, model, desc):
		# store the classifier and HOG descriptor
		self.model = model
		self.desc = desc

	def detect(self, image, winDim, winStep=4, pyramidScale=1.5, minProb=0.7):
		# initialize the list of bounding boxes and associated probabilities
		boxes = []
		probs = []

		# loop over the image pyramid
		for i,layer in enumerate(helpers.pyramid(image, scale=pyramidScale, minSize=winDim)):
			
			# determine the current scale of the pyramid
			scale = image.shape[0] / float(layer.shape[0])
			print("[INFO] Investigating layer {} at scale: {}".format(i,scale))
			# loop over the sliding windows for the current pyramid layer
			counter =0
			for (x, y, window) in helpers.sliding_window(layer, winStep, winDim):
				# grab the dimensions of the window
				(winH, winW) = window.shape[:2]
				counter = counter+1
				# ensure the window dimensions match the supplied sliding window dimensions
				if winH == winDim[1] and winW == winDim[0]:
					# extract HOG features from the current window and classifiy whether or
					# not this window contains an object we are interested in
					#print("[INFO] Extracting HOG features")
					features = self.desc.describe(window).reshape(1, -1)
					prob = self.model.predict_proba(features)[0][1]
					if counter % 200 ==0:
						print("[INFO] Model Probability: {}  Loop: {}   KeyPoint Top Left Corner (x,y) {}".format(prob, counter,[x,y]))

					# check to see if the classifier has found an object with sufficient
					# probability
					if prob > minProb:
						print("[INFO] ********** Found a candidate! **************")
						# compute the (x, y)-coordinates of the bounding box using the current
						# scale of the image pyramid
						(startX, startY) = (int(scale * x), int(scale * y))
						endX = int(startX + (scale * winW))
						endY = int(startY + (scale * winH))

						# update the list of bounding boxes and probabilities
						boxes.append((startX, startY, endX, endY))
						probs.append(prob)

		# return a tuple of the bounding boxes and probabilities
		return (boxes, probs)
