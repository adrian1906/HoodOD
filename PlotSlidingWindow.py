from Hood_ObjectDetection_module import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-cf", "--conf", required=True, help="path to the configuration file")
ap.add_argument("-s", "--start_index", required=True, help="index of 1st file to analyze")

args = vars(ap.parse_args())
# load the configuration file and initialize the list of widths and heights
conf = Conf(args["conf"])
testfilefolder=conf["test_files"]
TestFiles=glob.glob(testfilefolder + "*.jpg")
stepSize=conf["window_step"]
windowSize=(128,80)
start = int(args["start_index"])
for i in range(start,start+5):
	image = cv2.imread(TestFiles[i])
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	if windowSize[0]>windowSize[1]: # Based on object aspect ratio
		(newHeight,newWidth) = rescaleImage(image, 512, "L")
	else:
		(newHeight,newWidth) = rescaleImage(image, 512, "P")

	orig=image.copy()
	image=cv2.resize(image,(newWidth,newHeight))
	minSize=windowSize
	scale = 1.5
	keepscale=[]
	keeplayer=[]

	for i,layer in enumerate(pyramid(image, scale, minSize)):
		keepscale.append(image.shape[0] / float(layer.shape[0]))
		keeplayer.append(layer)


	for j in range(0,len(keepscale)):
		print("Scale: {}".format(keepscale[j]))
		#cv2.imshow("Image", orig)	
		#cv2.imshow("Resized", image)
		#cv2.waitKey(0)
		stepSize = 16
		x,y,windows = sliding_window_return(keeplayer[j], stepSize, windowSize)
		plot_sliding_window(keeplayer[j],stepSize,windowSize,x,y,25)