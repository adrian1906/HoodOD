from bodydetector import *
# DOES NOT WORK import bibtagger.bobydetector as bt
import dlib
import os
import simplejson as json
import sys

class Conf:
    def __init__(self, confPath):
        # load and store the configuration and update the object's dictionary
        conf = json.loads(open(confPath).read())
        self.__dict__.update(conf)

    def __getitem__(self, k):
        # return the value associated with the supplied key
        return self.__dict__.get(k, None)


def plotresults(image,boxes,outfolder,image_Filename):
	orig = image.copy()
	if len(boxes) <1:
	    print("The object was not found")
	else:
	    # loop over the original bounding boxes and draw them
	    print("{} boxes found".format(len(boxes)))
	    for (startX, startY, endX, endY) in boxes:
        	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 10)

	FN=os.path.split(image_Filename)
	resultsimagepath=outfolder + FN[1]
	print("Saving: {}".format(resultsimagepath))
	cv2.imwrite(resultsimagepath, orig );


def plot_DLIB_results(img,dets,outfolder,image_Filename):

	orig=img.copy()
	for k, d in enumerate(dets):
		#print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
		x1new,y1new,x2new,y2new = new_rectangle(orig,d)
		cv2.rectangle(orig,(x1new,y1new),(x2new,y2new),(0,255,0),2)    
		#cv2.rectangle(orig,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,0),2)

	FN=os.path.split(image_Filename)
	resultsimagepath=outfolder + FN[1]
	print("Saving: {}".format(resultsimagepath))
	cv2.imwrite(resultsimagepath, orig);

	#cv2.imshow("myImage",orig)         
	#cv2.waitKey(0)


def new_rectangle(image,rect):
	h,w,d=image.shape

	x1=rect.left()
	y1=rect.top()
	x2=rect.right()
	y2=rect.bottom()
	#print("x1 {}  y1 {} x2 {}  y2 {}".format(x1,y1,x2,y2))
	face_width=x2-x1
	face_height=y2-y1

	# assume torso is 3*face
	x1new=x1-face_width
	x2new=x2+face_width
	y1new=y1
	y2new=y1+5*face_height  # 4 is tight and get most. 5 will be the insurance box size
	#print("face_width {}  face_height {}".format(face_width,face_height))
	#print("x1new {}  y1new {} x2new {}  y2new {}".format(x1new,y1new,x2new,y2new))
	return(x1new,y1new,x2new,y2new)


#exts = [".jpg",".bmp"]
conf=Conf("/home/HoodML/bib-tagger-master/bibtagggerConfigFile.json")
sourcefolder = conf["image_dataset"]
outfolder = conf["image_results_folder"]
print("Source Folder: {}".format(sourcefolder))
print("Results Folder: {}".format(outfolder))

detector = dlib.get_frontal_face_detector()
#win = dlib.image_window()

filenames = sorted(os.listdir(sourcefolder))
#print("filenames")
#print(filenames)

Method ="2"
for filename in filenames:
	#name,ext = os.path.split(filename)

	if ".jpg" in filename:
		print("Processing image {}".format(sourcefolder + filename))

		if Method =="1":
			image=cv2.imread(sourcefolder + filename)
			#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			faces=findfaces(image)
			bodyrectangles=findbodies(image,faces)
			plotresults(image,bodyrectangles,outfolder,filename)

		else:
		    img = dlib.load_rgb_image(sourcefolder + filename)
		    # The 1 in the second argument indicates that we should upsample the image
		    # 1 time.  This will make everything bigger and allow us to detect more
		    # faces.
		    dets = detector(img, 1)
		    print("Number of faces detected: {}".format(len(dets)))
		    for i, d in enumerate(dets):
		        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
		            i, d.left(), d.top(), d.right(), d.bottom()))

		    plot_DLIB_results(img,dets,outfolder,sourcefolder + filename)

		    #win.clear_overlay()
		    #win.set_image(img)
		    #win.add_overlay(dets)
		    #dlib.hit_enter_to_continue()


			# Finally, if you really want to you can ask the detector to tell you the score
			# for each detection.  The score is bigger for more confident detections.
			# The third argument to run is an optional adjustment to the detection threshold,
			# where a negative value will return more detections and a positive value fewer.
			# Also, the idx tells you which of the face sub-detectors matched.  This can be
			# used to broadly identify faces in different orientations.
			# if (len(filenames) > 0):
			#     img = dlib.load_rgb_image(sys.argv[1])
			#     dets, scores, idx = detector.run(img, 1, -1)
			#     for i, d in enumerate(dets):
			#         print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))