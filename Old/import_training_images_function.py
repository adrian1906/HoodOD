from scipy import io
from lxml import etree
from imutils import paths
import random
import progressbar
import cv2
from pyimagesearch.object_detection import helpers

def import_with_Matlab(conf,hog):
    data = []   
    labels = []
    # grab the set of ground-truth images and select a percentage of them for training
    trnPaths = list(paths.list_images(conf["image_dataset"]))
    trnPaths = random.sample(trnPaths, int(len(trnPaths) * conf["percent_gt_images"]))
    print("[INFO] describing training ROIs...")
    # setup the progress bar
    widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(trnPaths), widgets=widgets).start()
    # loop over the training paths
    for (i, trnPath) in enumerate(trnPaths):
        	# load the image, convert it to grayscale, and extract the image ID from the path
        	image = cv2.imread(trnPath)
        	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        	imageID = trnPath[trnPath.rfind("_") + 1:].replace(".jpg", "")
        
        	# load the annotation file associated with the image and extract the bounding box
        	p = "{}/annotation_{}.mat".format(conf["image_annotations"], imageID)
        	bb = io.loadmat(p)["box_coord"][0] #(y,h,x,w)
        	roi = helpers.crop_ct101_bb(image, bb, padding=conf["offset"], dstSize=tuple(conf["window_dim"]))
        
        	# define the list of ROIs that will be described, based on whether or not the
        	# horizontal flip of the image should be used
        	rois = (roi, cv2.flip(roi, 1)) if conf["use_flip"] else (roi,)
        
        	# loop over the ROIs
        	for roi in rois:
        		# extract features from the ROI and update the list of features and labels
        		features = hog.describe(roi)
        		data.append(features)
        		labels.append(1)
    
    	# update the progress bar
        	pbar.update(i)
    return  features,data,labels



def import_from_XML(conf,hog):
    data = []   
    labels = []
    print("Importing: {}".format(conf["image_dataset_XML"])) 
    doc = etree.parse(conf["image_dataset_XML"])
    MyXML=doc.find('images')
    widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(doc.xpath(".//*")), widgets=widgets).start()
    # loop over the training paths
    #for (i, info) in enumerate(MyXML):
    i = 0
    for info in MyXML:
    	# load the image, convert it to grayscale, and extract the image ID from the path
        	imagename=info.get('file')
		print("Working on file: {}".format(imagename))
        	image = cv2.imread(imagename)
		#cv2.imshow("My Image",image)
		#cv2.waitKey(0)
        	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        	y=int(info[0].get('top'))
        	x=int(info[0].get('left'))
        	w=int(info[0].get('width'))
        	h=int(info[0].get('height')) 
        	bb =[int(y),int(y)+int(h),int(x),int(x)+int(w)]  # [ y h x w] % (Look into h may be top and top/y may actually be h
		#print("bb: {}".format(bb))
		#newimage=image[y:y+h,x:x+w]
      	
		roi = helpers.crop_ct101_bb(image, bb, padding=conf["offset"], dstSize=tuple(conf["window_dim"]))
		#cv2.imshow("Cropped Image",roi)
		#cv2.waitKey(0)         
        	# define the list of ROIs that will be described, based on whether or not the
        	# horizontal flip of the image should be used
        	rois = (roi, cv2.flip(roi, 1)) if conf["use_flip"] else (roi,)
        
        	# loop over the ROIs
        	for roi in rois:
        		# extract features from the ROI and update the list of features and labels
        		features = hog.describe(roi)
        		data.append(features)
        		labels.append(1)
        
        	# update the progress bar
        	pbar.update(i)
		i=i+1
    return  features,data,labels
