
# coding: utf-8

# # Object Detection  
# 
# This example program develops a HOG based object detector for things like faces, pedestrians, and any other semi-rigid object.  In particular, we go though the steps to train the kind of sliding window object detector first published by Dalal and Triggs in 2005 in the  paper Histograms of Oriented Gradients for Human Detection.
# 
# It is similar to the method implemented in dlib (more optimized). However, this technique allows more control of the parameters.
# 
# 

# ## Create the XML files
# DLIB requires images and bounding boxes around the labelled object. It has its own strructure for the XML files:
# 
# <?xml version="1.0" encoding="UTF-8"?>
# <dataset>
#     <name>dataset containing bounding box labels on images</name>
#     <comment>created by BBTag</comment>
#     <tags>
#         <tag name="RunBib" color="#032585"/>
#     </tags>
#     <images>
#         <image file="B:/DataSets/2016_USATF_Sprint_TrainingDataset/_hsp3997.jpg">
#             <box top="945" left="887" width="85" height="53">
#                 <label>RunBib</label>
#             </box>
#             <box top="971" left="43" width="103" height="56">
#                 <label>RunBib</label>
#             </box>
#             <box top="919" left="533" width="100" height="56">
#                 <label>RunBib</label>
#             </box>
#         </image>
#         <image file="B:/DataSets/2016_USATF_Sprint_TrainingDataset/_hsp3989.jpg">
#             <box top="878" left="513" width="111" height="62">
#                 <label>my_label</label>
#             </box>
#         </image>     
#    </images>
# </dataset>
# top: Top left y value
# height: Height (positive down)
# left: Top left x value
# width: Width (positive to the right)
# 
# To create your own XML files you can use the imglab tool which can be found in the tools/imglab folder.  It is a simple graphical tool for labeling objects in images with boxes.  To see how to use it read the tools/imglab/README.txt file.  But for this example, we just use the training.xml file included with dlib.
# 
# Its a two part process to load the tagger.
# 1.) typing the following command:
# #####    b:\HoodMachineLearning\dlib\tools\build\Release\imglab.exe -c mydataset.xml B:\HoodMachineLearning\datasets\MyImage
# 2.) 
# ####     b:\HoodMachineLearning\dlib\tools\build\Release\imglab.exe -c mydataset.xml

# ## Image pyramids and sliding windows
# 
# The technique uses image pyramids and sliding windows to minimize the effect of object location and object size. The pyramid is a set of subsample images and the sliding window remains the same and moves from left to right and top to bottom of each scale of the image.
# 
# ### Image Pyramids
# <img src="ImagePyramid.jpg">
# 
# Note: Dalai and Triggs showed that performance is reduced if you apply gaussian smoothing at each layer==> ski this stip
# 
# ### Sliding Window
# 
# <img src="sliding_window_example.gif" loop=3>
# 
# * It is common to use a stepSize of 4 to 8 pixels
# * windowSize is the size of the Kernal. An object detector will work best if the aspect ratio of the kernal is close to that of the desired object. Note: The sliding window size is also important for the HOG filter. For the HOG filter two parameters are important: <b>pixels_per_cell</b> and <b>cells_per_block </b>
# 
# In order to avoid having to 'guess' at the best window size that will satisfy object detector requirements and HOG requirments, a "explore_dims.py" method is used.
# 
# 1.) Meet object detection requirments: loads all the images and computes the average width, average height, and computes the aspect ratio from those values.
# 2.) Meet HOG requirments: Pyimage rule of thumb is to divide the above values by two (ie, 1/4th the average size)
#     * This reduces the size of the HOG feature vector
#     * By dividing by two, a nice balance is struck between HOG feature vector size and reasonable window size.
#     * Note: Our sliding_window dimension needs to be divisible by pixels_per_cell and cells_per_block so that the HOG descriptor will 'fit' into the window size
#     * Its common for 'pixels_per_cell' to be a multiple of 4 and cells_per_block in the set (1,2,3)
#     * Start with pixels_per_cell=(4,4) and cells_per_block=(2,2)
#     * For example, in the Pyimage example, average W: 184 and average H:62. Divide by 2 ==> 92,31
#     * Find values close to 92,31 that are divisible by 4 (and 2): 96,32  (Easy)
#     * OBSERVATION:  When defining the binding boxes, it is best if all are around the same size. This can be difficult.  
# 
# ### The 6 Step Framework
# 1. Sample P positive samples for your training data of the objects you want to detect. Extract HOG features from these objects.
#     * If given an a general image containing the object, bounding boxes will also need to be given that indicate the location of the image
# 2. Sample N negative samples that do not contain the object and extract HOG features. In general N>>P  (I'd suggest images similar in size and aspect ratio to the P samples. I'd also avoid the bounding boxes and make the entire image the negative image. Pyimagesearch recommends using the 13 Natural Scene Category of the vision.stanford.edu/resources_links.html page
# 3. Train a Linear Support Vector Machine (SVM) on the negative images (class 0) and positive image (class 1)
# 4. Hard Negative Mining - for the N negative images, apply Sliding window and test the classifier. Ideally, they should all return 0. If they return a 1 indicating an incorrect classification, add it to the training set (for the next round of re-training)
# 5. Re-train classifier using with the added images from Hard Negative Mining (Usually once is enough)
# 6. Apply against test dataset, define a box around regions of high probability, when finished with the image, find the boxed region with the highest probability using "non-maximum suppression" to removed redundant and overlapping bounding boxes and make that the final box.
# 
# #### Note on DLIB library
# * Similar to the 6 step framework but uses the entire training image to get the P's (indicated by bounding boxes) and the N's (not containing bounding boxes).  Note: It looks like it is important that all of the objects are identified in the image. For example, when doing running bibs, I may ignore some bibs for some reasons (too small, partially blocked, too many). My guess is that these images should just simply be avoided. This technique eliminates steps 2, 4, and 5.
# * non-maximum supression is applied during the trainig phase helping to reduce false positives
# * dlib using a highly accurate SVM engine used to find the hyperplane separating the TWO classes.
# 
# 
# #### Use a JSON file to hold the hyper-parameters
# {
# 
# "faces_folder": "B:\\DataSets\\2016_USATF_Sprint_TrainingDataset"
# "myTrainingFilename": "trainingset_small.xml"
# "myTestingFilename: "trainingset_small.xml"
# "myDetector": "detector.svm"
# }
# 
# #### Load and Dump hdf5 file
# * hdf5 provides efficient data storage
# 

# In[ ]:
# conda install -c conda-forge scikit-image
# conda install -c anaconda progressbar
# conda install -c anaconda simplejson
# conda install -c menpo opencv
# conda install -c anaconda h5py
# conda install -c conda-forge lxml
# conda install scikit-learn
from __future__ import print_function
from skimage import feature # 
##from skimage import exposure
from sklearn.svm import SVC
from sklearn.metrics import classification_report
##from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
##from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC 
import _pickle as cPickle

import numpy as np
import cv2
import h5py
import simplejson as json
##from sklearn.feature_extraction.image import extract_patches_2d
##import argparse
import random
import os
from scipy import io
from lxml import etree
import math
import glob
#from imutils import paths
##import progressbar
#import cv2
#from pyimagesearch.object_detection import helpers
# import the necessary packages

#import commentjson as json
#from pyimagesearch.object_detection import helpers
#from pyimagesearch.descriptors import HOG
#from pyimagesearch.utils import dataset
#from pyimagesearch.utils import Conf
##from imutils import paths
##from imutils import resize
#from scipy import io
##import progressbar
#import import_training_images_function2 as imp # Used to either import via Matlab file (CalTech) or XML (Scikit-learn)
#from skimage import exposure

#from dlib import progressbar


# In[ ]:

class HOG:
    def __init__(self, orientations=12, pixelsPerCell=(4, 4), cellsPerBlock=(2, 2), normalize=True):
        # store the number of orientations, pixels per cell, cells per block, and
        # whether normalization should be applied to the image
        self.orientations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.normalize = normalize

    def describe(self, image):
        # compute Histogram of Oriented Gradients features
        hist = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixelsPerCell,cells_per_block=self.cellsPerBlock, transform_sqrt=self.normalize, block_norm = "L2-Hys")
        hist[hist < 0] = 0

        # return the histogram
        return hist

    def describe_and_return_HOGImage(self, image):
        # compute Histogram of Oriented Gradients features
        (hist,hogImage) = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixelsPerCell,cells_per_block=self.cellsPerBlock, transform_sqrt=self.normalize, visualise=True)
        hist[hist < 0] = 0

        # return the histogram
        return hist,hogImage


# In[ ]:

# Referred to as helpers.py in PyImageSearch Class
def crop_ct101_bb(image, bb, padding=10, dstSize=(32, 32)):
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
        h = int(image.shape[0] / scale)        
        #image = imutils.resize(image, width=w)
        image = cv2.resize(image, (w,h))

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            #print("X: {}".format(x))
            #print("Y: {}".format(y))
            #print("Window Shape Check: {}".format(image.shape[:2]))
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# In[ ]:
from multiprocessing import Process,Queue
class ObjectDetector:
    def __init__(self, model, desc):
        # store the classifier and HOG descriptor
        self.model = model
        self.desc = desc

    def CalculateNumberOfScales(self,image,scale,minSize):
        keepscale=[]
        keeplayer=[]
        for i,layer in enumerate(pyramid(image, scale, minSize)):
            keepscale.append(image.shape[0] / float(layer.shape[0]))
            keeplayer.append(layer)
        
        
        return keepscale,keeplayer 

    def StartDection_MultiProcess(self,image,winDim, minSize,  winStep=4, pyramidScale=1.5, minProb=0.7):
        q=[]    # Queue List
        p=[]    # Process List
        d=[]    # Output List
        boxes=[]
        probs=[]
        (keepscale,keeplayer)=self.CalculateNumberOfScales(image,pyramidScale,minSize)
        numProcesses=len(keepscale)
        print("There are {} scales in this image.".format(numProcesses))
        for i in range(0,numProcesses):
            # Create a new process
            print("Adding process {} to list".format(i))
            q.append(Queue())
            p.append(Process(target=self.f,args=(q[i],image, keepscale[i],keeplayer[i], winDim, winStep, pyramidScale, minProb))) 
            #pp=p[i]
            #pp.start()
            
        #for j in range(0,numProcesses):
        
        #for j in p:
        for (jj, pp) in enumerate(p):
            print("Starting Process: {}".format(jj))
            pp.start()
            #pp=p[j]
            #pp.start()
            #pp.join()
        
        #for jj in range(0,numProcesses):
        #for (jj, pp) in enumerate(p):
        for (jj, pp) in enumerate(p):
        #for jj in p:
            pp.join()
            #ppp=p[j]
            #ppp.join()
    
        for k in range(0,numProcesses):
            print("Collecting results from process {}".format(k))
            d.append(q[k].get()) # Should return (data,labels)
            boxes = boxes + d[k][0] # Concatonate list
            probs = probs + d[k][1]
        
        print("Finished with this image. There were {} boxes found".format(len(boxes)))
        #(boxes, probs)=self.detect_single_layer(image, keepscale[i],keeplayer[i], winDim, winStep, pyramidScale, minProb)
        return(boxes,probs)
        
    def f(self,q,image, scale,layer, winDim, winStep, pyramidScale, minProb):
        print('parent process:', os.getppid())
        print('process id:', os.getpid())  
        #(data,labels) = Hard_Negative_Mining(conf,SW,scale)
        (boxes, probs)=self.detect_single_layer(image, scale,layer, winDim, winStep, pyramidScale, minProb)
        q.put([boxes,probs])
        
    def detect_single_layer(self, image, scale, layer, winDim, winStep=4, pyramidScale=1.5, minProb=0.7):
        # initialize the list of bounding boxes and associated probabilities
        boxes = []
        probs = []
        
        # loop over the image pyramid
        #for i,layer in enumerate(pyramid(image, scale=pyramidScale, minSize=winDim)):
            
            # determine the current scale of the pyramid
            #scale = image.shape[0] / float(layer.shape[0])
            #print("[INFO] Investigating layer {} at scale: {}".format(i,scale))
        print("[INFO] Investigating scale: {}".format(scale))
            # loop over the sliding windows for the current pyramid layer
        counter =0
        for (x, y, window) in sliding_window(layer, winStep, winDim):
                # grab the dimensions of the window
                (winH, winW) = window.shape[:2]
                counter = counter+1
                # ensure the window dimensions match the supplied sliding window dimensions
                if winH == winDim[1] and winW == winDim[0]:
                    # extract HOG features from the current window and classifiy whether or
                    # not this window contains an object we are interested in
                    #print("[INFO] Extracting HOG features")
                    features = self.desc.describe(window).reshape(1, -1)
                    #print("Object Detector Feature Size: {}".format(features.shape))
                    prob = self.model.predict_proba(features)[0][1]
                    if counter % 1000 ==0:
                        print("[INFO] Model Probability: {}  Loop: {}   KeyPoint Top Left Corner (x,y) {}".format(prob, counter,[x,y]))

                    # check to see if the classifier has found an object with sufficient
                    # probability
                    if prob > minProb:
                        ##print("[INFO] ********** Found a candidate! **************")
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




# In[ ]:

    
    def run_multiple_processes_using_lists(f,conf,SW,numProcesses=4):
        #numProcesses =5 # Number of processes
        # Function f needs to contain a q.put([]) statement to return the output
        q=[]    # Queue List
        p=[]    # Process List
        d=[]    # Output List
        datalist=[]
        labellist=[]
        for i in range(0,numProcesses-1):
            q.append(Queue())
            p.append(Process(target=f,args=(q[i],conf,SW))) 

    #print("*** Queue List ***")
    #print(q)
    #print("*** Process List ***")
    #print(p)
    
        print("Start All Processes ...")
        for j in range(0,numProcesses-1):
            pp=p[j]
            pp.start()
            pp.join()
    
        print("Collecting Results")
        for k in range(0,numProcesses-1):
            d.append(q[k].get()) # Should return (data,labels)
            datalist = datalist + d[k][0] # Concatonate list
            labellist=labellist + d[k][1]
    
        return(datalist,labellist)
    #print("Verify Output")
    #print(d[0]) # Print the first item in the returned list
    #for l in range(0,numProcesses-1):
    #    print(type(d[l]))

# In[ ]:

def non_max_suppression(boxes, probs, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this is important since
    # we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding boxes by their associated
    # probabilities
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of
        # picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the
        # smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater than the
        # provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")


# In[ ]:

def dump_dataset(data, labels, path, datasetName, writeMethod="w"):
    # open the database, create the dataset, write the data and labels to dataset,
    # and then close the database
    with h5py.File(path, writeMethod) as db:
        dataset = db.create_dataset(datasetName, (len(data), len(data[0]) + 1), dtype="float")
        dataset[0:len(data)] = np.c_[labels, data]
        db.close()
    print("Finished Dumping Data into: ".format(path))

def load_dataset(path, datasetName):
    # open the database, grab the labels and data, then close the dataset
    with h5py.File(path, "r") as db:
        (labels, data) = (db[datasetName][:, 0], db[datasetName][:, 1:])
        db.close()

    #list(db.keys())
    # return a tuple of the data and labels
    return (data, labels)


# In[ ]:

class Conf:
    def __init__(self, confPath):
        # load and store the configuration and update the object's dictionary
        conf = json.loads(open(confPath).read())
        self.__dict__.update(conf)

    def __getitem__(self, k):
        # return the value associated with the supplied key
        return self.__dict__.get(k, None)


# In[ ]:

def import_with_Matlab(conf,hog,SW):
    data = []   
    labels = []
    # grab the set of ground-truth images and select a percentage of them for training
    #trnPaths = list(paths.list_images(conf["image_dataset"]))
    trnPaths = list(os.listdir(conf["image_dataset"]))
    trnPaths = random.sample(trnPaths, int(len(trnPaths) * conf["percent_gt_images"]))
    print("[INFO] describing training ROIs...")
    # setup the progress bar
    #widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    #pbar = progressbar.ProgressBar(maxval=len(trnPaths), widgets=widgets).start()
    # loop over the training paths
    for (i, trnPath) in enumerate(trnPaths):
        # load the image, convert it to grayscale, and extract the image ID from the path
        image = cv2.imread(trnPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageID = trnPath[trnPath.rfind("_") + 1:].replace(".jpg", "")
        
        # load the annotation file associated with the image and extract the bounding box
        p = "{}/annotation_{}.mat".format(conf["image_annotations"], imageID)
        bb = io.loadmat(p)["box_coord"][0] #(y,h,x,w)
        # The next line crops the image to only the object. Because of this, no scanning is required
        # and the image size can simply be set to the scanning size (plus offset) so that only one scan is needed
        #roi = crop_ct101_bb(image, bb, padding=conf["offset_padding"], dstSize=tuple(conf["window_dim"]))
        roi = crop_ct101_bb(image, bb, padding=conf["offset_padding"], dstSize=SW)
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
        #    pbar.update(i)
    return  data,labels



def import_from_XML(conf,hog,SW):
    data = []   
    labels = []
    ##print("Importing: {}".format(conf["image_dataset_XML"])) 
    doc = etree.parse(conf["image_dataset_XML"])
    MyXML=doc.find('images')
    ## widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    ## pbar = progressbar.ProgressBar(maxval=len(doc.xpath(".//*")), widgets=widgets).start()
    # loop over the training paths
    #for (i, info) in enumerate(MyXML):
    i = 0
    for info in MyXML:
        # load the image, convert it to grayscale, and extract the image ID from the path
        try:
            imagename=conf["image_dataset"] + "\\" + info.get('file')
            #print("Working on file: {}".format(imagename))
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
            #roi = crop_ct101_bb(image, bb, padding=conf["offset"], dstSize=tuple(conf["window_dim"]))
            # The next line crops the image to only the object. Because of this, no scanning is required
            # and the image size can simply be set to the scanning size (plus offset) so that only one scan is needed
            #roi = crop_ct101_bb(image, bb, padding=conf["offset_padding"], dstSize=tuple(conf["image_resized"]))
            roi = crop_ct101_bb(image, bb, padding=conf["offset_padding"], dstSize=SW)
            #print("The image size is {}.".format(roi.shape))
            ##cv2.imshow(imagename,roi)
            ##cv2.waitKey(0)         
            # define the list of ROIs that will be described, based on whether or not the
            # horizontal flip of the image should be used
            if conf["use_flip"]:
                rois = (roi, cv2.flip(roi, 1))
            else:
                rois = (roi,)
                        
            # loop over the ROIs
            for roi in rois:
                # extract features from the ROI and update the list of features and labels
                status="computing features"
                features = hog.describe(roi)
                status="appending data"
                data.append(features)
                labels.append(1)
        
                # update the progress bar
                #pbar.update(i)
        except:
            print("Issue with file:  {}. Last stage: {}".format(imagename,status))
            
        i=i+1
        
    return  data,labels


# In[ ]:

def GetAvgDimensions(conf):
    doc = etree.parse(conf["image_dataset_XML"])
    MyXML=doc.find('images')
    widths = []
    heights = []
    for info in MyXML:
        ##imagename=conf["image_dataset"] + "\\" + info.get('file')
        widths.append(int(info[0].get('width')))
        heights.append(int(info[0].get('height')) )

    (avgWidth, avgHeight) = (np.mean(widths), np.mean(heights))
    (stdW,stdH)=(np.std(widths),np.std(heights))
    #print("The length of widths is {}".format(len(widths)))
    newW=math.ceil(int(avgWidth/2)/4)*4
    newH=math.ceil(int(avgHeight/2)/4)*4
    print("[INFO] avg. width: {:.2f} +/- {:.2f}".format(avgWidth,stdW))
    print("[INFO] avg. height: {:.2f} +/- {:.2f}".format(avgHeight,stdH))
    print("[INFO] aspect ratio: {:.2f}".format(avgWidth / avgHeight))
    print("[INFO] The recommended Sliding Window Size is W:{}  H:{}".format(newW,newH))
    print("[INFO] Sliding Window Aspect Ratio {:.2f}".format(newW / newH))
    return tuple([newW,newH])


# In[ ]:



def train_model(conf,useHardNegatives):
    # load the configuration file and the initial dataset
    print("[INFO] loading dataset...")
    #conf = Conf(args["conf"])
    (data, labels) = load_dataset(conf["features_path"], "features") # contains images labled as good (+1) and bad (-1)
    print("Example of a data point. Feature: {} and Label {}".format(data[0],labels[0]))
    if useHardNegatives > 0:
        print("[INFO] loading hard negatives...")
        (hardData, hardLabels) = load_dataset(conf["features_path"], "hard_negatives")
        data = np.vstack([data, hardData])  # Combine data with hardData
        labels = np.hstack([labels, hardLabels])

    # Determine an optimal value for C
    params = {"C": [.1, 1.0, 10.0, 100, 1000, 10000.0]}
    ### Note: I cannot use params here. I would first have to extract the features for the
    modeltemp = GridSearchCV(LinearSVC(random_state=42),params,cv=3)
    modeltemp.fit(data,labels)
    print("[INFO] best hyperparameters: {}".format(modeltemp.best_params_))

    # train the svd classifier
    print("[INFO] training classifier...")
    model = SVC(kernel="linear", C=conf["C"], probability=True, random_state=42)
    model.fit(data, labels)
    print(classification_report(labels,model.predict(data)))

    # dump the classifier to file
    print("[INFO] dumping classifier...")
    myFileName=conf["classifier_path"]
    f = open(conf["classifier_path"], "wb")
    f.write(cPickle.dumps(model))
    f.close()
    print("Finished.  SVD is saved as: {}".format(myFileName))

def test_model(hog,conf,image_Filename,SW):
    # load the classifier, then initialize the Histogram of Oriented Gradients descriptor
    # and the object detector
    # load the image and convert it to grayscale
    image = cv2.imread(image_Filename)
    #image = imutils.resize(image,width=int(conf["max_image_width"]))
    #image = imutils.resize(image,SW) # resizes to match size used during training
    #image = cv2.resize(image, width=min(260, image.shape[1]))
    minwidth=min(int(conf["max_image_width"]), image.shape[0])
    minheight=min(int(conf["max_image_width"]), image.shape[1])
    image = cv2.resize(image, (minwidth,minheight)) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print("Image size: {}".format(gray.shape))
    pfile=conf["classifier_path"]
    with open(pfile, 'rb') as f:
        model = cPickle.load(f, encoding='bytes')
    f.close()
  
    #hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]), 
    #      cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    #print(hog)
   
    od = ObjectDetector(model, hog)

    print(od)

    # detect objects in the image and apply non-maxima suppression to the bounding boxes
    print("Detecting the object")
    #winDim=conf["sliding_window_dim"]
    winDim=SW # Recall, the sliding window dimensions are computed.
    minSize=SW
    winStep=conf["window_step"]
    pyramidScale=conf["pyramid_scale"]
    minProb=conf["min_probability"]
    (boxes, probs) = od.StartDection_MultiProcess(gray,winDim, minSize,  winStep, pyramidScale, minProb)
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = image.copy()
    print("Finished detecting the object")  
    ##print("boxes: {}".format(boxes))

    if len(boxes) <1 :
        print("The object was not found")
    else:
        # loop over the original bounding boxes and draw them
        for (startX, startY, endX, endY) in boxes:
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # show the output images
    cv2.imshow("Original", orig)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


# In[ ]:

def Hard_Negative_Mining(conf,SW):
    data = []
    labels=[]
    # load the classifier, then initialize the Histogram of Oriented Gradients descriptor
    # and the object detector
    #model = cPickle.loads(open(conf["classifier_path"]).read())
    pfile=conf["classifier_path"]
    with open(pfile, 'rb') as f:
        model = cPickle.load(f, encoding='bytes')
    f.close()
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
    cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    od = ObjectDetector(model, hog)

    # grab the set of distraction paths and randomly sample them
    #dstPaths = list(paths.list_images(conf["image_distractions"]))
    dstPaths=glob.glob(conf["image_distractions"] + "\\*.jpg")
    number_of_distraction_images = conf["hn_num_distraction_images"]
    dstPaths = random.sample(dstPaths, number_of_distraction_images )

    # setup the progress bar
    #widgets = ["Mining: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    #pbar = progressbar.ProgressBar(maxval=len(dstPaths), widgets=widgets).start()

    # loop over the distraction paths
    for (i, imagePath) in enumerate(dstPaths):
        # load the image and convert it to grayscale
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("************ [INFO] Scanning image: {} of {}. Filename: {} ************ ".format(i,number_of_distraction_images,imagePath))
        # detect objects in the image
        #(boxes, probs) = od.detect(gray, conf["window_dim"], winStep=conf["hn_window_step"],
        #pyramidScale=conf["hn_pyramid_scale"], minProb=conf["hn_min_probability"])
        winDim=SW # Recall, the sliding window dimensions are computed.
        minSize=SW
        winStep=conf["window_step"]
        pyramidScale=conf["pyramid_scale"]
        minProb=conf["min_probability"]
        (boxes, probs) = od.StartDection_MultiProcess(gray,winDim, minSize,  winStep, pyramidScale, minProb)
        #(boxes, probs) = od.detect(gray, SW, winStep=conf["hn_window_step"],
        #    pyramidScale=conf["hn_pyramid_scale"], minProb=conf["hn_min_probability"])
        # loop over the bounding boxes
        print("[INFO] {} boxes were found with a probablility > {}".format(len(boxes),conf["hn_min_probability"]))
        for (prob, (startX, startY, endX, endY)) in zip(probs, boxes):
            # extract the ROI from the image, resize it to a known, canonical size, extract
            # HOG features from teh ROI, and finally update the data
            #roi = cv2.resize(gray[startY:endY, startX:endX], tuple(conf["window_dim"]),interpolation=cv2.INTER_AREA)
            roi = cv2.resize(gray[startY:endY, startX:endX],SW,interpolation=cv2.INTER_AREA)
            features = hog.describe(roi)
            ##data.append(np.hstack([[prob], features])) # This line gave errors
            data.append(features)
            labels.append(-1)
    return (data,labels)
    # update the progress bar
    #pbar.update(i)

    # sort the data points by confidence
    #pbar.finish()
    ###print("[INFO] sorting by probability...")
    ###data = np.array(data)
    ###data = data[data[:, 0].argsort()[::-1]]



# In[ ]:

def AppendDataToFile(conf,data,labels):
    # dump the dataset to file
    print("[INFO] dumping hard negatives to file...")
    #dump_dataset(data[:, 1:], [-1] * len(data), conf["features_path"], "hard_negatives",writeMethod="a")

    if len(data)>0 :
        dump_dataset(data, labels, conf["features_path"], "hard_negatives",writeMethod="a")
    else: 
        print("No Hard-Negatives were found in the images provided")


# # Begin HOG Feature Extraction for Positive images and Negative images

# ## Begin HOG feature extraction of Positive Images

# In[ ]:

def BeginHogFeatureExtraction(conf,SW):
    # initialize the HOG descriptor along with the list of data and labels
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]), 
          cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    print("[INFO] Begin HOG Feature Extraction of Positive Images...")
    print("Image XML File:")
    print(conf["image_dataset_XML"])
    print("Image files are located in:")
    print(conf["image_dataset"])
    # Open dataset images and extract features for different scales. The extension will
    # determine how the bounding box information is presented (.XML (as used by DLIB), or .Mat (as used by CalTech))
    tmp=os.path.splitext(conf["image_dataset_XML"]) #TODO  Need to come back to
    if tmp[1]==".xml":
        print("Using XML Format")
        # Run ExtractImageInfoFromXML_Hood.py
        # Note: The sliding window size is calculated from the images. The value in the json file is bypassed
        data,labels=import_from_XML(conf,hog,SW)
    else:
    # Run ExtractImageInfoFromMatlab_Hood.py
        print("Using Matlab Format")
        data,labels=import_with_Matlab(conf,hog,SW)
    lenPositiveFeatures=len(data)
    print("Finished")    
    print("There are {} feature vectors and each vector contains {} elements for a total of {} elements.".format(len(data),len(data[0]),len(data) * len(data[0])))
    print("[INFO] Begin HOG Feature Extraction of Negative Images...")
    #dstPaths = list(os.listdir(conf["image_distractions"]))
    dstPaths=glob.glob(conf["image_distractions"] + "\\*.jpg")
    patches=[]
    for i in np.arange(0, conf["num_distraction_images"]):
        # randomly select a distraction images, load it, convert it to grayscale, and
        # then extract random pathces from the image
        image = cv2.imread(random.choice(dstPaths))
        ##image = resize(image,width=int(conf["max_image_width"]))
        image = cv2.resize(image,SW)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        patches.append(image)
        # extract_patches_2d is a convienent ROI sampling implementation in scikit-learn
        ##patches = extract_patches_2d(image, tuple(conf["window_dim"]),max_patches=conf["num_distractions_per_image"])
        ##patches = extract_patches_2d(image, tuple(SW),max_patches=conf["num_distractions_per_image"])
    # loop over the patches,
    for patch in patches:
        # extract features from the patch, then update teh data and label list
        ##features = hog.describe(patch)
        features = hog.describe(patch)
        data.append(features)
        labels.append(-1)

    print("Finished")    
    print("There are now {} feature vectors and each vector contains {} elements for a total of {} elements.".format(len(data),len(data[0]),len(data) * len(data[0])))
    print("{} Positive features and {} Negative features".format(lenPositiveFeatures,len(patches)))
    # dump the dataset to file
    #pbar.finish()
    print("[INFO] dumping features and labels to file...")
    MyFeaturePath=conf["features_path"]
    print("Feature data is saved in {}".format(MyFeaturePath))
    dump_dataset(data, labels, MyFeaturePath, "features")
    return hog

# ## Begin HOG feature extraction of Negative Images

# ## Begin Training SVM using saved HOG feature vectors

# The 'C' parameter for the SVM measures how 'strict' SVM is. Larger values indicate a tolerance for fewer mistakes. While this can lead to higher accuracy on the training data, it could lead to overfitting. Smaller values lead to a 'soft-classifier'. Initially, we let it make many mistakes knowing downstream, hard negative mining will help rectify many of these mistakes.

# In[ ]:


def f(q,conf,SW,scale):
    print('parent process:', os.getppid())
    print('process id:', os.getpid())  
    (data,labels) = Hard_Negative_Mining(conf,SW,scale)
    print(data.shape)
    q.put([data,labels])


# In[ ]:

#from multiprocessing import Process,Queue
def run_multiple_processes_using_lists(f,conf,SW,numProcesses=4):
    #numProcesses =5 # Number of processes
    # Function f needs to contain a q.put([]) statement to return the output
    q=[]    # Queue List
    p=[]    # Process List
    d=[]    # Output List
    datalist=[]
    labellist=[]
    for i in range(0,numProcesses-1):
        q.append(Queue())
        p.append(Process(target=f,args=(q[i],conf,SW))) 

#print("*** Queue List ***")
#print(q)
#print("*** Process List ***")
#print(p)

    print("Start All Processes ...")
    for j in range(0,numProcesses-1):
        pp=p[j]
        pp.start()
        pp.join()

    print("Collecting Results")
    for k in range(0,numProcesses-1):
        d.append(q[k].get()) # Should return (data,labels)
        datalist = datalist + d[k][0] # Concatonate list
        labellist=labellist + d[k][1]
    
    return(datalist,labellist)
    #print("Verify Output")
    #print(d[0]) # Print the first item in the returned list
    #for l in range(0,numProcesses-1):
    #    print(type(d[l]))



# In[ ]:

if __name__ == '__main__':
    myJSONFile = os.getcwd() + "\\conf\\TrackBibs.json"
    conf = Conf(myJSONFile)
    SW=GetAvgDimensions(conf)
    hog=BeginHogFeatureExtraction(conf,SW)
    train_model(conf,0)
    (data,labels)=Hard_Negative_Mining(conf,SW) # Done using multiple processors for each pyramid layer
    #(datalist,labellist)=run_multiple_processes_using_lists(f,conf,SW,4)
    AppendDataToFile(conf,data,labels)
    train_model(conf,1)
    TestFile="M:\\DataSets\\SprintPhotos_Small\\dsc_3332.jpg"
    test_model(hog,conf,TestFile,SW)


# ## Testing Trained Model
# This step involves
#     1. Looping over all layers of the image pyramid.
#     2. Applying our sliding window at each layer of the pyramid.
#     3. Extracting HOG features from each window.
#     4. Passing the extracted HOG feature vectors to our model for classification.
#     5. Maintaining a list of bounding boxes that are reported to contain an object of interest with sufficient probability
