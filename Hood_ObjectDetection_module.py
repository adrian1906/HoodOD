
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
#
#
#from imutils import paths
##import progressbar
#import cv2
#from pyimagesearch.object_detection import helpers
# import the necessary packages
# conda install -c anaconda simplejson
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
#conda install -c anaconda progressbar
#from dlib import progressbar

from __future__ import print_function
from skimage import feature, exposure
##from skimage import exposure
import numpy as np
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
import matplotlib.pyplot as plt
from multiprocessing import Process,Queue,Pool
from multiprocessing.pool import ThreadPool
import cv2
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import sklearn
from sklearn.svm import LinearSVC 
import _pickle as cPickle
import time
import dlib
sklearn_version =sklearn.__version__;
if sklearn_version =="0.17.1":
    ##from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import GridSearchCV #for Scikit-learn < 2.0
else: # assumes a later version
    from sklearn.model_selection import GridSearchCV # For Scikit-learn 2.0


# *** CLASSES ***
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
    # *** Code for parallel processing of each window for a given layer ***    
    def start_detection_mp(self,image,winDim, minSize,  winStep=4, pyramidScale=1.5, minProb=0.7):
        boxes=[]
        probs=[]
        (keepscale,keeplayer)=self.CalculateNumberOfScales(image,pyramidScale,minSize)
        print("There are {} scales in this image.".format(len(keepscale)))
        for i in range(0,len(keepscale)):
            print("Working on layer {0:4d}. Scale {1:.2f}".format(i,keepscale[i]))
            #(b,p)=self.detect_single_layer_mp(keeplayer[i],keepscale[i],winStep,winDim,minProb)

            (b,p)=self.detect_single_layer_mt(keeplayer[i],keepscale[i],winStep,winDim,minProb)
            boxes =boxes + b
            probs =probs + p

        return(boxes,probs)


    def detect_single_layer_mt(self,layer,scale,winStep,winDim,minProb):  # Use multiple threads
        # Uses multithreading instead of multiprocessing.
        # map() takes two parameters, a function and a list
        # it will iterate through the list, and produce another list of the results (p).
        # In order to pass in multiple arguments, a list of those arguments had to be created first
        # That is myArgs.
        q=[]
        p=[]
        boxes=[]
        probs=[]
        myArgs=[]
        #q=Queue(); # Not used
        if scale == []:
            xx, yy, windows= sliding_window_return(layer, 1, winDim)
        else:
            xx, yy, windows= sliding_window_return(layer, winStep, winDim)

        for i in range(0,len(xx)-1):

            myArgs.append([xx[i],yy[i],windows[i],layer,winStep,winDim,minProb,scale])
            # myArgs.append(yy)
            # myArgs.append(windows)
            # myArgs.append(layer)
            # myArgs.append(winStep)
            # myArgs.append(winDim)
            # myArgs.append(minProb)
            # myArgs.append(scale)
        p = Pool(2)
        pp=list(p.map(self.fff,myArgs))
        #print(q)
        #print(pp)
        #print(p)
        for ppp in pp:
                boxes = boxes + ppp[0]
                probs = probs + ppp[1]
        p.close()   # Needed to avoid OSError: [Errno 12] Cannot allocate memory
                    # Solution: https://stackoverflow.com/questions/26717120/python-cannot-allocate-memory-using-multiprocessing-pool
        p.join()        
        return(boxes,probs)


    def detect_single_layer_mp(self,layer,scale,winStep,winDim,minProb): 
        # Use multiple processors
        q=[]
        p=[]
        d=[]
        i=0
        boxes=[]
        probs=[]
        xx, yy, windows= sliding_window_return(layer, winStep, winDim)
        #print("Window Shape")
        #print(len(windows))
        #print("x: {}  y: {}".format(xx,yy))
        # process in chunks of 4 (for four processors)
        NumOfProcessors=7;
        print("There are {} windows for this layer. It will take {} processor loops to complete".format(len(windows),math.floor(len(windows)/NumOfProcessors)))
        NumberOfChuncks=math.floor(len(xx)/4)
        for aa in range(0,len(xx)-1,4):
            for ii in range(0,NumOfProcessors):
                ##print("aa: {}  ii: {}".format(aa,ii))
                x=xx[aa]
                y=yy[aa]
                window=windows[aa]
                #q.append(Queue())
                q=Queue() # Only need to create one Queue (FIFO buffer) to hold output from each process
                # when all processes are completed, the buffer will be emptied.
                #p.append(Process(target=self.ff,args=(q[ii],x,y,window,scale, minProb,winDim)))
                p.append(Process(target=self.ff,args=(q,x,y,window,scale, minProb,winDim)))
            for pp in p:
                pp.start()
                # Expectation is that the next loop will not return until all the joins are finsihed.
            for pp in p:
                pp.join(timeout=0)  
            
            ##print("Waiting for processes to finish")
            for pp in p:    
                while pp.is_alive():
                    pass
            
            for pp in p:
                #print("Process {} is finished!".format(pp))
                pass

                       
            #print("Joins are complete for processes {}".format(p))
            
            while not q.empty():
                d=q.get()
                boxes = boxes + d[0]
                probs = probs + d[1]

            #for qq in q:
            #    d=qq.get()
            #    boxes = boxes + d[0]
            #    probs = probs + d[1]

            p=[]  # Clear Processes    
            p=[]
            q=[]   
            #for k in range(0,len(p)):
                #print("Collecting results from process {}".format(k))
                #d.append(q[k].get()) # Should return (data,labels)
                #boxes = boxes + d[k][0] # Concatonate list
                #probs = probs + d[k][1]
            #print("Finsihed with block group aa {}".format(aa))
            #print("There are {} Boxes.".format(len(boxes)))
        return(boxes,probs)

    def fff(self,myArgs):
        
        boxes=[]
        probs=[]
        xx=myArgs[0]
        yy=myArgs[1]
        window=myArgs[2]
        layer=myArgs[3]
        winStep=myArgs[4]
        winDim=myArgs[5]
        minProb=myArgs[6]
        scale=myArgs[7]
        #print("MyArgs has length {}".format(len(myArgs)))
        #print("xx: {} yy {} winStep {} winDim {} minProb {}".format(xx,yy,winStep,winDim, minProb))
        b,p = self.ff(xx,yy,window,scale,minProb,winDim)
        #while not q.empty():
        #    print(q.get())

        # #d=q.get()
        if len(b)>0: # Check for empty
            print("b: {} p {}".format(b,p))
            boxes = boxes + b
            probs = probs + p
        # #print("Probabilities")    
        # print(prob)
        #q.put([boxes,probs])
        #q.close()
        #q.join_thread()
        return(boxes,probs)

    def ff(self,q,x,y,window,scale,minProb,winDim):
        print("Inside ff()")
        self.processID = os.getpid()
        boxes=[]
        probs=[]
        #print("*** [INFO] ProcessID: {0:7d} window shape: {1} ***".format(self.processID,window.shape))
        (winH, winW) = window.shape[:2]
        if winW == winDim[0] and winH ==winDim[1]: # Check that window dimension is correct ('I've noticed that some are off by a factor of 2.. TODO: Figure out.)
            #if winW >0 and winH >0:    
            features = self.desc.describe(window).reshape(1, -1)
            #print("Object Detector Feature Size: {}".format(features.shape))
            prob = self.model.predict_proba(features)[0][1]
            #if counter ==1 or counter % 5000 ==0:
                #print("[INFO] Model Probability: {}  Loop: {}   KeyPoint Top Left Corner (x,y) {}".format(prob, counter,[x,y]))
                #print("[INFO] ProcessID: {0:7d} Probability: {1:.3f}  Loop: {2:8d}".format(self.processID,prob,counter))
                # check to see if the classifier has found an object with sufficient
                # probability
            if prob > minProb:
                print("*** [INFO] ProcessID: {0:7d} Probability: {1:.3f}  Scale {2:.3f} ***".format(self.processID,prob,scale))
                ##print("[INFO] ********** Found a candidate! **************")
                # compute the (x, y)-coordinates of the bounding box using the current
                # scale of the image pyramid
                (startX, startY) = (int(scale * x), int(scale * y))
                endX = int(startX + (scale * winW))
                endY = int(startY + (scale * winH))

                # update the list of bounding boxes and probabilities
                boxes.append((startX, startY, endX, endY))
                probs.append(prob)
        q.put([boxes,probs])
        return(boxes,probs)

    def ff(self,x,y,window,scale,minProb,winDim):
        # Overload version without the Queue
        # for use with multithreading
        #print("Inside ff()")
        self.processID = os.getpid()
        boxes=[]
        probs=[]
        #print("*** [INFO] ProcessID: {0:7d} window shape: {1} ***".format(self.processID,window.shape))
        (winH, winW) = window.shape[:2]
        if winW == winDim[0] and winH ==winDim[1]: # Check that window dimension is correct ('I've noticed that some are off by a factor of 2.. TODO: Figure out.)
            #if winW >0 and winH >0:    
            features = self.desc.describe(window).reshape(1, -1)
            #print("Object Detector Feature Size: {}".format(features.shape))
            prob = self.model.predict_proba(features)[0][1]
            #if counter ==1 or counter % 5000 ==0:
                #print("[INFO] Model Probability: {}  Loop: {}   KeyPoint Top Left Corner (x,y) {}".format(prob, counter,[x,y]))
                #print("[INFO] ProcessID: {0:7d} Probability: {1:.3f}  Loop: {2:8d}".format(self.processID,prob,counter))
                # check to see if the classifier has found an object with sufficient
                # probability
            if prob > minProb:
                print("*** [INFO] ProcessID: {0:7d} Probability: {1:.3f}  Scale {2:.3f} ***".format(self.processID,prob,scale))
                ##print("[INFO] ********** Found a candidate! **************")
                # compute the (x, y)-coordinates of the bounding box using the current
                # scale of the image pyramid
                (startX, startY) = (int(scale * x), int(scale * y))
                endX = int(startX + (scale * winW))
                endY = int(startY + (scale * winH))

                # update the list of bounding boxes and probabilities
                boxes.append((startX, startY, endX, endY))
                probs.append(prob)
        #q.put([boxes,probs])
        return(boxes,probs)


    # *** Code for parallel processing of each layer ***
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
        self.processID = os.getpid()
        ##print('parent process:', os.getppid())
        print('process id:', self.processID )  
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
        print("[INFO] Investigating scale: {0:.2f}".format(scale))
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
                    #if prob > .9:
                        
                    if counter ==1 or counter % 5000 ==0:
                        #print("[INFO] Model Probability: {}  Loop: {}   KeyPoint Top Left Corner (x,y) {}".format(prob, counter,[x,y]))
                        print("[INFO] ProcessID: {0:7d} Probability: {1:.3f}  Loop: {2:8d}".format(self.processID,prob,counter))
                    # check to see if the classifier has found an object with sufficient
                    # probability
                    if prob > minProb:
                        print("*** [INFO] ProcessID: {0:7d} Probability: {1:.3f}  Loop: {2:8d} scale: {3:.2f} ***".format(self.processID,prob,counter,scale))
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

class HOG:
    def __init__(self, orientations=12, pixelsPerCell=(4, 4), cellsPerBlock=(2, 2), normalize=True):
        # store the number of orientations, pixels per cell, cells per block, and
        # whether normalization should be applied to the image
        self.orientations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.normalize = normalize
        print("HOG configured for Orientations: {}, PixelsPerCell: {}, CellsPerBlock {}, Normalize: {}".format(orientations,
            pixelsPerCell,cellsPerBlock,normalize))
        # Pre-calculate feature vector size

    def describe(self, image):
        # compute Histogram of Oriented Gradients features
        # To return the hogImage, Visualize needs to be set tot True. 
        # Use describe_and_return_HOGImage() instead
        #hist,hogImage = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixelsPerCell,
            #cells_per_block=self.cellsPerBlock, transform_sqrt=self.normalize, visualize=True, block_norm ='L2')
        hist = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixelsPerCell,
            cells_per_block=self.cellsPerBlock, transform_sqrt=self.normalize, visualize=False, block_norm ='L2')
        # Note: Feature size will be number of orientations * number of cells in a block * 
        # number of scans horizontal * number of scans vertical  (Assumes a stride of 1)
        #print("hist size")
        #print(hist.shape)
        hist[hist < 0] = 0
        #imsize=image.shape
        #print(imsize)
        #numberofscansheight=math.floor(imsize[0]/self.pixelsPerCell[0])-1
        #print("Number of Vertical Scans: ")
        #print(numberofscansheight)

        #numberofscanswidth=math.floor(imsize[1]/self.pixelsPerCell[1])-1
        #print("Number of Horizontal Scans: ")
        #print(numberofscanswidth)

        #numFeaturesPerBlock = self.orientations *self.cellsPerBlock[0]*self.cellsPerBlock[1]
        #print("Number of features per block")
        #print(numFeaturesPerBlock)

        #featureLength = numberofscansheight*numberofscanswidth*numFeaturesPerBlock
        #print("Number of features")
        #print(featureLength)

        

        #self.plot_hog(image,hogImage)
        # return the histogram
        return hist

    def describe_and_return_HOGImage(self, image):
        # compute Histogram of Oriented Gradients features
        (hist,hogImage) = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixelsPerCell,
            cells_per_block=self.cellsPerBlock, transform_sqrt=self.normalize, visualize=True)
        hist[hist < 0] = 0
        self.plot_hog(image,hogImage)
        # return the histogram
        return hist,hogImage
    
    def plot_hog(self,image,hog_image):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()   

class Conf:
    def __init__(self, confPath):
        # load and store the configuration and update the object's dictionary
        conf = json.loads(open(confPath).read())
        self.__dict__.update(conf)

    def __getitem__(self, k):
        # return the value associated with the supplied key
        return self.__dict__.get(k, None)

class Get_BodyboxHood: # will need to run this prior to calling class detector = dlib.get_frontal_face_detector()
    def __init__(self, image):
        #self.filename_path = filename_path
        self.detector =  dlib.get_frontal_face_detector()
        #self.image = dlib.load_rgb_image(self.filename_path)
        self.image = image
    
    def plot_DLIB_results(self,dets,outfolder):

        orig=self.image.copy()
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            x1new,y1new,x2new,y2new = self.new_rectangle(orig,d)
            cv2.rectangle(orig,(x1new,y1new),(x2new,y2new),(0,255,0),2)    
            #cv2.rectangle(orig,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,0),2)

        #FN=os.path.split(self.filename_path)
        #resultsimagepath=outfolder + FN[1]
        #print("Saving: {}".format(resultsimagepath))
        #cv2.imwrite(resultsimagepath, orig);

        cv2.imshow("myImage",orig)         
        cv2.waitKey(1000)


    def new_rectangle(self,dets):

        x1=dets.left()
        y1=dets.top()
        x2=dets.right()
        y2=dets.bottom()
        print("x1 {}  y1 {} x2 {}  y2 {}".format(x1,y1,x2,y2))
        face_width=x2-x1
        face_height=y2-y1

        scale=8; # Keep even to ensure extraction is square
        # assume torso is 3*face
        x1new=x1-int(scale/2*face_width)
        x2new=x2+int(scale/2*face_width)
        y1new=y1
        y2new=y1+scale*face_height  # 4 is tight and get most. 5 will be the insurance box size
        print("face_width {}  face_height {}".format(face_width,face_height))
        print("x1new {}  y1new {} x2new {}  y2new {}".format(x1new,y1new,x2new,y2new))
        return(x1new,y1new,x2new,y2new)


    def find_boxes(self):
        x1=[]
        y1=[]
        x2=[]
        y2=[]

        #print("Processing image {}".format(self.filename_path))
        #img = dlib.load_rgb_image(self.filename_path)
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        dets = self.detector(self.image ,1)
        #print("Number of faces detected: {}".format(len(dets)))
        for i, d in enumerate(dets):
            x1new,y1new,x2new,y2new = self.new_rectangle(d)
            x1.append(x1new)
            y1.append(y1new)
            x2.append(x2new)
            y2.append(y2new)

        return(x1,y1,x2,y2)









# *** Stand Alone METHODS***
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

def sliding_window_return(image, stepSize, windowSize):
    # slide a window across the image
    print("Image Size {}, StepSize {}  WindowSize {}".format(image.shape, stepSize, windowSize))
    #cv2.imshow("Image", image)
    #cv2.waitKey(0) 
    
    xx=[]
    yy=[]
    windows=[]
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            xx.append(x)
            yy.append(y)
            windows.append(image[y:y + windowSize[1], x:x + windowSize[0]])
    return(xx, yy, windows)
    
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
            imagename=conf["image_dataset"] + info.get('file')
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
            ##print("The image size is {}.".format(roi.shape))
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
                features = hog.describe(roi)
                data.append(features)
                labels.append(1)
        
                # update the progress bar
                #pbar.update(i)
        except:
            print("Issue with file:  {}".format(imagename))
            
        i=i+1
        
    return  data,labels

def import_from_folder(folderpath,conf,hog,SW,label):
    data = []   
    labels = []
    dstPaths=glob.glob(folderpath + "*.jpg")
    patches=[]
    #for i in np.arange(0, len(dstPaths)):
    for myfile in dstPaths:
        # randomly select a distraction images, load it, convert it to grayscale, and
        # then extract random pathces from the image
        #image = cv2.imread(random.choice(dstPaths))
        image = cv2.imread(myfile)
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
        labels.append(label)
        
    return  data,labels    

def GetAvgDimensions(conf):
    # Returns W,H in that order.
    doc = etree.parse(conf["image_dataset_XML"])
    MyXML=doc.find("images")
    widths = []
    heights = []
    for info in MyXML:
        ##imagename=conf["image_dataset"] + "\\" + info.get('file')
        widths.append(int(info[0].get('width')))
        heights.append(int(info[0].get('height')) )

    (avgWidth, avgHeight) = (np.mean(widths), np.mean(heights))
    (stdW,stdH)=(np.std(widths),np.std(heights))
    #print("The length of widths is {}".format(len(widths)))
    pixelsPerCell=conf["pixels_per_cell"]
    PPCx = pixelsPerCell[0]
    PPCy = pixelsPerCell[1]
    newW=math.ceil(int(avgWidth/2)/PPCx)*PPCx
    newH=math.ceil(int(avgHeight/2)/PPCy)*PPCy
    print("Value prior to scaling: newW {}  newH {}".format(newW,newH))
    AR=newW / newH
    mhws=conf["max_hog_window_size"]
    if newW>newH: # Width will govern overall size
        if newW>mhws:
            newW=mhws
            newH=round((newW/AR)/PPCy)*PPCy
    else:
        if newH > mhws:
            newH=mhws
            newW=round(AR*newH/PPCx)*PPCx

        

    print("[INFO] avg. width: {:.2f} +/- {:.2f}".format(avgWidth,stdW))
    print("[INFO] avg. height: {:.2f} +/- {:.2f}".format(avgHeight,stdH))
    print("[INFO] aspect ratio: {:.2f}".format(avgWidth / avgHeight))
    print("[INFO] The recommended Sliding Window Size is W:{}  H:{}".format(newW,newH))
    print("[INFO] Sliding Window Aspect Ratio {:.2f}".format(newW / newH))
    return tuple([newW,newH])

def train_model(conf,useHardNegatives):
    # load the configuration file and the initial dataset
    print("[INFO] loading dataset...")
    #conf = Conf(args["conf"])
    (data, labels) = load_dataset(conf["features_path"], "features") # contains images labled as good (+1) and bad (-1)
    #print("Example of a data point. Feature: {} and Label {}".format(data[0],labels[0]))
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
    start=time.time()
    image = cv2.imread(image_Filename)
    #image = imutils.resize(image,width=int(conf["max_image_width"]))
    #image = imutils.resize(image,SW) # resizes to match size used during training
    #image = cv2.resize(image, width=min(260, image.shape[1]))
    
    h,w,d=image.shape
    print("Testing image: {}  Original shape: {}".format(image_Filename,(h,w)))
    # Note span reflect # of SW that can fit into the window
    # TODO: add to hyperparameter JSON File
    pixelsPerCell=tuple(conf["pixels_per_cell"])
    #span =5
    span = int(conf["span"])
    (objectsize_H,objectsize_W) = calculate_optimal_image_size(pixelsPerCell,SW,span)
    print("Sliding Window Width: {}  Height: {}".format(SW[0],SW[1]))
    if SW[0]>SW[1]: # Wide
        (newHeight,newWidth) =rescaleImage(image, objectsize_W,"L")
    else:
        (newHeight,newWidth) =rescaleImage(image, objectsize_H,"P")
    #if w> h : #Landscape mode
    #    print("Skpping")

    #        print("[INFO] The image is in landscape mode")
    #        minWidth=min(int(conf["max_image_width"]), w)  # 1 ==> Columns
    #        newHeight=math.floor(h/w*minWidth)
    #        image = cv2.resize(image, (minWidth,newHeight))
    #else: # Portriat Mode
    #    print("[INFO] The image is in portrait mode")
    #    minHeight=min(int(conf["max_image_width"]), h) # 0 ==> Rows
    #    newWidth =math.floor(w/h*minHeight)
    #print("Shape before resize")
    #print(image.shape)
    image = cv2.resize(image, (newWidth,newHeight))
    #print("Shape after resize:")
    #print(image.shape)
    print("Shape before: {}  Shape after {}".format((h,w),image.shape[0:2]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pfile=conf["classifier_path"]

    # LOAD THE MODEL
    with open(pfile, 'rb') as f:
        model = cPickle.load(f, encoding='bytes')
    f.close()
  
    #hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]), 
    #      cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    #print(hog)
   
    od = ObjectDetector(model, hog)

    #print(od)

    # detect objects in the image and apply non-maxima suppression to the bounding boxes
    print("Detecting the object")
    #winDim=conf["sliding_window_dim"]
    winDim=SW # Recall, the sliding window dimensions are computed. (w,h Opposite shape command)
    minSize=SW
    # Need to verify that the window_step is a factor of pixelsPerCell (in both directions)
    winStep=conf["window_step"]
    pyramidScale=conf["pyramid_scale"]
    minProb=conf["min_probability"]
    #PPC=conf["pixels_per_cell"]
    #winStep=PPC[0]
    #maxImageWidth=math.floor(SW[0]/PPC[0])*PPC[0]*4 # scale the image to be 4 time integer multiple of scanning window width

    #(boxes, probs) = od.StartDection_MultiProcess(gray,winDim, minSize,  winStep, pyramidScale, minProb)
    

    boxed_torso_image = Get_Bodybox(gray)
    x1new,y1new,x2new,y2new = boxed_torso_image.find_boxes()
    
    #for iii in range(0,len(x1new)-1):
    # For now, only use one of the found boxes
    gray = gray[(x1new[0],y1new[0]),(x2new[0],y2new[0])]

    (boxes, probs) = od.start_detection_mp(gray,winDim, minSize,  winStep, pyramidScale, minProb)
    pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
    orig = image.copy()
    print("Finished detecting the object")  
    ##print("boxes: {}".format(boxes))

    if len(boxes) <1 :
        print("The object was not found")
    else:
        # loop over the original bounding boxes and draw them
        for (startX, startY, endX, endY) in boxes:
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 10)

    # loop over the allowed bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 10)

    # show the output images
    ##plt.subplot(121),plt.imshow(orig,'gray'),plt.title('Original')
    ##plt.subplot(122),plt.imshow(image,'gray'),plt.title('Image')
    ##plt.show()
    #cv2.imshow("Original", orig)
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    end = time.time()
    duration = end-start
    
    imageresults=conf["image_results_folder"]
    FN=os.path.split(image_Filename)
    print("Time taken to process {0}:  {1:.2f} s".format(FN[1],duration))
    resultsimagepath=imageresults + FN[1]
    print("Saving: {}".format(resultsimagepath))
    cv2.imwrite(resultsimagepath, image );
    #pause(5) #Pause 5 seconds before the next image.


def test_model_using_DLIB_FaceDetection(hog,conf,image_Filename,SW):
    # load the classifier, then initialize the Histogram of Oriented Gradients descriptor
    # and the object detector
    # load the image and convert it to grayscale
    start=time.time()
    image = cv2.imread(image_Filename)
    #image = imutils.resize(image,width=int(conf["max_image_width"]))
    #image = imutils.resize(image,SW) # resizes to match size used during training
    #image = cv2.resize(image, width=min(260, image.shape[1]))
    
    h,w,d=image.shape
    print("Testing image: {}  Original shape: {}".format(image_Filename,(h,w)))
    # Note span reflect # of SW that can fit into the window
    # TODO: add to hyperparameter JSON File
    pixelsPerCell=tuple(conf["pixels_per_cell"])
    #span =5
    span = int(conf["span"])
    (objectsize_H,objectsize_W) = calculate_optimal_image_size(pixelsPerCell,SW,span)
    print("Sliding Window Width: {}  Height: {}".format(SW[0],SW[1]))
    if SW[0]>SW[1]: # Wide
        (newHeight,newWidth) =rescaleImage(image, objectsize_W,"L")
    else:
        (newHeight,newWidth) =rescaleImage(image, objectsize_H,"P")
    #if w> h : #Landscape mode
    #    print("Skpping")

    #        print("[INFO] The image is in landscape mode")
    #        minWidth=min(int(conf["max_image_width"]), w)  # 1 ==> Columns
    #        newHeight=math.floor(h/w*minWidth)
    #        image = cv2.resize(image, (minWidth,newHeight))
    #else: # Portriat Mode
    #    print("[INFO] The image is in portrait mode")
    #    minHeight=min(int(conf["max_image_width"]), h) # 0 ==> Rows
    #    newWidth =math.floor(w/h*minHeight)
    #print("Shape before resize")
    #print(image.shape)
    image = cv2.resize(image, (newWidth,newHeight))
    #print("Shape after resize:")
    #print(image.shape)
    print("Shape before: {}  Shape after {}".format((h,w),image.shape[0:2]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pfile=conf["classifier_path"]

    # LOAD THE MODEL
    with open(pfile, 'rb') as f:
        model = cPickle.load(f, encoding='bytes')
    f.close()
  
    #hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]), 
    #      cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    #print(hog)
   
    od = ObjectDetector(model, hog)

    #print(od)

    # detect objects in the image and apply non-maxima suppression to the bounding boxes
    print("Detecting the object")
    #winDim=conf["sliding_window_dim"]
    winDim=SW # Recall, the sliding window dimensions are computed. (w,h Opposite shape command)
    minSize=SW
    # Need to verify that the window_step is a factor of pixelsPerCell (in both directions)
    winStep=conf["window_step"]
    pyramidScale=conf["pyramid_scale"]
    minProb=conf["min_probability"]
    #PPC=conf["pixels_per_cell"]
    #winStep=PPC[0]
    #maxImageWidth=math.floor(SW[0]/PPC[0])*PPC[0]*4 # scale the image to be 4 time integer multiple of scanning window width

    #(boxes, probs) = od.StartDection_MultiProcess(gray,winDim, minSize,  winStep, pyramidScale, minProb)
    

    boxed_torso_image = Get_BodyboxHood(gray)
    x1new,y1new,x2new,y2new = boxed_torso_image.find_boxes()
    print("Length of x1new {}".format(x1new))
    for iii in range(0,len(x1new)):
        # For now, only use one of the found boxes
        ww=x2new[iii]-x1new[iii]
        hh=y2new[iii]-y1new[iii]
        print("ww : {}  hh: {}".format(ww,hh))
        if ww >= SW[0] and hh >int(1.5*SW[1]):
            gray = gray[max(0,x1new[iii]):min(x2new[iii],w), max(0,y1new[iii]):min(y2new[iii],h)]

            #(boxes, probs) = od.start_detection_mp(gray,winDim, minSize,  winStep, pyramidScale, minProb)
            (boxes, probs) = od.detect_single_layer_mt(gray,1,winStep,winDim,minProb)
            pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
            orig = gray.copy()
            print("Finished detecting the object")  
            ##print("boxes: {}".format(boxes))

            if len(boxes) <1 :
                print("The object was not found")
            else:
                # loop over the original bounding boxes and draw them
                for (startX, startY, endX, endY) in boxes:
                    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 10)

                # loop over the allowed bounding boxes and draw them
                for (startX, startY, endX, endY) in pick:
                    cv2.rectangle(gray, (startX, startY), (endX, endY), (0, 255, 0), 10)

            # show the output images
            ##plt.subplot(121),plt.imshow(orig,'gray'),plt.title('Original')
            ##plt.subplot(122),plt.imshow(image,'gray'),plt.title('Image')
            ##plt.show()
            #cv2.imshow("Original", orig)
            #cv2.imshow("Image", image)
            #cv2.waitKey(0)
            end = time.time()
            duration = end-start
            
            imageresults=conf["image_results_folder"]
            FN=os.path.split(image_Filename)
            print("Time taken to process {0}:  {1:.2f} s".format(FN[1],duration))
            resultsimagepath=imageresults + FN[1]
            print("Saving: {}".format(resultsimagepath))
            cv2.imwrite(resultsimagepath, gray);
            #pause(5) #Pause 5 seconds before the next image.




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
    dstPaths=glob.glob(conf["image_distractions"] + "*.jpg")
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

def AppendDataToFile(conf,data,labels):
    # dump the dataset to file
    print("[INFO] dumping hard negatives to file...")
    #dump_dataset(data[:, 1:], [-1] * len(data), conf["features_path"], "hard_negatives",writeMethod="a")

    if len(data)>0 :
        dump_dataset(data, labels, conf["features_path"], "hard_negatives",writeMethod="a")
    else: 
        print("No Hard-Negatives were found in the images provided")

def initialize_hog(conf):
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]), 
          cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    return hog

def BeginHogFeatureExtraction(hog,conf,SW,MyFlag=0):
    # initialize the HOG descriptor along with the list of data and labels
    #hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]), 
    #      cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
    print("[INFO] Begin HOG Feature Extraction of Positive Images...")
    print("[INFO] Image XML File is located in: ")
    print(conf["image_dataset_XML"])
    print("[INFO] Image files are located in: ")
    print(conf["image_dataset"])
    # Open dataset images and extract features for different scales. The extension will
    # determine how the bounding box information is presented (.XML (as used by DLIB), or .Mat (as used by CalTech))
    if MyFlag == 1:
        print("Importing from foler of Cropped Files (Not using .xml or .mat file")
        #folderpath=conf["cropped_image_dataset"]
        label = 1
        data,labels = import_from_folder(conf["cropped_image_dataset"],conf,hog,SW,label)
        print("There are {} images analyzed".format(len(data)))
    else:    
        
        tmp=os.path.splitext(conf["image_dataset_XML"]) #TODO  Need to come back to
        if tmp[1]==".xml":
            print("Using XML Format")
            # Run ExtractImageInfoFromXML_Hood.py
            # Note: The sliding window size is calculated from the images. The value in the json file is bypassed
            data,labels=import_from_XML(conf,hog,SW)
        elif tmp[1]==".mat":
            # Run ExtractImageInfoFromMatlab_Hood.py
            print("Using Matlab Format")
            data,labels=import_with_Matlab(conf,hog,SW)

        
    lenPositiveFeatures=len(data)
    print("Finished")    
    print("There are {} feature vectors and each vector contains {} elements for a total of {} elements.".format(len(data),len(data[0]),len(data) * len(data[0])))
    print("[INFO] Begin HOG Feature Extraction of Negative Images...")
    #dstPaths = list(os.listdir(conf["image_distractions"]))
    dstPaths=glob.glob(conf["image_distractions"] + "*.jpg")
    print("[INFO] There are {} distraction images".format(len(dstPaths)))
    patches=[]
    #random_start = random.randrange(0,len(dstPaths))
    for i in np.arange(0, conf["num_distraction_images"]):
        # randomly select a distraction images, load it, convert it to grayscale, and
        # then extract random pathces from the image

        #image = cv2.imread(random.choice(dstPaths))
        #image = cv2.imread(dstPaths[random_start+i]) # Consecutive images

        image = cv2.imread(dstPaths[i]) # Consecutive images
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

def f(q,conf,SW,scale):
    print('parent process:', os.getppid())
    print('process id:', os.getpid())  
    (data,labels) = Hard_Negative_Mining(conf,SW,scale)
    print(data.shape)
    q.put([data,labels])

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

def pause(timeInSeconds):
    duration=0
    start=time.time()
    print("Waiting")
    while duration <timeInSeconds:
        end=time.time()
        duration=end-start
        
    print("Finished waiting {} s.".format(duration))
    
  

def calculate_optimal_image_size(pixelsPerCell,SW,span):
    # Span is an integer representing how many SW boxes that can span across the image
    # For example, saw SW is 128x80, then a span of 5 would lock the width to 128 *5 = 640
    # The height would also scale to 400, so the image dimension would be rescaled to 640x400.
    PPCx = pixelsPerCell[0]
    PPCy = pixelsPerCell[1]
    objectsize_W = round(SW[0]/PPCx)*PPCx*span
    objectsize_H = round(SW[1]/PPCy)*PPCy*span
    print("objectsize_H: {}  objectsize_W: {}".format(objectsize_H,objectsize_W))
    return(objectsize_H,objectsize_W)

def rescaleImage(image, max_dimension, direction):
    print("Ready to Scale")

    h,w=image.shape[0:2]
    print("Original h: {}  w: {} ".format(h,w))
    if direction == "P":
        print("Portrait Mode")
        newHeight = max_dimension 
        newWidth  = round (w/h*newHeight)
    else:
        print("Landscape Mode")
        newWidth = max_dimension
        newHeight = round(h/w * newWidth)
    print("newHeight: {}  newWidth: {}".format(newHeight,newWidth))    
    return (newHeight,newWidth)


def plot_sliding_window(image,stepSize,WindowSize,x,y,delayms):
    h,w = image.shape
    print("w: {}  h: {}  maxx {}  maxy {}".format(w,h,w-WindowSize[0],h-WindowSize[1]))
    for i in range(0,len(x)):
        if x[i]< w-WindowSize[0] and y[i]<h-WindowSize[1]:
            orig = image.copy()
            startX=x[i]
            startY=y[i]
            endX=startX + WindowSize[0]
            endY=startY + WindowSize[1]
            #print("Image shape {}".format(image.shape))
            #print("Box coordinates {}".format((startX, startY, endX, endY)))
            cv2.rectangle(orig,(startX, startY), (endX, endY),(0,0,255),3)
            cv2.imshow("Orig", orig)
            cv2.waitKey(delayms) 