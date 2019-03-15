      
from Hood_ObjectDetection_module import *
import argparse

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-cf", "--conf", required=True, help="path to the configuration file")
ap.add_argument("-s", "--start_index", required=True, help="index of 1st file to analyze")
ap.add_argument("-e", "--end_index", required=True, help="index of last file to analyze")
ap.add_argument("-at", "--atrained", required=True, help="index of last file to analyze")
args = vars(ap.parse_args())
# load the configuration file and initialize the list of widths and heights
conf = Conf(args["conf"])
start = int(args["start_index"])
stop = int(args["end_index"])
# In[ ]:

if __name__ == '__main__':
    #myJSONFile = os.getcwd() + "\\conf\\TrackBibs.json"
    #alreadytrained=0;
    alreadytrained=int(args["atrained"])
    ##myJSONFile = os.getcwd() + "\\conf\\TrackBibs2019b.json"
    #myJSONFile = os.getcwd() + "/conf/TrackBibs2019_ARL.json"
    ##conf = Conf(myJSONFile)
    SW=GetAvgDimensions(conf)
    hog = initialize_hog(conf)
    if alreadytrained == 0: 
        BeginHogFeatureExtraction(hog,conf,SW,1)
   
        train_model(conf,0)
        #(data,labels)=Hard_Negative_Mining(conf,SW) # Done using multiple processors for each pyramid layer
        #(datalist,labellist)=run_multiple_processes_using_lists(f,conf,SW,4)
        #AppendDataToFile(conf,data,labels)
        #train_model(conf,1)

    #TestFiles=glob.glob("M:\\DataSets\\2016_USATF_Sprint_TrainingDataset" + "\\*.jpg")
    #TestFiles=glob.glob("M:\\DataSets\\SprintPhotos_Small" + "\\*.jpg")
    testfilefolder=conf["test_files"]
    TestFiles=glob.glob(testfilefolder + "*.jpg")
    #TestFiles=glob.glob("M:\\DataSets\\2016_USATF_Sprint_TrainingDataset\\Extracted_Objects\\Bib" + "\\*.jpg")
    print("There are {} Testfiles".format(len(TestFiles)))
    ##numtestimages=50
    ##random_start = random.randrange(0,len(TestFiles)-numtestimages)
    print("Begin Testing Images ")
    for i in range(start,stop):
        print("*** Testing Image {} ****".format(i))
        #test_model(hog,conf,TestFiles[random_start+i],SW)
        test_model(hog,conf,TestFiles[i],SW)


# ## Testing Trained Model
# This step involves
#     1. Looping over all layers of the image pyramid.
#     2. Applying our sliding window at each layer of the pyramid.
#     3. Extracting HOG features from each window.
#     4. Passing the extracted HOG feature vectors to our model for classification.
#     5. Maintaining a list of bounding boxes that are reported to contain an object of interest with sufficient probability
