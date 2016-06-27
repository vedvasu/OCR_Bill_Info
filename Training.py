import cv2
import numpy as np

############################ Global Parameters ###############################
MIN_CONTOUR_AREA = 0
RESIZED_IMAGE_WIDTH = 24
RESIZED_IMAGE_HEIGHT =24
##############################################################################

class createTextFile():
    
    def __init__(self,source_path,destination_path):
        
        self.npaFlattenedImages=np.empty((0,576))              # np array of size 24x24
        self.source = source_path
        self.destination = destination_path
    
    def preProcessImage(self,path):

        '''
        * This function takes in the values input data element and processes it to create binany images
        * Image are saved as required by trainer (K- nearest for intial testing)
        * The np array in used as trainer requires flattened np-array
        '''

        self.npaFlattenedImages=np.empty((0,576))
        self.image_file=cv2.imread(path)
        
        self.imgGray = cv2.cvtColor(self.image_file, cv2.COLOR_BGR2GRAY)
        self.imgThresh = cv2.adaptiveThreshold(self.imgGray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

        npaFlattenedImage = self.imgThresh.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        self.npaFlattenedImages = np.append(self.npaFlattenedImages, npaFlattenedImage, 0)
                
        return self.npaFlattenedImages

    
    def setup(self,number_of_files):

        '''
        * Creating flattened text files for all the samples
        * setup files for envoking operations is the class
        '''
        
        for sample in xrange(1,63):                       # total training for 62 samples (26 lowercase, 26 uppercase, 10 digits)
            for i in range(1,number_of_files+1):                # 10 font samples for each case

                #text =obj=preProcessing('dataset_test1/sample (1)/'+str(i)+'.jpg')
                
                text = self.preProcessImage(self.source + 'sample ('+str(sample)+')/'+str(i)+'.jpg')
                
                np.savetxt(self.destination + 'sample ('+str(sample)+')/'+str(i)+'.txt', text)

            print 'Flattened text file created for all samples in folder '+str(sample)+'...'
        print 'All flattened text files stored'


class createTrainingFiles():

    def __init__(self,source_path,destination_path):

        self.npaClassifications = []                            # represents User_defined training objects
        self.npaFlattenedImages = np.empty((0,576))             # represents training sets for each sample
        self.source = source_path
        self.destination = destination_path   
    
    
    def trainingClassification(self,number_of_files):

        '''
        * Creates user classfied sets for the samples to be trained.
        * Uses ASCII value for alphabets
        '''

        for i in range(0,26):
            for j in range(1,number_of_files+1):
                self.npaClassifications.append(97+i)          # ASCII value can be checked through ord('A')

        for i in range(0,26):
            for j in range(1,number_of_files+1):
                self.npaClassifications.append(65+i)
                
        for i in range(0,10):
            for j in range(1,number_of_files+1):
                self.npaClassifications.append(i)
        
        self.npaClassifications = np.array(self.npaClassifications,np.float32)      # K- nearest trainer requires np array
        #self.npaClassifications = self.npaClassifications.reshape((self.npaClassifications.size, 1))

        print len(self.npaClassifications),self.npaClassifications.shape,self.npaClassifications
        return self.npaClassifications

    def trainingFlattened(self,number_of_files):

        '''
        * Creates corresponding array files for classified files.
        * The trainer searcher/matches the test_case with these arrays
        ''' 

        for i in range(1,63):
               for j in range(1,number_of_files+1):
                   npaFlattenedImage= np.loadtxt(self.source +'sample ('+str(i)+')/'+str(j)+'.txt',np.float32)
                   self.npaFlattenedImages = np.append(self.npaFlattenedImages, [npaFlattenedImage],0)       
        
        self.npaFlattenedImages = np.array(self.npaFlattenedImages,np.float32)
        
        print len(self.npaFlattenedImages),self.npaFlattenedImages.shape,self.npaFlattenedImages
        return self.npaFlattenedImages

    
    def setup(self,number_of_files):

        npaClassification = self.trainingClassification(number_of_files)
        np.savetxt(self.destination + "classifications.txt", npaClassification) 
        print 'Classification file saved'

        npaFlattenedImage =self.trainingFlattened(number_of_files)
        np.savetxt(self.destination + "flattenedImages.txt", npaFlattenedImage) 
        print 'Flattened text files saved'

# # Part 1 : Creating seperate flattened text files for all the samples
# s = createTextFile('dataset_test1/','flat_text/')
# s.setup(10)

# Part 2 : Creating final files for the trainer
s = createTrainingFiles('flat_text/','saved_data/')
s.setup(10)

cv2.waitKey(0)
cv2.destroyAllWindows()
        
