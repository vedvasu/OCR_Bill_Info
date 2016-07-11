# OCR_Bill_Info

Extracting the important information form an electronic Bill Image.

The project aims at using Optical Character Recognition to extract Bill ID, Name, Phone, Address, Bill Amount, form the bill.

PART 1: Creation of the training data (saved as text files in the save_data folder)
Step 1: Creating training data set for each letter (26 lowercase + 26 upper case + 9 numerals)
•	Choosing fonts frequently occurring and converting .ttl file to .jpeg

•	Extracting the text between the two lines (this has all the letters and numbers)  
-	This image is manually corrected to detach and letter if joined
-	Function regionOI_extractor(numberOfFonts) in file Extractor_dataset.py does this for all the .jpeg images in a folder

•	Every letter is cropped out form the image and saved in the folder data_set1 as-


-	The wrong detections if any must be deleted (I have kept best 10 cropped out images for each sample)
-	Function letterExtractor() in file Extractor_dataset.py does this for all the letters in the image and uses following: 
I) Letter_Extractor.py file for all the extracting letters in an order form left to right.  
II) save_data.py file to save cropped letter automatically after the previously existing samples with proper name.
-	This creates 62 folder in the data_set1 folder containing 10 samples of every letter.

•	Next, the above cropped letters are threshed inverted as files are saved as text files(.txt)
                                                            

-	Class createTextFile() in Training.py is used for this purpose
-	The resultant text files is saved in flat_text folder having 52 folders for each sample

•	Now the all the text files are combined as a single text file for creating a set of data for the trainer finally saved in saved_data folder.
-	Classification.txt: contains the letter with which the sample matches.
eg. matrix: [97. 97. 97. 97. 97. 97. 97. 97. 97. 97. 96. 96. 96. 96.……0. 0. 0. 0. 0. 0 .0.] resembling a a a a a a……A A A A A…… 0 0 0 0 ….
-	flattenedImages.txt: contains combined text files created from above procedure.
eg. matrix: [[0. 0. 0. 255…….0. 0. 255.]
                [255. 0. 0………………0.]
	     ……..
     [0. 0. 0. 0. 0. 0. 0. 0.]]
-	Matrix are float32 form with shape (620,) and (620,576) as all images are cropped with 24x24 size.
-	These files can be used directly and above process need not be repeated.
-	But the files are flexible enough to automate the process for implementing more efficient learning procedure in future.

PART 2: Letter extraction (General Process)
•	This is the most imp part of the implementation to detect the letters from any general image. As logos are highly variable in colors, text, background and other shapes, so this makes this process tough to generalize.
•	Implementation is still in process but work done till date is shown below.
•	File General_Letter_Extractor.py  has series of functions which perform this task.
•	Step 1: finding edges using canny edge detection algorithm.

•	Step 2: Threshing the image for properly detecting contours.

•	Step 3: blurring to remove small dots and erosion of the edged to get significant edges

-	A rectangle around the boundaries is intentionally made to detect the letters which are not fully closed contours or half cut by the edges (as in Facebook logo)

•	Step 4: finding contours and removing unwanted contours.

-	Unwanted contours are removed using the area of the contour, hierarchy methods and few image matrix reconstructions.
-	Still the unwanted contours between the letters or near the edges are to be removed.
-	The ‘ f ’ from Facebook is perfectly detected as it is simplest logo with no designing.

•	This process requires lots of further additions for generalization but works well for extremely simple images.

PART 3: Recognizing the letters detected by the General_Letter_Extractor.py
•	File ML_Testing.py has the classes having implementation of K-nearest and SVM machine learning methods for initial testing.
•	Run database_matching.py file to view the search results.
•	Step 1: Extract the letters bounding rectangle and pass into trainer to get the output.

-	These are resized into 24x24 as trainer requires the same shape as that of saved data in  flattenedImages.txt

•	Step 2: The result is stores in and array as [‘ f ’]
-	Results from every image in the database in again stored in different array as
 [‘ a ’,  ‘  p ’, ‘ p ’,  ‘ l ’,  ‘ e ’ ]

