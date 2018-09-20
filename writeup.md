**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/sliding_windows.png
[image3]: ./examples/heatmap_example.png
[image4]: ./examples/output.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook. See get_hog_features in line 39.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and decided to go with `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` because the classifer trained with the HOG features using the above parameters works fine.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the 7th cell of the IPython notebook, I trained a SVM using GridSearchCV with parameters `{'kernel':('linear', 'rbf'), 'C':[1, 10]}`. GridSearchCV ended up choosing rbf as kernal.

I used RGB channels with histogram bin size 32 and spatially bined color with size `(32, 32)` as additional features.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I choosed scale 1, windows size 64 and overlapping 16 pixels (~67%) per window. The result seemed ok.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used RGB 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image2]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./video_output/vehicle_detection.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code can be found in the second cell of IPython notebook, starting from line 240. I generalized the positions of positive detections (region of interest) in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

### Here's an example of the labeled heatmap:
![alt text][image3]

### Here's an example of resulting bounding boxes are drawn onto the test image:
![alt text][image4]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the beginning I use HLS color space to extract all the features, including spatially binned color, color histograms and HOG. Turned out that the classifier trained with these features generate too many false positives/negatives. It looked fine in the sample images, but once applied to the video there're too many flaws.

The other issue I had was that the classifier generates false positives. I use heatmap and adjust the threshold value and it helped.

The pipeline might failed if the video is filmed at night or ranny day. To make it more robust I think more image preprocessing is necessary. For example use more than one color space features. Also the sampling rate of HOG, bin size of histogram and many other combinations of parameters should also be tested to fit different scenarios.

Another potential improvement is to smooth out the bounding boxes. Because we already know that cars in the video move in a steady speed and known direction, we can leverage this fact and predict where the next boudning box should be. Combining the prediction with the result obtained from the classifier we should be able to get a pretty smooth result.

