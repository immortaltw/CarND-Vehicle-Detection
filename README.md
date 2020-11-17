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
[image2]: ./output_images/sliding_windows.png
[image3]: ./output_images/sliding_windows_ycrcb.png
[image4]: ./output_images/heatmap_ycrcb.png

## Project Discussions

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook. See get_hog_features in line 39.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and decided to go with `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` because the classifier trained with the HOG features using the above parameters works fine.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the 7th cell of the IPython notebook, I trained a SVM using GridSearchCV with parameters `{'kernel':('linear', 'rbf'), 'C':[1, 10]}`. GridSearchCV ended up choosing rbf as kernal.

I used RGB channels with histogram bin size 32 and spatially binned color with size `(32, 32)` as additional features.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

At first trial I chose scale 1, windows size 64 and overlapping 16 pixels (75%) per window. The result seemed ok.

At the second trial I chose scale 1.5 because in the result of the first trial there're some cases which the windows are too small to cover the vehicle. I also chose the windows size as 64 with overlapping 8 pixels per window. The resulting window size was good enough to cover the vehicle entirely.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

At the first trial, I used RGB 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here's an example image:

![alt text][image2]

In the second trial, I used YCrCb channels instead of RGB in all feature extraction, which gives an even better result which has fewer false positives. Here's an example image of the classification result:

![alt text][image3]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./video_output/vehicle_detection.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code can be found in the second cell of IPython notebook, starting from line 240. I generalized the positions of positive detections (region of interest) in each frame of the video. Based on observation, the region of interest in the video in Y direction which covers the lanes is roughly from 400 to 680 (ystart and ystop).

I used a naive filter which takes the average of heatmap of previous 8 frames and the heatmap of the current frame, applies 0.5 as the weight on each and sums them up to get a final heatmap. The implementation is at line 15 of the 15th cell in the IPython notebook. The code snippet looks like:

```python
    # Apply filter
    res_heat = np.zeros_like(image[:,:,0]).astype(np.float)
    for h in history:
        res_heat = np.add(res_heat, h)

    res_heat = 0.5 * res_heat/len(history) + 0.5 * heat
```

I then thresholded that heatmap to identify vehicle positions with a threshold value 0.1 based on observation.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

### Here's an example of the resulting image and the labeled heatmap:
![alt text][image4]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the beginning I use HLS color space to extract all the features, including spatially binned color, color histograms and HOG. Turned out that the classifier trained with these features generate too many false positives/negatives. It looked fine in the sample images, but once applied to the video there're too many flaws. So I chose RGB in the first trial.

The other issue I had was that the classifier generates false positives. I use heatmap and adjust the threshold value and it helped. Also choosing YCrCb in the second trial as color space when extracting features instead of RGB helped a lot.

The pipeline might fail if the video is filmed at night or rainy day. To make it more robust I think more image preprocessing is necessary. For example, use more than one color space features. Also, the sampling rate of HOG, bin size of the histogram and many other combinations of parameters should also be tested to fit different scenarios.
