#**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images_and_videos/cal_demo.png "Undistorted chessboard after calibration"
[image2]: ./images_and_videos/undistorted.jpg "Road Transformed"
[image3]: ./images_and_videos/colored.jpg "Binary Example"
[image4]: ./images_and_videos/straight_lines_warped.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./images_and_videos/lane_highlight.png "Output"
[video]: ./images_and_videos/final_output.mp4 "Video"

 [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

## Writeup / README

My solution consists of two python files, cameracal.py and process_video.py
. Other than the camera calibration functionality, every function is in the
 process_video.py file.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I copied the core code from "./examples/example.ipynb" into cameracal.py
. This code created the necessary image point and object point arrays, and
  filled out the object points with a 9x6 mesh. 

It used the cv2.findChessboardCorners function to identify the corners in
 each image, followed by the cv2.calibrateCamera function to build a
  calibration matrix from this data.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

I also added a couple of lines of code to save the calibration coefficients
 as a pickle file, so that the image processing step doesn't have to waste
  time redoing the calibration each time it is run (each time it is debugged
   :) )

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I used the pickle file created by the cameracal.py program to load the camera
 distortion matrix. The frames in the process_frame pipeline are undistorted
 , using the cv2.undistort function.  
Here is an example of the result:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I wrote a quick-and-dirty tool that would let me experiment with the effect
of applying different pixel value tresholds as well as Sobel (sobel x
, magnitude, angle) tresholds to all frames in the project video (see
hsltester.py). I ended up identifying two different combinations of
parameters, one I use as a main profile, and one as an alternate (see
details later). 

My primary choice was a simple tresholding of the H, S and L channels, and
combining the result. I was unable to get any better results with either a
Sobel transformation or a Sobel magnitude or angle calculation for this
particular video. See the parameters in the color_detect function.

This color_detect function generates a binary image such as this:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
warpsrc=np.float32([[579,460], [700,460], [1060,690], [230, 690]])
warpdst=np.float32([[300,0], [980,0], [980, 720], [300, 720]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

I also used this opportunity to measure the world coordinate to pixel
 coordinate conversion factor, using the distance between the two lanes and
  the length of each dashed line segment.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the approach I learned in the previous chapters of the course. First, I
 use a histogram-based approach on the bottom half of the image to identify the
  most likely starting poisition for the left and right lane markers. Then, I
   use a sliding window algorithm to track the lane lines across the screen
   , from bottom to top. Then on subsequent frames I search for the lane lines
    in the positions that are predicted based on the polynomial approximation of
the lanes in the previous frame.

I found that the biggest challenge was knowing when the data produced by the
 algorithm is correct. Incorrect data produces an incorrect lane
  interpolation, which can make subsequent interpolations incorrect as well.
  
I settled on two key quality indicators for the interpolated lane lines:
* The lanes have to be parallel, meaning that over a 30m stretch the
 difference between the closest point of the two interpolated points and the
  farthest away point should be within 0.6 m. This way if either lane is
   misdetected, the incorrect value can be quickly filtered out. See
    the check_lane_distance function.
* The number of "stray pixels" that are falling outside of the detection
 window for both left and right lanes have to be below a certain treshold
 (10000 in this case). If there are more pixels, it means that there is a
 high likelyhood that there is too much noise in the image and all three
 algorithms (histogram, sliding window and polynomial-predictive) will
 generate random data. In cases where a lot of stray pixels are detected
 , my code will use an alternative approach to performing the color
 transformation. (Toward the end of the video there is a section where the S channel is drowned
 by big blotches of noise. So my alternative mapping ignores the S channel
  altogether, and uses the lower quality H and L channel data)
 
If the algorithms fail to detect two parallel lane markers on more than 5
 consecutive frames, the algorithm reinitializes itself, and starts with
  the histogram and sliding window detection. 

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I used the measure_curvature_real function that I had to complete in the
 previous lesson of the course. When I interpolate the lane lines in pixel
  coordinates, I simultaneously also calculate the world coordinate
   interpolation, and from there it is only a question of applying the
    formula to the interpolated polynomial.

The world coordinate interpolation is important for both the lane distance
 calculation and the lane center line offset calculation. Calculating the
  center is done by comparing the mean of the two lane world coordinates with
   the the car center line world coordinate. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I first draw a green rectangle with purple outlines into an image buffer in the
 highlight_drivable_area function. This is done in perspective corrected
  space first and then warped back to the original undistorted image space
   and composited on top of the undistorted camera image. This is done around
    line 400 of process_video.py

Here is an example of the final output:

![alt text][image6]

---

### Pipeline (video)

Here's the [video]

---

### Discussion

As I mentioned in section 4 above, the difficulty was to tell when the
 results are plausible and when to ignore them, or substitute a different
  approach. It is also a very long pipeline with lots of moving parts and
   tunable parameters, and each one impacts the others. Therefore I could
    easily spend days further tuning the parameters, even without major
     changes ot the pipeline.
     
I will come back and solve the challenge puzzles as well, as they are showing
 some very interesting special cases. I'm afraid that even beyond those every
  new video would highlight fragile components in this solution. 
  
This lane detection algorithm indeed works a lot better than the one in the
previous project, but many scenarios are still not handled, such as up and down
slopes, twilight/night driving, imperfect camera alignment, other cars in
our lane, secondary, higher focal length cameras to provide better look
-ahead, etc... 

