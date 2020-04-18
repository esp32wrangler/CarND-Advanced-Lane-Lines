from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

warpsrc=np.float32([[579,460], [700,460], [1060,690], [230, 690]])
# - till y=500
#warpsrc=np.float32([[519,500], [762,500], [1060,690], [230, 690]])
warpdst=np.float32([[300,0], [980,0], [980, 720], [300, 720]])
calM, calD, calRV, calTV = pickle.load( open( "cameracal.pickle", "rb" ) )



ym_per_pix = 3/100 # meters per pixel in y dimension
xm_per_pix = 3.7/664 # meters per pixel in x dimension

class State ():
    left_fit=None
    right_fit=None
    leftx=None
    rightx=None
    def __init__(self):
        pass

warpM = cv2.getPerspectiveTransform(warpsrc, warpdst)
unwarpM = cv2.getPerspectiveTransform(warpdst,warpsrc)

def warp(img):
    global warpM
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])

    warped = cv2.warpPerspective(img, warpM, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def unwarp(img):
    global unwarpM
    img_size = (img.shape[1], img.shape[0])

    unwarped = cv2.warpPerspective(img, unwarpM, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return unwarped

def color_detect(img, alternative_mapping = False):
    hchannellow = 21
    hchannelhigh = 26
    lchannellow = 201
    lchannelhigh = 255
    schannellow = 125
    schannelhigh = 255

    # turn off the S channel if there are too many stray pixels
    if alternative_mapping:
        schannellow=1
        schannelhigh=0


    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]


    #sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    #abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    #scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    #sxbinary = np.zeros_like(scaled_sobel)
    #sxbinary[(scaled_sobel >= sxlow) & (scaled_sobel <= sxhigh)] = 1
    #cv2.imshow("sxbin", sxbinary * 255)
    #cv2.waitKey(3000)
    # Threshold color channel
    #s_binary = np.zeros_like(s_channel)
    #s_binary[(s_channel >= channellow) & (s_channel <= channelhigh)] = 1
    # Stack each channel
    #cv2.imshow("s_binary", s_binary * 255)
    #cv2.waitKey(3000)

    colored_img = np.zeros_like(s_channel)
    #colored_img[ ((s_channel >= schannellow) & (s_channel <= schannelhigh))] = 1
    #colored_img[ ((h_channel >= hchannellow) & (h_channel <= hchannelhigh)) ] = 1
    colored_img[
        ((l_channel >= lchannellow) & (l_channel <= lchannelhigh)) |
        ((h_channel >= hchannellow) & (h_channel <= hchannelhigh)) |
        ((s_channel >= schannellow) & (s_channel <= schannelhigh))
    ] = 1
    return colored_img




def histogram_based_lane_search(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = np.int(histogram.shape[0] // 2)
    leftx = np.argmax(histogram[:midpoint])
    rightx = np.argmax(histogram[midpoint:]) + midpoint
    return leftx, rightx

def sliding_window_search(state, binary_warped):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = state.leftx
    rightx_current = state.rightx

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, len(nonzerox)-len(leftx)-len(rightx), out_img

def fit_with_window(state, binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, nonmatched_num, out_img = sliding_window_search(state, binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_fit_world = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_world = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return left_fit, right_fit, left_fit_world, right_fit_world, nonmatched_num, out_img

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_fit_world = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_world = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


    return left_fitx, right_fitx, left_fit, right_fit, left_fit_world, right_fit_world, ploty


def search_around_poly(state, binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = (np.abs(nonzerox - state.left_fit[0] * (nonzeroy ** 2) - state.left_fit[1] * nonzeroy -
                             state.left_fit[2]) < margin)
    right_lane_inds = (np.abs(nonzerox - state.right_fit[0] * (nonzeroy ** 2) - state.right_fit[1] * nonzeroy -
                              state.right_fit[2]) < margin)

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, left_fit, right_fit, left_fit_world, right_fit_world, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

    return left_fit, right_fit, left_fit_world, right_fit_world, len(nonzerox)-len(leftx)-len(rightx), result


def find_lane_pixels(state, binary_warped, reinitialize):
    if state.leftx is None or reinitialize:
        state.leftx, state.rightx = histogram_based_lane_search(binary_warped)
        left_fit, right_fit, left_fit_world, right_fit_world, nonmatched_num, illustration = fit_with_window(state, binary_warped)
    else:
        left_fit, right_fit, left_fit_world, right_fit_world, nonmatched_num, illustration = search_around_poly(state, binary_warped)

    return left_fit, right_fit, left_fit_world, right_fit_world, nonmatched_num, illustration


def measure_curvature_real(y_eval, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''

    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad


def highlight_drivable_area(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    lane_highlight = np.zeros_like(img)

    xleft = left_fit[0] * (ploty ** 2) + left_fit[1] * ploty + left_fit[2]
    xright = right_fit[0] * (ploty ** 2) + right_fit[1] * ploty + right_fit[2]

    for l, r, y in zip(xleft, xright, ploty):
        li = int(l)
        ri = int(r)
        yi = int(y)
        lane_highlight[yi, li:ri, 1 ] = 255
        lane_highlight[yi, li - 4:li + 4, 0] = 255
        lane_highlight[yi, ri - 4:ri + 4, 0] = 255
        lane_highlight[yi, li - 4:li + 4, 2] = 255
        lane_highlight[yi, ri - 4:ri + 4, 2] = 255

    return lane_highlight

def check_lane_distance(left_fit_world, right_fit_world):
    ploty = np.linspace(0, 29, 30)
    xleft = left_fit_world[0] * (ploty ** 2) + left_fit_world[1] * ploty + left_fit_world[2]
    xright =  right_fit_world[0] * (ploty ** 2) + right_fit_world[1] * ploty + right_fit_world[2]
    dist = xright-xleft
    return np.min(dist), np.max(dist)


frameno = 0
ignored_frames=0
make_submission_files=True

def process_frame (image):
    global frameno, make_submission_files
    global process_state, ignored_frames
    global calM, calD
    undist = cv2.undistort(image, calM, calD)

    colored = color_detect(image)
    #cv2.imshow("colored", colored*255)

    #cv2.waitKey(3000)
    warped = warp(colored)
    #cv2.imshow("warped", warped*255)

    #cv2.imshow("warpimg", warp(image))
    #cv2.waitKey(0)
    if make_submission_files:
        cv2.imwrite("images_and_videos/undistorted.jpg", cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
        cv2.imwrite("images_and_videos/colored.jpg", colored * 255)
        cv2.imwrite("images_and_videos/warped.jpg", warped * 255)

    # After ignored_frame_treshold frames that resulted in incorrect data, reinitialize the lane search window
    ignored_frame_treshold = 5
    reinitialize = ignored_frames > ignored_frame_treshold
    left_fit, right_fit, left_fit_world, right_fit_world, nonmatched_num, illustration = find_lane_pixels(process_state, warped, reinitialize)
    # if there are too many stray pixels that do not belong to the lane line, switch to an alternative color mapping and retry
    alternative_colormap = False
    if nonmatched_num > 10000:
        alternative_colormap = True
        coloredalt = color_detect(image, alternative_colormap)
        warpedalt = warp(coloredalt)
        left_fitalt, right_fitalt, left_fit_worldalt, right_fit_worldalt, nonmatched_numalt, illustrationalt = find_lane_pixels(process_state, warpedalt, reinitialize)
        if nonmatched_numalt < 10000:
            left_fit, right_fit, left_fit_world, right_fit_world,  illustration = left_fitalt, right_fitalt, left_fit_worldalt, right_fit_worldalt, illustrationalt
    prevl=process_state.left_fit
    prevr=process_state.right_fit
    # Experiment: Only accept a new curve it the third coefficient is within fit_accept_treshold units from previous. It worked OK, but was not needed for this video.
    #fit_accept_treshold = 100
    '''if (reinitialize or np.abs(process_state.left_fit[2]-left_fit[2]) < fit_accept_treshold):
        process_state.left_fit = left_fit
        ignored_framesl = 0
    else:
        ignored_framesl += 1
    if (reinitialize or np.abs(process_state.right_fit[2]-right_fit[2]) < fit_accept_treshold):
        process_state.right_fit = right_fit
        ignored_framesr = 0
    else:
        ignored_framesr += 1
    '''

    # the two lane markers should be roughly parallel, within 0.6m error on 30 meters
    mind, maxd = check_lane_distance(left_fit_world, right_fit_world)
    lane_divergence_treshold = 0.6
    if (process_state.left_fit is None or maxd-mind < lane_divergence_treshold):
        process_state.left_fit = left_fit
        process_state.right_fit = right_fit
        ignored_frames = 0
    else:
        ignored_frames += 1

    #cv2.imshow("illustration", illustration)
    #cv2.imwrite("illustration.jpg", illustration)
    #cv2.waitKey(1000)
    left_curverad, right_curverad = measure_curvature_real(warped.shape[0], left_fit_world, right_fit_world)

    #print ("curves: ", left_curverad, right_curverad)

    unwarped= unwarp(illustration)

    highlight = highlight_drivable_area(image, process_state.left_fit, process_state.right_fit)
    unwarped_highlight = unwarp(highlight)
    result = cv2.addWeighted(image, 0.7, unwarped_highlight, 0.3, 0.8, 0)

    xleft = left_fit_world[0] * (720*ym_per_pix ** 2) + left_fit_world[1] * 720*ym_per_pix + left_fit_world[2]
    xright = right_fit_world[0] * (720*ym_per_pix ** 2) + right_fit_world[1] * 720*ym_per_pix + right_fit_world[2]
    off_center = (xleft+xright)/2-(image.shape[1]/2)*xm_per_pix

    #result = cv2.addWeighted(result, 0.7, unwarped, 0.8, 0.7, 0)

    cv2.putText(result, f'{frameno} Curvature: {(left_curverad+right_curverad)//2} ', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2 )
    cv2.putText(result, f'lane dist: min:{mind:0.2}, max:{maxd:0.2} {maxd-mind:0.2}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(result, f'stray pixels:{nonmatched_num} alt: {nonmatched_numalt if alternative_colormap else -1}', (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(result, f'distance from center:{off_center:0.2}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # composit in a view of the warped, color mapped lane view
    small = cv2.resize(cv2.cvtColor(warpedalt*255 if alternative_colormap else warped*255, cv2.COLOR_GRAY2BGR), (355, 200))
    x_offset = 900
    y_offset = 10
    result[y_offset:y_offset + small.shape[0], x_offset:x_offset + small.shape[1]] = small

    #cv2.imshow("result", cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    #cv2.waitKey(10000)
    frameno += 1
    return result


process_state=State()

video_output = 'images_and_videos/out.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the  first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("project_video.mp4")
outclip = clip1.fl_image(process_frame) #NOTE: this function expects color images!!
outclip.write_videofile(video_output, audio=False)