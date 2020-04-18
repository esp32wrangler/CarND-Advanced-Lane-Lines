import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

class CalcData():
    xkernelsize = 3
    h_tresh_low = 21
    h_tresh_high = 26
    hx_tresh_low = 1
    hx_tresh_high = 0
    h_dir_low = 1
    h_dir_high = 0
    h_mag_low = 1
    h_mag_high = 0
    l_tresh_low = 201
    l_tresh_high = 255
    lx_tresh_low = 1
    lx_tresh_high = 0
    l_dir_low = 1
    l_dir_high = 0
    l_mag_low = 1
    l_mag_high = 0
    s_tresh_low = 1
    s_tresh_high = 0
    s_mag_low = 1
    s_mag_high = 0
    sx_tresh_low = 1
    sx_tresh_high = 0
    s_dir_low = 1
    s_dir_high = 0

    frame_num = 0

    def __init__(self):
       pass


class PipelineTuner(object):

    hwindow = "H"
    lwindow = "L"
    swindow = "S"
    h_channel = None
    l_channel = None
    s_channel = None
    combinedwindow = "Combined"
    data = CalcData()
    #clip1 = VideoFileClip("challenge_video.mp4")
    clip1 = VideoFileClip("project_video.mp4")
    n_frames = None
    def __init__(self):
        self.n_frames = self.clip1.reader.nframes

        cv2.namedWindow(self.hwindow)
        cv2.namedWindow(self.lwindow)
        cv2.namedWindow(self.swindow)
        cv2.namedWindow(self.combinedwindow)

        cv2.createTrackbar("hx_tresh_low", self.hwindow,
                           self.data.hx_tresh_low, 255,
                           self.up_hx_tresh_low)
        cv2.createTrackbar("hx_tresh_high", self.hwindow,
                           self.data.hx_tresh_high, 255,
                           self.up_hx_tresh_high)
        cv2.createTrackbar("lx_tresh_low", self.lwindow,
                           self.data.lx_tresh_low, 255,
                           self.up_lx_tresh_low)
        cv2.createTrackbar("lx_tresh_high", self.lwindow,
                           self.data.lx_tresh_high, 255,
                           self.up_lx_tresh_high)
        cv2.createTrackbar("sx_tresh_low", self.swindow,
                           self.data.sx_tresh_low, 255,
                           self.up_sx_tresh_low)
        cv2.createTrackbar("sx_tresh_high", self.swindow,
                           self.data.sx_tresh_high, 255,
                           self.up_sx_tresh_high)

        cv2.createTrackbar("h_tresh_low", self.hwindow,
                           self.data.h_tresh_low, 255,
                           self.up_h_tresh_low)
        cv2.createTrackbar("h_tresh_high", self.hwindow,
                           self.data.h_tresh_high, 255,
                           self.up_h_tresh_high)
        cv2.createTrackbar("l_tresh_low", self.lwindow,
                           self.data.l_tresh_low, 255,
                           self.up_l_tresh_low)
        cv2.createTrackbar("l_tresh_high", self.lwindow,
                           self.data.l_tresh_high, 255,
                           self.up_l_tresh_high)
        cv2.createTrackbar("s_tresh_low", self.swindow,
                           self.data.s_tresh_low, 255,
                           self.up_s_tresh_low)
        cv2.createTrackbar("s_tresh_high", self.swindow,
                           self.data.s_tresh_high, 255,
                           self.up_s_tresh_high)
        cv2.createTrackbar("s_dir_low", self.swindow,
                           self.data.s_dir_low, 314,
                           self.up_s_dir_low)
        cv2.createTrackbar("s_dir_high", self.swindow,
                           self.data.s_dir_high, 314,
                           self.up_s_dir_high)
        cv2.createTrackbar("h_dir_low", self.hwindow,
                           self.data.h_dir_low, 314,
                           self.up_h_dir_low)
        cv2.createTrackbar("h_dir_high", self.hwindow,
                           self.data.h_dir_high, 314,
                           self.up_h_dir_high)
        cv2.createTrackbar("l_dir_low", self.lwindow,
                           self.data.l_dir_low, 314,
                           self.up_l_dir_low)
        cv2.createTrackbar("l_dir_high", self.lwindow,
                           self.data.l_dir_high, 314,
                           self.up_l_dir_high)

        cv2.createTrackbar("s_mag_low", self.swindow,
                           self.data.s_mag_low, 314,
                           self.up_s_mag_low)
        cv2.createTrackbar("s_mag_high", self.swindow,
                           self.data.s_mag_high, 314,
                           self.up_s_mag_high)
        cv2.createTrackbar("h_mag_low", self.hwindow,
                           self.data.h_mag_low, 314,
                           self.up_h_mag_low)
        cv2.createTrackbar("h_mag_high", self.hwindow,
                           self.data.h_mag_high, 314,
                           self.up_h_mag_high)
        cv2.createTrackbar("l_mag_low", self.lwindow,
                           self.data.l_mag_low, 314,
                           self.up_l_mag_low)
        cv2.createTrackbar("l_mag_high", self.lwindow,
                           self.data.l_mag_high, 314,
                           self.up_l_mag_high)

        cv2.createTrackbar("frame_num", self.combinedwindow,
                           self.data.frame_num, self.n_frames,
                           self.change_frame)
        self.change_frame(0)


    def up_sxkernelsize (self,x):
       self.data.sxkernelsize = x + 1 - x%2
       self.update (x)

    def up_sx_tresh_low (self,x):
       self.data.sx_tresh_low = x
       self.update (x)

    def up_sx_tresh_high (self,x):
       self.data.sx_tresh_high = x
       self.update (x)
    def up_hx_tresh_low (self,x):
       self.data.hx_tresh_low = x
       self.update (x)

    def up_hx_tresh_high (self,x):
       self.data.hx_tresh_high = x
       self.update (x)
    def up_lx_tresh_low (self,x):
       self.data.lx_tresh_low = x
       self.update (x)

    def up_lx_tresh_high (self,x):
       self.data.lx_tresh_high = x
       self.update (x)
    def up_h_tresh_low (self,x):
       self.data.h_tresh_low = x
       self.update (x)

    def up_h_tresh_high (self,x):
       self.data.h_tresh_high = x
       self.update (x)

    def up_l_tresh_low (self,x):
       self.data.l_tresh_low = x
       self.update (x)

    def up_l_tresh_high  (self,x):
       self.data.l_tresh_high = x
       self.update (x)

    def up_s_tresh_low  (self,x):
       self.data.s_tresh_low = x
       self.update (x)

    def up_s_tresh_high  (self,x):
       self.data.s_tresh_high = x
       self.update (x)

    def up_s_dir_low  (self,x):
       self.data.s_dir_low = x
       self.update (x)

    def up_s_dir_high  (self,x):
       self.data.s_dir_high = x
       self.update (x)

    def up_h_dir_low  (self,x):
       self.data.h_dir_low = x
       self.update (x)

    def up_h_dir_high  (self,x):
       self.data.h_dir_high = x
       self.update (x)

    def up_l_dir_low  (self,x):
       self.data.l_dir_low = x
       self.update (x)

    def up_l_dir_high  (self,x):
       self.data.l_dir_high = x
       self.update (x)

    def up_s_mag_low  (self,x):
       self.data.s_mag_low = x
       self.update (x)

    def up_s_mag_high  (self,x):
       self.data.s_mag_high = x
       self.update (x)

    def up_h_mag_low  (self,x):
       self.data.h_mag_low = x
       self.update (x)

    def up_h_mag_high  (self,x):
       self.data.h_mag_high = x
       self.update (x)

    def up_l_mag_low  (self,x):
       self.data.l_mag_low = x
       self.update (x)

    def up_l_mag_high  (self,x):
       self.data.l_mag_high = x
       self.update (x)

    def change_frame(self, x):
        self.data.frame_num = x
        img = self.clip1.get_frame(self.data.frame_num*1.0/self.clip1.fps)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        self.h_channel = hls[:, :, 0]
        self.l_channel = hls[:, :, 1]
        self.s_channel = hls[:, :, 2]
        self.update(0)

    def update(self, x):
        h, l, s, combined = get_images(self.data, self.h_channel, self.l_channel, self.s_channel)

        cv2.imshow(self.hwindow, h)
        cv2.imshow(self.lwindow, l)
        cv2.imshow(self.swindow, s)
        combined = combined * 255
        combined [460,:,0] = 255
        cv2.imshow(self.combinedwindow, combined)

def channel_process (channel, sxlow, sxhigh, channellow, channelhigh, channeldirlow, channeldirhigh, channelmaglow, channelmaghigh):
	# Sobel x
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1)  # Take the derivative in x
    abs_sobely = np.absolute(sobely)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))

    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    #dir_output = np.zeros_like(channel)
    #dir_output[(absgraddir >= channeldirlow/100) & (absgraddir <= channeldirhigh/100)] = 1

    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    #mag_output = np.zeros_like(gradmag)
    #mag_output[(gradmag >= channelmaglow) & (gradmag <= channelmaghigh)] = 1
    magdir_output = np.zeros_like(gradmag)
    magdir_output[
        ((gradmag >= channelmaglow) & (gradmag <= channelmaghigh)) | ((absgraddir >= channeldirlow/100) & (absgraddir <= channeldirhigh/100))] = 1

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sxlow) & (scaled_sobel <= sxhigh)] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(channel)
    s_binary[(channel >= channellow) & (channel <= channelhigh)] = 1
    # Stack each channel
    color_binary = np.dstack(( magdir_output, sxbinary, s_binary)) * 255

    return color_binary


def get_images(data, h_channel, l_channel, s_channel):
    # Convert to HLS color space and separate the V channel
    
    hres = channel_process (h_channel, data.hx_tresh_low, data.hx_tresh_high, data.h_tresh_low, data.h_tresh_high, data.h_dir_low, data.h_dir_high, data.h_mag_low, data.h_mag_high)
    lres = channel_process (l_channel, data.lx_tresh_low, data.lx_tresh_high, data.l_tresh_low, data.l_tresh_high, data.l_dir_low, data.l_dir_high, data.l_mag_low, data.l_mag_high)
    sres = channel_process (s_channel, data.sx_tresh_low, data.sx_tresh_high, data.s_tresh_low, data.s_tresh_high, data.s_dir_low, data.s_dir_high, data.s_mag_low, data.s_mag_high)

    combined = np.zeros_like(hres)
    combined[((hres != 0) | (lres != 0) | (sres !=0))] = 1
    
    return hres, lres, sres, combined
    


def process_image(image):
    return image





thingie = PipelineTuner()
cv2.waitKey()



