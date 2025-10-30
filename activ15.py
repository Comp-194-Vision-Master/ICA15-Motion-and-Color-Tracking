""" ==============================================================================================
File: activ15.py
Author: Susan Fox
Date: Fall 2025

Contains functions to practice with motion detection, background substitution, and color tracking
==================================================================================================
"""

import cv2
import numpy as np


# -----------------------------------------------------------------------------------------------------------
# simple motion detection


def simpleMotion(whichSource=0):
    """Takes in either a number for a webcam or a string that is the path/filename for a video file,
    and it connects to the video source. It processes each frame of the video with processImage, and
    displays the result, until the user hits q to quit."""
    vidCap = cv2.VideoCapture(whichSource)

    # Read a starter frame to be prevFrame
    ret, frame = vidCap.read()
    if not ret:
        print("Could not connect to camera")
        exit(0)

    prevFrame = frame

    while True:
        res, nextFrame = vidCap.read()
        if not res:
            print("Video feed done")
            break

        diffPic = cv2.absdiff(prevFrame, nextFrame)
        (b, g, r) = cv2.split(diffPic)
        diff2 = cv2.add(b, cv2.add(g, r))
        prevFrame = nextFrame

        cv2.imshow("Difference", diff2)
        x = cv2.waitKey(10)
        if x > 0:
            if chr(x) == 'q':
                break

    vidCap.release()


# -----------------------------------------------------------------------------------------------------------
# Experimenting with background subtraction

def backgroundSubtract(source, model):
    if model == 0:  # MOG2
        backsub = cv2.createBackgroundSubtractorMOG2()
    else:
        backsub = cv2.createBackgroundSubtractorKNN()

    capture = cv2.VideoCapture(source)
    while True:
        gotIm, frame = capture.read()
        if not gotIm:
            break

        fgMask = backsub.apply(frame)

        maskedFrame = cv2.bitwise_and(frame, frame, mask=fgMask)
        cv2.imshow('Frame', frame)
        cv2.imshow("Back Subtr Mask", fgMask)
        cv2.imshow("Masked image", maskedFrame)
        x = cv2.waitKey(10)
        if x > 0:
            if chr(x) == 'q':
                break

    capture.release()


# -----------------------------------------------------------------------------------------------------------
# Camshift object tracking

def camshift(refImage1):
    """Takes in a reference image and sets up the Camshift process for it. It makes a histograms, and sets
    up two a track window. It then runs Camshift on the images from a videe feed, using the histogram and
    track window, and draws the resulting track box as an ellipse on the image."""
    cam = cv2.VideoCapture(0)
    hueHist1 = makeHueHist(refImage1)
    trackWindow1 = None

    while True:
        ret, frame = cam.read()
        if not ret:
            print("No more frames...")
            break

        frame = cv2.flip(frame, 1)
        hgt, wid, dep = frame.shape

        # Initialize the track window to be the whole frame the first time
        if emptyTrackWindow(trackWindow1):
            trackWindow1 = (0, 0, wid, hgt)

        trackBox1, trackWindow1 = processFrame(frame, trackWindow1, hueHist1)
        cv2.ellipse(frame, trackBox1, (0, 0, 255), 2)

        cv2.imshow('camshift', frame)
        v = cv2.waitKey(5)
        if v > 0 and chr(v) == 'q':
            break


def makeHueHist(refImage):
    """Takes in a reference image, and it builds a histogram of its hue values. It masks away low value or
    low saturation pixels, leaving them out of the histogram. It normalizes the values in the histogram so that
    the max value is 255, letting us map from histogram values directly to brightness values in the back projection.
    It returns the histogram it created."""
    hsvRef = cv2.cvtColor(refImage, cv2.COLOR_BGR2HSV)
    maskedHistIm = cv2.inRange(hsvRef, (0, 60, 32), (180, 255, 255))
    hist = cv2.calcHist([hsvRef], [0], maskedHistIm, [16], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    hist = hist.reshape(-1)
    show_hist(hist)
    return hist


def emptyTrackWindow(trackW):
    """Takes in the current track window. If the track window is None, then this is the first frame, so we return
    True: the track wondow is empty right now and should be reset. If the track window exists, but its size is too
    small, then it returns True: the track window is nearly empty and should be reset. Otherwise, it returns False,
    meaning that the track window is not empty."""
    if trackW is None:
        return True
    else:
        (x1, y1, x2, y2) = trackW
        return abs(x2 - x1) < 5 or abs(y2 - y1) < 5


def processFrame(image, trackWindow, hist):
    """Takes in an image, the track window and the histogram, and it runs the Camshift process. It converts
    the image to HSV, masks away low saturation and low value pixels, and calculates the back-projection for
    the resulting image. It then runs Camshift, which updates the position of the track window and creates a
    track box, a rotated rectangle, for the object being tracked. It returns the track box and the track window."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert to HSV
    maskHSV = cv2.inRange(hsv, (0, 60, 32), (180, 255, 255))
    prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
    prob &= maskHSV
    cv2.imshow("Backproject", prob)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    trackBox, trackWindow = cv2.CamShift(prob, trackWindow, term_crit)
    return trackBox, trackWindow


def show_hist(hist):
    """Takes in the histogram, and displays it in the histogram window."""
    bin_count = hist.shape[0]
    bin_w = 24
    image = np.zeros((256, bin_count * bin_w, 3), np.uint8)
    for i in range(bin_count):
        h = int(hist[i])
        cv2.rectangle(image,
                      (i * bin_w + 2, 255),
                      ((i + 1) * bin_w - 2, 255 - h),
                      (int(180.0 * i / bin_count), 255, 255),
                      -1)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imshow('histogram', image)



# ===========================================================================================================
# Main script

if __name__ == "__main__":
    # TODO: Put sample calls to your functions below this, along with reading in images, etc.
    # -----------------------------------------------------------------------------------------------------------
    # Example with simple motion detection
    simpleMotion(0)
    simpleMotion("../SampleVideos/People - 6387.mp4")
    simpleMotion("../SampleVideos/CarsStreet.avi")

    # -----------------------------------------------------------------------------------------------------------
    # Example with simple motion detection
    MOG2MODEL = 0
    KNNMODEL = 1
    backgroundSubtract(0, MOG2MODEL)
    backgroundSubtract("../SampleVideos/People - 6387.mp4", KNNMODEL)
    backgroundSubtract("../SampleVideos/run.mp4", MOG2MODEL)

    # -----------------------------------------------------------------------------------------------------------
    # Example with CamShift
    refIm = cv2.imread("refBlue.png")
    camshift(refIm)
