import cv2

VIDEO_PATH = "pedestrians.mp4"
vs = cv2.VideoCapture(VIDEO_PATH)
writer = None
(W, H) = (None, None)

# Try calculating the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# Raise an error if the number of frames couldn't be calculated
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1