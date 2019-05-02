import sys
import cv2

# https://github.com/JetsonHacksNano/CSI-Camera
def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0) :
	return ('nvarguscamerasrc ! ' 
	'video/x-raw(memory:NVMM), '
	'width=(int)%d, height=(int)%d, '
	'format=(string)NV12, framerate=(fraction)%d/1 ! '
	'nvvidconv flip-method=%d ! '
	'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
	'videoconvert ! '
	'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

if __name__ == '__main__':
	image = cv2.imread('lena.jpg')
	cv2.imshow('Display', image)

	# cap = cv2.VideoCapture(1)
	cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

	if cap.isOpened():
		while True:
			ret_val, img = cap.read()
			cv2.imshow('DisplayCamera', img)
			key = cv2.waitKey(10)
			if key > 0:
				break
		cap.release()
	else:
		print('cannot open camera')
		cv2.waitKey(-1)

	cv2.destroyAllWindows()
