import cv2
import tkinter
import imutils
from pathlib import Path
import os
import time

class TrailcamTrain:
	def __init__(self,outpath):
		# Initializing the HOG person detector
		self.outpath = outpath
		self.fn = False
		self.framenum = 0

	def train(self,fn):
		# Reading the Image
		self.fn = fn
		self.framenum = 0
		vc = cv2.VideoCapture(str(fn))

		last_modified = os.path.getmtime(fn)
		mtime = time.localtime(last_modified)
		mtime_str = time.strftime('%Y%m%d_%H%M%S', mtime)

		sz = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		newsz = (640,(sz[1]*640)//sz[0])

		print('Press <s> to save frane, any other key to advance')
		while True:
			success, image = vc.read()
			if not success: break
			self.framenum += 1

			# Resizing the Image
			# After testing imutils and cv2, the imutils version looks better, likely due to anti-aliasing
			#image = cv2.resize(image,newsz)
			image = imutils.resize(image,width=newsz[0])

			# Showing the output Image
			cv2.imshow('Image', image)

			key = cv2.waitKey(0)
			# Check if the 'Esc' key was pressed
			if key == 27: break
			if key == ord('s'):
				fn_img = self.fn.with_suffix(f'.{self.framenum}.jpg')
				if self.outpath: fn_img = Path(self.outpath).joinpath(f'{mtime_str}_{self.framenum}.jpg')
				print('saving to ',fn_img)
				cv2.imwrite(str(fn_img),image)

		cv2.destroyAllWindows()