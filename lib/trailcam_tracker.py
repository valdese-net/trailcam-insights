from ultralytics import YOLO
import cv2
import imutils
import math

class TrailcamTracker:
	def __init__(self, modelname='yolov8l-seg'):
		self.model = YOLO("cache/%s.pt" % modelname)
		self.debugF = False

	def setDebug(self,f):
		self.debugF = f

	def scan(self, fn):
		frame_count = 0
		for r in self.model.track(fn, stream=True):
			frame_count += 1

			classnames = r.names
			for box in r.boxes:
				objname_idx = int(box.cls[0])
				objname = classnames[objname_idx]
				if self.debugF:
					self.debugF(f"{frame_count} {objname}")
					self.debugF(box.xyxy)
					self.debugF(box.id)

		return {
			"frame_count": frame_count
		}
