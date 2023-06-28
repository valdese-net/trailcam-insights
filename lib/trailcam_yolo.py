from ultralytics import YOLO
import cv2
import imutils
import math
import csv

class Trailcam_YOLO:
	def __init__(self, modelname='yolov8l'):
		self.modelname = modelname
		self.model = YOLO("cache/%s.pt" % modelname)
		self.debugF = False

	def setDebug(self,f):
		self.debugF = f

	def predict(self, fn):
		classnames = self.model.names
		rdata = []
		vc = cv2.VideoCapture(fn)
		frame_count = 0
		fstats = {
			'filename': fn,
			'framerate': int(vc.get(cv2.CAP_PROP_FPS)),
			"model": self.modelname
		}

		if self.debugF: self.debugF(f'running predict on {fn}')

		while True:
			success, frame = vc.read()
			if not success: break

			timecode = math.ceil(vc.get(cv2.CAP_PROP_POS_MSEC))
			frame_count += 1

			for r in self.model.predict(frame, verbose=True, stream=True):
				if self.debugF and not (frame_count % 10): self.debugF(f'frame {frame_count}')
				for box in r.boxes:
					objname_idx = int(box.cls[0])
					objname = classnames[objname_idx]
					confidence = math.ceil(box.conf[0]*100)
					ptx,pty,sz_w,sz_h = box.xywh[0]
					ptx,pty,sz_w,sz_h = int(ptx),int(pty),int(sz_w),int(sz_h)
					d = [frame_count,timecode,objname,confidence,ptx,pty,sz_w,sz_h]
					rdata.append(d)

		return {
			"source": fstats,
			"detect": rdata
		}
