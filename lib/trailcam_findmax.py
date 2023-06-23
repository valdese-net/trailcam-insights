from ultralytics import YOLO
import cv2
import imutils
import math

class TrailcamObjectGrp:
	def __init__(self,objtype,frameidx):
		self.objtype = objtype
		self.frameidx = frameidx
		self.detections = []

	def __str__(self):
		return f'({self.objtype} in frame #{self.frameidx}, count {self.count()}, {self.detections})'

	__repr__ = __str__

	def add(self, box):
		conf = math.ceil(box.conf[0]*100)
		self.detections.append(conf)
		self.detections.sort()

	def count(self):
		return len(self.detections)

	def minConfidence(self):
		return self.detections[0]


class TrailcamFindMax:
	def __init__(self):
		self.model = YOLO('cache/yolov8l.pt')
		self.debugF = False

	def setDebug(self,f):
		self.debugF = f

	def scan(self, fn):
		vc = cv2.VideoCapture(fn)
		framerate = int(vc.get(cv2.CAP_PROP_FPS))
		keyframes = max(framerate,5) // 2
		frame_count = 0
		sample_count = 0
		pstats = {}
		success = True

		while success:
			frame_count += 1
			if (frame_count % keyframes) != 1:
				success = vc.grab()
				continue

			fstats = {}
			success, img = vc.read()
			if not success:
				break

			#img = imutils.resize(img,height=1000)
			results = self.model(img,verbose=False)
			sample_count += 1

			for r in results:
				classnames = r.names
				for box in r.boxes:
					objname_idx = int(box.cls[0])
					objname = classnames[objname_idx]
					if not objname in fstats:
						fstats[objname] = TrailcamObjectGrp(objname,frame_count)
					fstats[objname].add(box)

			if self.debugF and (len(fstats) > 0): self.debugF(fstats)

			for k in fstats:
				keepit = False
				if not k in pstats:
					keepit = True
				elif pstats[k].count() < fstats[k].count():
					keepit = True
				elif (pstats[k].count() == fstats[k].count()) and (pstats[k].minConfidence() < fstats[k].minConfidence()):
					keepit = True

				if keepit: pstats[k] = fstats[k]

		return {
			"framerate": framerate,
			"frame_count": frame_count,
			"sample_count": sample_count,
			"detected": pstats
		}
