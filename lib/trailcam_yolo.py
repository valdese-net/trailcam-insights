from ultralytics import YOLO
import os
import cv2
import imutils

class Trailcam_YOLO:
	def __init__(self, modelname='yolov8l'):
		self.modelname = modelname
		self.model = False
		self.debugF = False

	def setDebug(self,f):
		self.debugF = f

	def predict(self, fn, resize_fn=False):
		fn = str(fn) # fn is a Path object, which can cause problems
		if not self.model:
			if self.debugF: self.debugF(f'loading YOLOv8({self.modelname})')
			if self.modelname.startswith('yolo'):
				self.model = YOLO("cache/%s.pt" % self.modelname)
			else:
				self.model = YOLO(self.modelname)

		last_modified = os.path.getmtime(fn)
		classnames = self.model.names
		rdata = []
		vc = cv2.VideoCapture(fn)
		writer = False
		frame_count = 0
		frame_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
		frame_new_width = 640
		fstats = {
			'filename': fn,
			'timestamp': last_modified,
			'framerate': vc.get(cv2.CAP_PROP_FPS),
			'framesize-orig': (frame_width,frame_height),
			'framesize': (frame_new_width,(frame_new_width*frame_height) // frame_width),
			"model": self.modelname
		}

		if self.debugF: self.debugF(f'running predict on {fstats}')

		if resize_fn:
			fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
			writer = cv2.VideoWriter(str(resize_fn), fourcc, fstats['framerate'], fstats['framesize'])

		while True:
			success, frame = vc.read()
			if not success: break

			timecode = round(vc.get(cv2.CAP_PROP_POS_MSEC))
			frame_count += 1

			# After testing imutils and cv2, the imutils version looks better, likely due to anti-aliasing
			#frame = cv2.resize(frame, fstats['framesize'])
			frame = imutils.resize(frame,width=frame_new_width)
			if writer: writer.write(frame)

			for r in self.model.predict(frame, verbose=(self.debugF != False), stream=True):
				for box in r.boxes:
					objname_idx = int(box.cls[0])
					objname = classnames[objname_idx]
					confidence = int(box.conf[0]*100)
					ptx,pty,sz_w,sz_h = box.xywh[0]
					ptx,pty,sz_w,sz_h = int(ptx),int(pty),int(sz_w),int(sz_h)
					d = [frame_count,timecode,objname,confidence,ptx,pty,sz_w,sz_h]
					rdata.append(d)

		if writer: writer.release()

		return {
			"source": fstats,
			"detect": rdata
		}
