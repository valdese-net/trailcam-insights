from ultralytics import YOLO
import os
import cv2
import imutils

class Trailcam_YOLO:
	def __init__(self, modelname='yolov8l'):
		self.modelname = modelname
		self.model = YOLO("cache/%s.pt" % modelname)
		self.debugF = False

	def setDebug(self,f):
		self.debugF = f

	def predict(self, fn, resized_fn=False):
		last_modified = os.path.getmtime(fn)
		classnames = self.model.names
		rdata = []
		vc = cv2.VideoCapture(fn)
		writer = False
		frame_count = 0
		frame_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fstats = {
			'filename': fn,
			'timestamp': last_modified,
			'framerate': vc.get(cv2.CAP_PROP_FPS),
			'framesize-orig': (frame_width,frame_height),
			'framesize': (1000,(1000*frame_height) // frame_width),
			"model": self.modelname
		}

		if self.debugF: self.debugF(f'running predict on {fstats}')

		if resized_fn:
			fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
			writer = cv2.VideoWriter(resized_fn, fourcc, fstats['framerate'], fstats['framesize'])

		while True:
			success, frame = vc.read()
			if not success: break

			timecode = round(vc.get(cv2.CAP_PROP_POS_MSEC))
			frame_count += 1

			frame = cv2.resize(frame, fstats['framesize'])
			if writer: writer.write(frame)

			for r in self.model.predict(frame, verbose=True, stream=True):
				if self.debugF and not (frame_count % 10): self.debugF(f'frame {frame_count}')
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
