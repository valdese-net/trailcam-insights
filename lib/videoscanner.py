from ultralytics import YOLO
import cv2
import imutils

class VideoScanner:
	def __init__(self):
		self.model = YOLO('cache/yolov8l.pt')

	def scan(self, fn):
		vc = cv2.VideoCapture(fn)
		framerate = int(vc.get(cv2.CAP_PROP_FPS))
		keyframes = max(framerate,5)
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
			results = self.model(img,stream=True,verbose=False)
			sample_count += 1

			for r in results:
				classnames = r.names
				for box in r.boxes:
					objname_idx = int(box.cls[0])
					objname = classnames[objname_idx]
					fstats[objname] = fstats.get(objname, 0) + 1

			for k in fstats:
				pstats[k] = max(fstats[k],pstats.get(k,0))

		return {
			"framerate": framerate,
			"frame_count": frame_count,
			"sample_count": sample_count,
			"detected": pstats
		}
