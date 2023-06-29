from lib.trailcam_yolo import *
import sys
import os

class MainApp:
	def __init__(self, modelname='yolov8l'):
		self.yolo = Trailcam_YOLO(modelname)
		self.yolo.setDebug(lambda msg: print(f'debug: {msg}',file=sys.stderr))
		self.flist = []

	def preload(self,p):
		if os.path.isfile(p):
			self.flist.append(p)
		else:
			for root, _, files in os.walk(p):
				for f in files:
					if f.lower().endswith((".mp4", ".avi")):
						self.flist.append(os.path.join(root, f))

	def detect(self):
		for fn in self.flist:
			r = self.yolo.predict(fn, resized_fn='runs/_test.mp4')
			print(r['source'])
			for obj in r['detect']: print(obj)

if __name__ == '__main__':
	args = sys.argv
	fn =  sys.argv[1] if len(args) > 1 else ''
	if not (fn and os.path.exists(fn)):
		fn = input(f"Video Filename: ")

	if not fn:
		pass
	elif not os.path.exists(fn):
		print(f'file not found: {fn}', file=sys.stderr)
	else:
		app = MainApp('yolov8l')
		app.preload(fn)
		app.detect()
