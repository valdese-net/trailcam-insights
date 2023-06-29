from lib.trailcam_yolo import *
import argparse
import sys
import os
import json

class MainApp:
	def __init__(self, modelname='yolov8x'):
		self.modelname = modelname
		self.yolo = False
		self.flist = []

	def preload(self,p):
		if os.path.isfile(p):
			self.flist.append(p)
		else:
			for root, _, files in os.walk(p):
				for f in files:
					if f.lower().endswith((".mp4", ".avi")):
						self.flist.append(os.path.join(root, f))

	def predictOn(self,fn):
		fn_data_fn = f'{fn}.{self.modelname}.json'
		d = ''
		if os.path.isfile(fn_data_fn):
			with open(fn_data_fn, 'r') as o1: d = json.load(o1)
		else:
			if not self.yolo:
				self.yolo = Trailcam_YOLO(self.modelname)
				self.yolo.setDebug(lambda msg: print(f'debug: {msg}',file=sys.stderr))

			d = self.yolo.predict(fn, resized_fn=False)
			with open(fn_data_fn, 'w') as o1: json.dump(d, o1)
		return d
			
	def detect(self):
		results = []
		for fn in self.flist:
			r = self.predictOn(fn)
			print(r['source'])
			for obj in r['detect']: print(obj)
			results.append(r)

if __name__ == '__main__':
	ap = argparse.ArgumentParser(
		prog='TrailCamInsights',
		description='Uses YOLO to gether insights on the contents of trailcam video footage'
	)
	ap.add_argument('pathname',help="path to a specific video or directory containing trailcam videos")
	ap.add_argument('-m', '--model', required=False, default='yolov8l',help="model used by YOLOv8")
	args = vars(ap.parse_args())
	print(args,file=sys.stderr)

	fn =  args['pathname']

	if not fn:
		print('no file specified', file=sys.stderr)
	elif not os.path.exists(fn):
		print(f'file not found: {fn}', file=sys.stderr)
	else:
		app = MainApp(args['model'])
		app.preload(fn)
		app.detect()
