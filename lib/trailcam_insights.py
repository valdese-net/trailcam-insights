from lib.trailcam_yolo import *
import os
import sys
import json

class TrailcamInsights:
	def __init__(self, modelname='yolov8x'):
		self.modelname = modelname
		self.yolo = Trailcam_YOLO(self.modelname)
		self.flist = []

	def showDebug(self):
		self.yolo.setDebug(lambda msg: print(f'debug: {msg}',file=sys.stderr))

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
			d = self.yolo.predict(fn, False)
			with open(fn_data_fn, 'w') as o1: json.dump(d, o1)
		return d
			
	def detect(self):
		results = []
		for fn in self.flist:
			r = self.predictOn(fn)
			print(r['source'])
			for obj in r['detect']: print(obj)
			results.append(r)
