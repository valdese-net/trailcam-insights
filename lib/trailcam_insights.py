from lib.trailcam_yolo import *
from pathlib import Path
import sys
import time
import json

class TrailcamInsights:
	def __init__(self, modelname='yolov8x'):
		self.modelname = modelname
		self.yolo = Trailcam_YOLO(self.modelname)
		self.flist = []
		self.debugMode = False

	def showDebug(self):
		self.debugMode = True
		self.yolo.setDebug(lambda msg: print(f'debug: {msg}',file=sys.stderr))

	def preload(self,p):
		pf = Path(p)
		if pf.is_file():
			self.flist.append(pf)
			return True

		if not pf.is_dir(): return False

		for f in pf.rglob('*.avi'): self.flist.append(f)
		for f in pf.rglob('*.mp4'): self.flist.append(f)
		self.flist.sort()
		if self.debugMode: print(f'preloaded: {self.flist}',file=sys.stderr)
		return True

	def predictOn(self,fn):
		fn_data_fn = fn.parent / f'{fn.name}.{self.modelname}.json'
		d = ''
		if fn_data_fn.is_file():
			if self.debugMode: print(f'loading prior detection for: {fn_data_fn}',file=sys.stderr)
			with open(fn_data_fn, 'r') as o1: d = json.load(o1)
		else:
			d = self.yolo.predict(fn, False)
			with open(fn_data_fn, 'w') as o1: json.dump(d, o1)
			time.sleep(2) # briefly cool the cpu
		return d
			
	def detect(self):
		results = []
		for fn in self.flist:
			print(f'predict: {fn}',file=sys.stderr)
			r = self.predictOn(fn)
			print(r['source'])
			for obj in r['detect']: print(obj)
			results.append(r)
