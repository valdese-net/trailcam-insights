from lib.trailcam_yolo import *
from lib.trailcam_train import *
from lib.trailcam_transcode import *
from pathlib import Path
import time
import sys
import time
import os
import json
from datetime import date

def outDebug(msg):
	print(f'debug: {msg}',file=sys.stderr)

class TrailcamApp:
	def __init__(self, args):
		outDebug(f'initializing using {args}')
		self.modelname = args['model']
		self.results = []
		self.flist = []
		self.args = args
		self.debugMode = args['debug']
		# yolo, train, and transcode are dynamically created as needed
		self.actionObj = False

	def preload(self):
		pf = Path(self.args['inpath'])
		if pf.is_file():
			self.flist.append(pf)
			return True

		if not pf.is_dir(): return False

		for f in pf.rglob('*.avi'): self.flist.append(f)
		for f in pf.rglob('*.mp4'): self.flist.append(f)
		self.flist.sort()
		if self.debugMode: outDebug(f'preloaded: {self.flist}')
		return True

	def predictOn(self,fn):
		if not self.actionObj:
			self.actionObj = Trailcam_YOLO(self.modelname)
			if self.debugMode: self.actionObj.setDebug(outDebug)

		fn_data_fn = fn.parent / f'{fn.name}.{self.modelname}.json'
		d = ''
		if fn_data_fn.is_file():
			if self.debugMode: outDebug(f'loading prior detection for: {fn_data_fn}')
			with open(fn_data_fn, 'r') as o1: d = json.load(o1)
		else:
			last_modified = os.path.getmtime(fn)
			d = self.actionObj.predict(fn, False)
			with open(fn_data_fn, 'w') as o1: json.dump(d, o1)
			time.sleep(2) # briefly cool the cpu
		return d

	def trainOn(self,fn):
		if not self.actionObj:
			self.actionObj = TrailcamTrain(self.args['fps'])
		self.actionObj.train(fn)

	def transcodeOn(self,fn):
		if not self.actionObj:
			self.actionObj = TrailcamTranscode(self.args['fps'])
			if self.debugMode: self.actionObj.setDebug(outDebug)

		last_modified = os.path.getmtime(fn)
		mtime = time.localtime(last_modified)
		mtime_str = time.strftime('%Y%m%d_%H%M%S', mtime)
		transcodeFN = fn.with_suffix('.640x.mp4')
		if self.args['outpath']: transcodeFN = Path(self.args['outpath']).joinpath(f'{mtime_str}.mp4')
		if not transcodeFN.is_file():
			self.actionObj.transcode(fn,transcodeFN)
			os.utime(transcodeFN,(last_modified,last_modified))

	def runOn(self,fn):
		if self.args['action'] == 'train':
			self.trainOn(fn)
		elif self.args['action'] == 'transcode':
			self.transcodeOn(fn)
		elif self.args['action'] == 'detect':
			print(f'predict: {fn}',file=sys.stderr)
			r = self.predictOn(fn)
			print(r['source'])
			for obj in r['detect']: print(obj)
			self.results.append(r)
		else:
			outDebug(f'skipped {fn}')

	def execute(self):
		for fn in self.flist:
			self.runOn(fn)
