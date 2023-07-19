import os
import sys
import cv2
import imutils
from pathlib import Path

class TrailcamTranscode:
	def __init__(self,fps:int=5):
		self.fps = fps
		self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		self.debugF = False

	def setDebug(self,f):
		self.debugF = f

	def transcode(self,src,dst):
		print(f'transcode({src},{dst})',file=sys.stderr)
		src = str(src)
		vc = cv2.VideoCapture(src)
		src_fps = vc.get(cv2.CAP_PROP_FPS)
		dst_fps = self.fps if ((self.fps > 0) and (src_fps > self.fps)) else src_fps

		src_sz = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		dst_sz = (640,(src_sz[1]*640)//src_sz[0])

		writer = cv2.VideoWriter(str(dst), self.fourcc, dst_fps, dst_sz)
		unused_frame = False
		src_framenum = dst_framenum = 0

		while True:
			success, image = vc.read()
			if not success: break
			src_framenum += 1
			if (self.fps > 0):
				# if a new fps was requested, skip frames that are too early
				src_tick = float(src_framenum)/float(src_fps)
				dst_tick = float(dst_framenum+1)/float(dst_fps)
				if self.debugF: self.debugF(f'frame {src_framenum} comparing ticks {dst_tick} > {src_tick}')
				if dst_tick > src_tick:
					if self.debugF: self.debugF(f'skipping frame {src_framenum}')
					unused_frame = image
					continue

			# Resizing the Image
			# After testing imutils and cv2, the imutils version looks better, likely due to anti-aliasing
			#image = cv2.resize(image,newsz)
			image = imutils.resize(image,width=dst_sz[0])
			writer.write(image)
			dst_framenum += 1
			unused_frame = False

		if not unused_frame is False:
			# always write the final frame
			if self.debugF: self.debugF('writing final frame (previously skipped)')
			unused_frame = imutils.resize(unused_frame,width=dst_sz[0])
			writer.write(unused_frame)
			unused_frame = False

		writer.release()
		vc.release()
