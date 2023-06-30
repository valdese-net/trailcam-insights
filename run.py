import argparse
import os
from lib.trailcam_insights import *

ap = argparse.ArgumentParser(description='Use YOLOv8 on trailcam videos to gather insights on trail utilization')
ap.add_argument('pathname', help='path to a specific video or directory containing trailcam videos')
ap.add_argument('-m', '--model', required=False, default='yolov8l',help='specify the model (yolov8n,yolov8m,yolov8l,yolov8x) used by YOLOv8')
ap.add_argument('-d', '--debug', required=False, action='store_true', default=False, help='turn on info and debug messages')

args = vars(ap.parse_args())

fn =  args['pathname']

if not fn:
	print('no file specified', file=sys.stderr)
elif not os.path.exists(fn):
	print(f'file not found: {fn}', file=sys.stderr)
else:
	app = TrailcamInsights(args['model'])
	if args['debug']: app.showDebug()
	app.preload(fn)
	app.detect()
