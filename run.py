import sys
import argparse
from lib.trailcam_app import *

ap = argparse.ArgumentParser(description='Use YOLOv8 on trailcam videos to gather insights on trail utilization')
ap.add_argument('action', choices=('detect','train','transcode'), help='action to perform')
ap.add_argument('inpath', help='path to a specific video or directory containing trailcam videos')
ap.add_argument('-m', '--model', required=False, default='yolov8l',help='specify the model (yolov8n,yolov8m,yolov8l,yolov8x) used by YOLOv8')
ap.add_argument('--fps', required=False, default=-1,type=int,help='specify a new framerate for processing')
ap.add_argument('-d', '--debug', required=False, action='store_true', default=False, help='turn on info and debug messages')
ap.add_argument('-o','--outpath', required=False, default=False, help='path to a specific directory that should be used for output')

args = vars(ap.parse_args())

app = TrailcamApp(args)
if not app.preload():
	sys.exit(f'path not found')
app.execute()
