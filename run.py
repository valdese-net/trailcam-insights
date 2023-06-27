from lib.trailcam_findmax import *
from lib.trailcam_tracker import *
import sys
import os.path

def main(fn):
	#scanner = TrailcamFindMax()
	scanner = TrailcamTracker()

	scanner.setDebug(lambda msg: print('debug:',msg))

	stats = scanner.scan(fn)

	print(stats)

if __name__ == '__main__':
	args = sys.argv
	fn =  sys.argv[1] if len(args) > 1 else ''
	while not (fn and os.path.isfile(fn)):
		fn = input(f"Video Filename: [{fn}]")
	main(fn)
