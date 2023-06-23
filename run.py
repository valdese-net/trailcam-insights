from lib.trailcam_findmax import *

scanner = TrailcamFindMax()

scanner.setDebug(lambda msg: print('debug:',msg))

fn = input("Video Path: ")
stats = scanner.scan(fn)
print(stats)
