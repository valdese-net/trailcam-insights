from lib.trailcam_findmax import *

scanner = TrailcamFindMax()

scanner.setDebug(print)

fn = input("Video Path: ")
stats = scanner.scan(fn)
print(stats)
