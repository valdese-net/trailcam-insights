from lib.videoscanner import VideoScanner

scanner = VideoScanner()

fn = input("Video Path: ")
stats = scanner.scan(fn)
print('scan complete')
print(stats)
