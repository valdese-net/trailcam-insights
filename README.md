# Introduction

The trail camera output currently collected from Valdese Lakeside Park
shares the following common traits:

- video is activated after motion is detected
- a short clip is recorded
- many trees and other natural obstructions are contained in the footage
- video often contains a large pixel range, but with poor quality and motion blur

The goal is to automate the processing of video clips produced by the trail
cameras in order to gain insights into the park eco system and its utilization.

This tool is invoked from the command line console. For help, start with:

`python run.py -h`

## Design Decisions

This tools currently uses Ultralytics YOLOv8 as the video processing tool.
This system contains built-in supoprt for `predict` and `track` methods
on standard, as well as customized, models. Details can be found at:

<https://docs.ultralytics.com/>

While experimenting with Detection and Instance Segmentation Models, using
both `predict` and `track` model methods, it is clear that this effort consumes
significant processing time and power. The tool creates a `json` capture of
`predict` results, which allows a directory of videos to be stopped and resumed
later.

Early results indicate that a custom model might be required to get good
results on our trailcam videos. The current processing using large model
is slow, and not very accurate.

### Custom Training

Might try training YOLO using actual tralcam data that has previously been captured,
as explained here:

<https://www.youtube.com/watch?v=m9fH9OWn8YM>

Possible tools for data annotation:

- <https://github.com/tunahansalih/yolo-annotation-tool>
- <https://github.com/opencv/cvat>

## Python Environment

Python 3.9 is required. I tried to use Python 3.11, but encountered problems
with the `lap` module. Using a virtual environment for Python is advised.

Create the env:
> python -m venv d:/python/yolov8

Use the env:
> source d:/python/yolov8/Scripts/activate
> d:/python/yolov8/Scripts/activate.bat

Test yolo predict:
> yolo predict model=cache/yolov8s.pt source=/path/to/video

Invoke the trailcam scanner:
> python run.py /path/to/video/folder
