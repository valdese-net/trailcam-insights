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

After some experimentation, I decided to use Ultralytics YOLOv8 as the video
processing tool. This system contains built-in supoprt for `predict` and `track`
methods on standard, as well as customized, models. Details can be found at:

<https://docs.ultralytics.com/>

While experimenting with Detection and Instance Segmentation Models, using
both `predict` and `track` model methods, it is clear that this effort consumes
significant processing time and power. In classic Unix fashion, I think it is
best to break the analysis process into a series of smaller tasks. The first
task should be to run `predict` on the video data, collecting and capturing the
results for further processing in later tasks.

## Python Environment

Python 3.9 is required. I tried to use Python 3.11, but encountered problems
with the `lap` module. Using a virtual environment for Python is advised.

Create the env:
> python -m venv d:/python/yolov8

Use the env:
> source d:/python/yolov8/Scripts/activate
> d:/python/yolov8/Scripts/activate.bat

Test yolo predict:
> yolo predict model=cache/yolov8l.pt source=/path/to/video

Invoke the trailcam scanner:
> python run.py /path/to/video/folder > /path/to/output/file.csv
