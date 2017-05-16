# EasyPR-python
EasyPR-python by fh

slow, some mistake
## Requirements
* python 3.5
* opencv 3.1
* tensorflow 1.0

## Introduction
Use easypr-1.4's way to do locate and segment.

Use Lenet to do chars identify.

Use a cnn network to judge whether a car.

## Compare
### easypr-1.4
![performance](pic/easypr_1.4.png)
### easy-python
![cnn_performance](pic/cnn_res_loss.png)
![cnn_performance](pic/cnn_res_acc.png)
## TODO
...

## Done
- [x] plate locate ( need test )
- [x] plate judge
- [x] plate judge (cnn 98.79%)
- [x] plate segment
- [x] char identify ( cnn 99.63%)

## How to train identify
unzip `resources/train_data/chars.7z`

`python demo.py`

(There is already a trained checkpoint in `train/model/chars/models`)

## How to train judge
similar with identify

## Demo
`python demo.py`