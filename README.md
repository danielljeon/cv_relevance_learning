# cv_relevance_learning

![black_formatter](https://github.com/danielljeon/cv_relevance_learning/actions/workflows/black_formatter.yaml/badge.svg)

Online-learning relevance classifier built on pre-trained computer vision for
machine learning university course.

---

<details markdown="1">
  <summary>Table of Contents</summary>

<!-- TOC -->
* [cv_relevance_learning](#cv_relevance_learning)
  * [1 Overview](#1-overview)
  * [2 Setup](#2-setup)
    * [2.1 Install Python (pip) Packages](#21-install-python-pip-packages)
    * [2.2 Download YOLO ONNX](#22-download-yolo-onnx)
    * [2.3 Running the Code](#23-running-the-code)
  * [3 Dev Notes](#3-dev-notes)
<!-- TOC -->

</details>

---

## 1 Overview

---

## 2 Setup

- Assumes python environment is already setup (Python `3.10` or newer).

### 2.1 Install Python (pip) Packages

- Install packages your preferred way. If using pip, use the generic command
  below:

```shell
pip install -r requirements.txt
```

### 2.2 Download YOLO ONNX

Download the YOLOv5 ONNX file in the root directory:

```shell
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.onnx
```

### 2.3 Running the Code

The main code can be run right off of `main.py`:

```shell
python main.py
```

---

## 3 Dev Notes
