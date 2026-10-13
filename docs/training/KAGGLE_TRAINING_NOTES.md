# Kaggle Training Notes

This file contains snippets and notes for when you move to Kaggle or Google Colab for the heavy cloud training.

## Dataset Download Snippet (Roboflow)
Use this snippet in your Kaggle Notebook to download the real dataset directly into the cloud environment:

```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="__________________")
project = rf.workspace("daffodil-international-university-s5vpr").project("doctors-prescriptions-handwriting")
version = project.version(1)
dataset = version.download("yolov8")
```
