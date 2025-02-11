# Newton - A Small Benchmark for Interactive Foundation World Models

This is the repo for reproducing the `Newton` dataset. To do so, you'll need to install some dependencies:

```
bpy
opencv-python
scipy
numpy
```

## Reproduction 

After that, `physics.py` and `object_permanence.py` contain the two current tasks. You can use `launch.sh` to speed them up - they'll begin dumping videos in the `output/` folder. You can then use `hf.py` to upload to HuggingFace, or use the data yourself. 

## Evaluation

Run `eval.py {folder1} {folder2}` to compare two sets of images with MSE, LPIPS and Pose Estimation metrics.

## Miscellaneous
### Pose Estimation

Part of evaluating a model on the benchmark is pose estimation of 3D objects. For now, this is done with CV2 methods, but eventually we'll need to upgrade to a fully neural backbone such as FoundationPose or YOLOv11. You can try it at `predict.py`.

### Pretrained models

Estimated release date: as soon as they're good
