# mmpose_PD_interpolation
Interpolation of [mmpose](https://github.com/open-mmlab/mmpose) facial keypoint extraction for Parkinson's disease-relevant regions for video action recognition and classification, in order to detect presence of Parkinson's disease.

Please follow mmpose installation instructions, and place `zitong_crop.py` into `mmpose/demo/` directory, and replace `mmpose/demo/top_down_video_demo_with_mmdet.py` and `mmpose/mmpose/core/visualization/image.py`. You can center and crop around the faces with 
```
python demo/zitong_crop.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    --video-path /home/swli2/mmpose/vis_results/video76_256.mp4 \
    --out-video-root vis_results
``` 
Then, you can run the interpolation on the face-centered video using `python demo/top_down_video_demo_with_mmdet.py ...` using the same arguments as above. Remember to update to the correct path and file name.

In demo_results, you will see the off-the-shelf keypoint extraction performance on a video with a Parkinson's disease-positive subject from our novel dataset. The mapping of the original keypoints is provided as well. This original set of 68 keypoints is used to interpolate 9 Parkinson's disease informative regions. Both the interpolated keypoints alone, as well as the keypoints color coded with bounding boxes, are provided. The interpolation on a second subject using the same interpolation code is shown as well, to demonstrate the rigor of the model and interpolation on Parkinson's disease datasets.

This interpolation is an intermediate step, to define the regions that are fed into the head of a new classifier.

Reference: 
```
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
