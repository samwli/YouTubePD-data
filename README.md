# mmpose_PD_interpolation
Interpolation of [mmpose] (https://github.com/open-mmlab/mmpose) facial keypoint extraction for Parkinson's disease-relevant regions for video action recognition and classification, in order to detect presence of Parkinson's disease.

Please place `zitong_crop.py` into 'mmpose/demo/' directory, and replace 'mmpose/demo/top_down_video_demo_with_mmdet.py' and 'mmpose/mmpose/core/visualization/image.py'. You can run demo with 
```
python demo/top_down_video_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    --video-path /home/swli2/mmpose/vis_results/crop_video76_256.mp4 \
    --out-video-root vis_results
``` 
```
Reference: @misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
