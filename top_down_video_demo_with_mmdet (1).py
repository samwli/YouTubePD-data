# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the pose'
        'estimation stage. Default: False.')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    print('Initializing model...')
    # build the detection model from a config file and a checkpoint file
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # read video
    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = video.fps
        size = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # frame index offsets for inference, used in multi-frame inference setting
    if args.use_multi_frames:
        assert 'frame_indices_test' in pose_model.cfg.data.test.data_cfg
        indices = pose_model.cfg.data.test.data_cfg['frame_indices_test']

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    print('Running inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        
        
        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, cur_frame)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        if args.use_multi_frames:
            frames = collect_multi_frames(video, frame_id, indices,
                                          args.online)
        
        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            frames if args.use_multi_frames else cur_frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
       
        def interpolate(p1, p2, t):
            x = p2[0]+t*(p2[0]-p1[0])
            y = p2[1]+t*(p2[1]-p1[1])
            return [x, y, 0.9]
        #23 to 91 for facial keypoint
        #init array contains 68 facial keypoints, for interpolation
        init_arr = pose_results[0]['keypoints'][23:91]
        #box array contains (x1, y1, x2, y2) bbox
        box_arr = pose_results[0]['bbox']
        #interpolate new keypoint_arr
        keypoint_arr = [[]]
        #interpolate 4 keypoints for forehead region
        keypoint_arr.append(interpolate(init_arr[36], init_arr[18], 1))
        keypoint_arr.append(interpolate(init_arr[36], init_arr[18], 3))
        keypoint_arr.append(interpolate(init_arr[45], init_arr[25], 3))
        keypoint_arr.append(interpolate(init_arr[45], init_arr[25], 1))
        #interpolate 8 keypoints for nasolabial folds region
        keypoint_arr.append(interpolate(init_arr[1], init_arr[29], -3/11))
        keypoint_arr.append(interpolate(init_arr[1], init_arr[29], -1/11))
        keypoint_arr.append(interpolate(init_arr[4], init_arr[48], -1/8))
        keypoint_arr.append(interpolate(init_arr[4], init_arr[48], -4/8))
        keypoint_arr.append(interpolate(init_arr[15], init_arr[29], -1/11))
        keypoint_arr.append(interpolate(init_arr[15], init_arr[29], -3/11))
        keypoint_arr.append(interpolate(init_arr[12], init_arr[54], -4/8))  
        keypoint_arr.append(interpolate(init_arr[12], init_arr[54], -1/8))
        #interpolate 8 keypoints for marionette lines region
        keypoint_arr.append(interpolate(init_arr[9], init_arr[55], -1/5))
        keypoint_arr.append(interpolate(init_arr[9], init_arr[55], -4/5))
        keypoint_arr.append(interpolate(init_arr[35], init_arr[11], -2/13))
        keypoint_arr.append(interpolate(init_arr[35], init_arr[11], -8/13))
        keypoint_arr.append(interpolate(init_arr[7], init_arr[59], -1/5))
        keypoint_arr.append(interpolate(init_arr[7], init_arr[59], -4/5))
        keypoint_arr.append(interpolate(init_arr[31], init_arr[5], -2/13))
        keypoint_arr.append(interpolate(init_arr[31], init_arr[5], -8/13))
        #interpolate 8 keypoints for crows feet region
        keypoint_arr.append(interpolate(init_arr[18], init_arr[36], -1/2))
        keypoint_arr.append(interpolate(init_arr[18], init_arr[36], 5/6))
        keypoint_arr.append(interpolate(init_arr[39], init_arr[1], -3/16))
        keypoint_arr.append(interpolate(init_arr[17], init_arr[0], -1/4))
        keypoint_arr.append(interpolate(init_arr[25], init_arr[45], -1/2))
        keypoint_arr.append(interpolate(init_arr[25], init_arr[45], 5/6))
        keypoint_arr.append(interpolate(init_arr[42], init_arr[15], -3/16))
        keypoint_arr.append(interpolate(init_arr[26], init_arr[16], -1/4))
        #interpolate 4 keypoints for transverse nasal lines region
        keypoint_arr.append(interpolate(init_arr[27], init_arr[39], -1/4))
        keypoint_arr.append(interpolate(init_arr[27], init_arr[42], -1/4))
        keypoint_arr.append(interpolate(init_arr[19], init_arr[24], -2/8))
        keypoint_arr.append(interpolate(init_arr[19], init_arr[24], -6/8))
        #interpolate 5 keypoints for vertical glabellar lines region
        keypoint_arr.append(interpolate(init_arr[33], init_arr[27], 13/16))
        keypoint_arr.append(interpolate(init_arr[35], init_arr[22], 1/3))
        keypoint_arr.append(interpolate(init_arr[42], init_arr[22], 0/8))
        keypoint_arr.append(interpolate(init_arr[39], init_arr[21], 0/8))
        keypoint_arr.append(interpolate(init_arr[31], init_arr[21], 1/3))
        #interpolate 8 keypoints for eyelids region
        keypoint_arr.append(interpolate(init_arr[25], init_arr[42], 1/2))
        keypoint_arr.append(interpolate(init_arr[12], init_arr[26], -1/5))
        keypoint_arr.append(interpolate(init_arr[22], init_arr[26], 0/4))
        keypoint_arr.append(interpolate(init_arr[42], init_arr[22], 0))
        keypoint_arr.append(interpolate(init_arr[18], init_arr[39], 1/2))
        keypoint_arr.append(interpolate(init_arr[4], init_arr[17], -1/5))
        keypoint_arr.append(interpolate(init_arr[21], init_arr[17], 0/4))
        keypoint_arr.append(interpolate(init_arr[39], init_arr[21], 0))
        #interpolate 5 keypoints for mouth region
        keypoint_arr.append(interpolate(init_arr[51], init_arr[57], 1))
        keypoint_arr.append(interpolate(init_arr[10], init_arr[54], -1/3))
        keypoint_arr.append(interpolate(init_arr[12], init_arr[33], -1/2))
        keypoint_arr.append(interpolate(init_arr[4], init_arr[33], -1/2))
        keypoint_arr.append(interpolate(init_arr[6], init_arr[48], -1/3))
        #interpolate 8 keypoints for eyebrows region
        keypoint_arr.append(interpolate(init_arr[16], init_arr[42], 1/6))
        keypoint_arr.append(interpolate(init_arr[12], init_arr[26], -1/8))
        keypoint_arr.append(interpolate(init_arr[54], init_arr[26], 1/6))
        keypoint_arr.append(interpolate(init_arr[35], init_arr[22], 1/4))
        keypoint_arr.append(interpolate(init_arr[0], init_arr[39], 1/6))
        keypoint_arr.append(interpolate(init_arr[4], init_arr[17], -1/8))
        keypoint_arr.append(interpolate(init_arr[48], init_arr[17], 1/6))
        keypoint_arr.append(interpolate(init_arr[31], init_arr[21], 1/4))
        
        pose_results[0]['keypoints'] = keypoint_arr[1:]
        pose_results = [pose_results[0]]
        # show the results
        vis_frame = vis_pose_result(
            pose_model,
            cur_frame,
            pose_results,
            dataset=dataset,
            dataset_info=None,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=False)

        if args.show:
            cv2.imshow('Frame', vis_frame)

        if save_out_video:
            videoWriter.write(vis_frame)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
