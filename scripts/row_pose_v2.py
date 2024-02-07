# Copyright (c) OpenMMLab. All rights reserved.
import time
from argparse import ArgumentParser
from mmdet.evaluation.functional import bbox_overlaps
import cv2
import json_tricks as json
import mmcv
from mmengine.utils import track_iter_progress
import numpy as np
from mmcv.transforms import Compose
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
from pose_estimation import HeadPoseEstimator

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from tracker import BYTETracker

def in_bbox(rect,pt):
    logic = rect[0] < pt[0] < rect[2] and rect[1] < pt[1] < rect[3]
    return logic

def parse_args():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser(description='MMDetection video demo')
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('video', help='Video file')
    parser.add_argument(
        "--start_second", 
        type=float, 
        default=0.0, 
        help="ts of video start")
    parser.add_argument(
        '--skip_seconds', 
        type=float, 
        default=1.0, 
        help='number of seconds between frames to process')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--wait_time',
        type=int,
        default=1,
        help='The interval of show (s), 0 is block')
    parser.add_argument(
        '--out_video', 
        type=str, 
        help='Output video file')
    parser.add_argument(
        '--out_preds',
        type=str,
        help='whether to save predicted results')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    # tracking args
    parser.add_argument(
        "--track_thresh", 
        type=float, 
        default=0.3, 
        help="tracking confidence threshold")
    parser.add_argument(
        "--track_buffer", 
        type=int, 
        default=30, 
        help="the frames for keep lost tracks")
    parser.add_argument(
        "--match_thresh", 
        type=float, 
        default=0.8, 
        help="matching threshold for tracking")
    parser.add_argument(
        '--min_overlap', 
        type=float, 
        default=0.3, 
        help='min overlap with seat to count as sitting')
    parser.add_argument(
        '--start_overlap', 
        type=float, 
        default=0.5, 
        help='min overlap to start tracking once person is in seat')
    #vis parameters
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--alpha', 
        type=float, 
        default=0.8, 
        help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', 
        action='store_true', 
        help='Draw bboxes of instances')
    
    args = parser.parse_args()
    return args

def main():
    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parse_args()

    assert args.out_video or args.show or args.out_preds, \
        ('Please specify at least one operation with the argument "--out_video" or "--show" or "--out_preds"')

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device='cuda:0')
    # build test pipeline
    detector.cfg.test_dataloader.dataset.pipeline[
        0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(detector.cfg.test_dataloader.dataset.pipeline)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device='cuda:0',
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))
    
    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
    hide_kpts = np.append(np.arange(0,5),np.arange(11,23))

    #seat detect
    with open(f"{args.video}.json",'r') as f:
        bbox_seat = {a['label']:np.array(a['points']).flatten() for a in json.load(f)["shapes"]}
        seat_labels = sorted(bbox_seat)
        bbox_seat = np.stack([bbox_seat[l] for l in seat_labels],axis=0)

    #video reader
    video_reader = mmcv.VideoReader(args.video)
    if args.out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out_video, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))
    if args.out_preds:
        pred_instances_list = []
    frame_nums = list(np.arange(int(args.start_second*video_reader.fps), len(video_reader),int(args.skip_seconds*video_reader.fps)))

    # Setup a pose estimator to solve pose.
    head_pose_estimator = HeadPoseEstimator(video_reader.width, video_reader.height)

    #tracker
    tracker = BYTETracker(args, frame_rate=video_reader.fps)
    seat_id = []
    for i in track_iter_progress(frame_nums):
        frame = video_reader.get_frame(i)

        # predict bbox
        result = inference_detector(detector, frame, test_pipeline=test_pipeline)
        preds = result.pred_instances
        labels = np.array(preds.labels.squeeze().cpu())
        preds = preds[labels==0] #filter only person
        
        #track
        if len(preds) != 0:
            online_targets = tracker.update(preds)
            preds = preds[:len(online_targets)]
            if len(online_targets) != 0:
                online_boxes = []
                online_ids = []
                online_scores = []
                online_masks = []
                for t in online_targets:
                    online_boxes.append(t.tlbr)
                    online_ids.append(t.track_id)
                    online_scores.append(t.score)
                    online_masks.append(t.mask)
                preds.scores = np.array(online_scores)
                preds.bboxes = np.row_stack(online_boxes)
                preds.masks = np.stack(online_masks,axis=0)

                #determine which seat
                online_ids = np.array(online_ids)
                overlaps = bbox_overlaps(preds.bboxes,bbox_seat)
                keep = (np.any(overlaps>args.start_overlap,axis=1) | np.array([id in seat_id for id in online_ids])) \
                        & np.any(overlaps>args.min_overlap,axis=1)  
                seat_id = np.unique(np.append(seat_id, online_ids[keep]))
                seat_label = np.argmax(overlaps,axis=1)
                preds = preds[keep]
                preds.labels = seat_label[keep]
                    
            # predict keypoints
            preds = preds[preds.scores>args.bbox_thr]
            if len(preds) != 0:
                pose_results = inference_topdown(pose_estimator, mmcv.bgr2rgb(frame), preds.bboxes)
                pred_instances = merge_data_samples(pose_results)
                
                #head pose
                for pred in pose_results:
                    marks = pred.pred_instances.keypoints[0,23:91,:]
                    head_pose = head_pose_estimator.solve(marks)
                    head_pose_estimator.draw_axes(frame, head_pose)
            else:
                pred_instances = []

            #visualize
            visualizer.add_datasample(
                name=args.video,
                image=frame,
                data_sample=pred_instances,
                draw_gt=False,
                draw_heatmap=args.draw_heatmap,
                draw_bbox=args.draw_bbox,
                show_kpt_idx=args.show_kpt_idx,
                hide_kpts=hide_kpts,
                skeleton_style=args.skeleton_style,
                kpt_thr=args.kpt_thr)
            frame = visualizer.get_image()

            #save
            if args.out_preds:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=i,
                        instances=split_instances(pred_instances.get('pred_instances', None))))
            if args.out_video:
                video_writer.write(frame)
            if args.show:
                mmcv.imshow(frame, args.video, args.wait_time)


    if video_writer:
        video_writer.release()

    cv2.destroyAllWindows()

    if args.out_preds:
        with open(args.out_preds, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')


if __name__ == '__main__':
    main()
