# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import math
from pathlib import Path
import os
import numpy as np
import json_tricks as json
import yaml
from argparse import ArgumentParser
from box import Box
import pandas as pd

import mmcv
from mmcv.transforms import Compose
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.utils import adapt_mmdet_pipeline
from mmdet.evaluation.functional import bbox_overlaps as bbox_iou
from mmengine.utils import track_iter_progress

from utils import *
from pose_estimation import HeadPoseEstimator
import ransac.core as ransac
from ransac.models.conic_section import ConicSection

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from tracker import BYTETracker


def parse_args():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser(description='MMDetection video demo')
    parser.add_argument('config', help='Config file')
    parser.add_argument('video', help='Video file')
    parser.add_argument(
        "--start-second", 
        type=float, 
        default=0.0, 
        help="ts of video start")
    parser.add_argument(
        "--end-second", 
        type=float, 
        default=0.0, 
        help="ts of video start")
    parser.add_argument(
        '--skip-seconds', 
        type=float, 
        default=1.0, 
        help='number of seconds between frames to process')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-video', 
        type=str, 
        help='Output video file')
    parser.add_argument(
        '--out-preds',
        type=str,
        help='Output predicted results')
    parser.add_argument(
        '--wait-time',
        type=int,
        default=1,
        help='The interval of show (s), 0 is block')
    parser.add_argument(
        '--draw-pose', 
        action='store_true', 
        default=False,
        help='Draw keypoints of instances')

    args = parser.parse_args()
    return args

def main():
    assert has_mmdet, 'Please install mmdet to run the demo.'
    #read configs
    args = parse_args()
    assert args.out_video or args.show or args.out_preds, \
        ('Please specify at least one operation with the argument "--out_video" or "--show" or "--out_preds"')
    with open(args.config) as stream:
        try:
            config = Box(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    video_file = os.path.join(config.inputs.video_dir, args.video)

    # build detector
    detector = init_detector(
        config.models.det_config, config.models.det_checkpoint, device='cuda:0')
    # build test pipeline
    detector.cfg.test_dataloader.dataset.pipeline[
        0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(detector.cfg.test_dataloader.dataset.pipeline)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        config.models.pose_config,
        config.models.pose_checkpoint,
        device='cuda:0')
    
    # build visualizer
    if args.draw_pose:
        pose_estimator.cfg.visualizer.radius = config.vis.radius
        pose_estimator.cfg.visualizer.alpha = config.vis.alpha
        pose_estimator.cfg.visualizer.line_width = config.vis.thickness
        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        visualizer.set_dataset_meta(
            pose_estimator.dataset_meta, skeleton_style=config.vis.skeleton_style)
        hide_kpts = np.arange(0,23)

    #load annotations
    video_name = Path(video_file).stem
    annotations = json.load(open(f"{config.inputs.calib_dir}/images/{video_name}.json",'r'))
    img_shape = (annotations['imageHeight'],annotations['imageWidth'])
    bbox_seat = {}
    mask_seat = {}
    mask_seat_back =  []
    bbox_seat_back = []
    for ann in annotations["shapes"]:
        mask = shape_to_mask(img_shape, ann['points'])
        bbox = mask_to_box(mask)
        if ann['label'] in ['aisle', 'middle', 'window']: 
            bbox_seat[ann['label']] = bbox
            mask_seat[ann['label']] = mask
        elif ann['label'] == 'shutter':
            bbox_wind = bbox
            mask_wind = mask
        elif ann['label'] == 'seat':
            bbox_seat_back.append(bbox)
            mask_seat_back.append(mask)
      
    seat_labels = sorted(bbox_seat)
    bbox_seat = np.stack([bbox_seat[l] for l in seat_labels],axis=0)
    mask_seat = np.stack([mask_seat[l] for l in seat_labels],axis=0)
    #ref_image = img_b64_to_arr(annotations['imageData'])

    #video reader
    video_reader = mmcv.VideoReader(video_file)
    if args.end_second == 0:
        end_second = len(video_reader)
    else:
        end_second = int(args.end_second*video_reader.fps)
    skip_seconds = int(args.skip_seconds*video_reader.fps)
    if args.skip_seconds == 0:
        skip_seconds = 1        
    frame_nums = list(np.arange(int(args.start_second*video_reader.fps),end_second,skip_seconds))

    #save results
    frame_results_list = []
    if args.out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out_video, fourcc, int(video_reader.fps/skip_seconds),
            (video_reader.width, video_reader.height))

    # Setup a pose estimator to solve pose.
    cam_m = np.loadtxt(f"{config.inputs.calib_dir}/calibration/{video_name}_M.txt")
    cam_d = np.loadtxt(f"{config.inputs.calib_dir}/calibration/{video_name}_D.txt")
    head_pose_estimator = HeadPoseEstimator(cam_m,cam_d)
    dir_lim = config.face.direction

    #set up ellipse finder
    ellipse_estimator = ransac.Modeler(ConicSection, number_of_trials=config.face.ransac_trials, acceptable_error=config.face.ellipse_error)
    dilation_kernel = np.ones((config.face.face_dilation, config.face.face_dilation), np.uint8) 

    #tracker
    tracker = BYTETracker(config.tracking, frame_rate=video_reader.fps)
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
                mask_overlaps = mask_iou(preds.masks,mask_seat)
                keep = (np.any(mask_overlaps>config.tracking.start_sit_thr,axis=1) | np.array([id in seat_id for id in online_ids])) \
                        & np.any(mask_overlaps>config.tracking.track_sit_thr,axis=1) \
                        & ((preds.bboxes[:,2]-preds.bboxes[:,0])<(config.tracking.max_size*(bbox_seat[:,2]-bbox_seat[:,0]).mean()))
                
                bbox_overlaps = bbox_iou(preds.bboxes,bbox_seat)
                seat_scores  = np.argsort(-np.max(bbox_overlaps[keep],axis=1))
                seat_label = np.argmax(bbox_overlaps[keep],axis=1)[seat_scores]
                _, keep_labels = np.unique(seat_label,return_index=True)

                seat_id = np.unique(np.append(seat_id, online_ids[keep][seat_scores][keep_labels]))
                preds = preds[keep][seat_scores][keep_labels]
                preds.labels = seat_label[keep_labels]
                    
            # predict keypoints
            frame_results = {'frame_id':i}
            vis_boxes = []
            vis_labels = []
            vis_names = []
            if len(preds) != 0:
                pose_results = inference_topdown(pose_estimator, mmcv.bgr2rgb(frame), preds.bboxes)
                for n,pred in enumerate(pose_results):
                    vis_labels.append(len(vis_names))
                    vis_boxes.append(preds.bboxes[n])
                    seat_name = seat_labels[preds.labels[n]]
                    face = pred.pred_instances.keypoints[0,23:91,:]
                    face_score = pred.pred_instances.keypoint_scores[0,23:91]
                    face_dist = np.ptp(face,axis=0)/[preds.bboxes[n][2]-preds.bboxes[n][0],preds.bboxes[n][3]-preds.bboxes[n][1]]
                    keep_face = np.all(face_dist<config.face.max_proportion) and (face_score.mean()>config.face.kpt_thr)
                    hands = pred.pred_instances.keypoints[0,91:,:]
                    hand_score = pred.pred_instances.keypoint_scores[0,91:]
                    hand_dist = np.append(
                                    np.ptp(hands[:21],axis=0)/[preds.bboxes[n][2]-preds.bboxes[n][0],preds.bboxes[n][3]-preds.bboxes[n][1]],
                                    np.ptp(hands[21:],axis=0)/[preds.bboxes[n][2]-preds.bboxes[n][0],preds.bboxes[n][3]-preds.bboxes[n][1]]
                                )
                    keep_hand = np.all(hand_dist<config.touch.max_proportion)
                    frame_results[f'{seat_name} occupied'] = 1
                    if keep_face:
                        #face touch
                        mask_face = np.zeros_like(frame)
                        face_points = [(tuple(xy),0) for xy in face[:26]]
                        face_found = False
                        try:
                            consensus_conic_section, inliers, outliers = ellipse_estimator.ConsensusModel(face_points)
                            face_found = True
                        except ValueError:
                            pass
                        if face_found and (consensus_conic_section.ConicSectionType() == 'ellipse'):
                            center, major, minor, theta = consensus_conic_section.EllipseParameters()
                            cv2.ellipse(mask_face, (int(center[0]),int(center[1])), (int(major),int(minor)), math.degrees(theta), 0, 360, (255, 255, 255), -1)
                            mask_face = cv2.dilate(mask_face, dilation_kernel, iterations=1) 
                            mask_face = (mask_face[:,:,0]!=0) & preds.masks[n]
                            if mask_face.sum()>0:
                                bbox_face = mask_to_box(mask_face)
                        
                        #face direction
                        head_pose = head_pose_estimator.solve(face)
                        rmat, _ = cv2.Rodrigues(head_pose[0])
                        yaw = cv2.RQDecomp3x3(rmat)[0][1]
                        if seat_name == 'window':
                            lim = dir_lim.left if 'L' in video_name else dir_lim.right
                        elif seat_name == 'middle':
                            lim = dir_lim.middle
                        elif seat_name == 'aisle':
                            lim = dir_lim.right if 'L' in video_name else dir_lim.left
                        if (yaw<lim.left):
                            direction = 'left'
                        elif  (lim.left<=yaw<=lim.right):
                            direction = 'forward'
                        elif (lim.right<yaw):
                            direction = 'right'
                        vis_names.append(f'{seat_name}: {direction} {int(yaw)}')
                        frame_results[f'{seat_name} facing'] = direction
                        frame_results[f'{seat_name} angle'] = int(yaw)
                        if args.draw_pose:
                            head_pose_estimator.draw_axes(frame, head_pose)
                    else:
                        vis_names.append(f'{seat_name}: low pose')

                    #window touch
                    if keep_hand:
                        num_touch = in_mask(mask_wind,hands,hand_score,config.touch.kpt_thr)
                        if num_touch>=config.touch.wind_min:
                            vis_boxes.append(bbox_wind)
                            vis_labels.append(len(vis_names))
                            vis_names.append('window touch')
                            frame_results['window touch'] = 1
                            frame_results['window num touch'] = num_touch
                        if keep_face:
                            num_touch = in_mask(mask_face,hands,hand_score,config.touch.kpt_thr)
                            if num_touch>=config.touch.face_min:
                                vis_boxes.append(bbox_face)
                                vis_labels.append(len(vis_names))
                                vis_names.append('face touch')
                                frame_results[f'{seat_name} face touch'] = 1
                                frame_results[f'{seat_name} num touch'] = num_touch
            else:
                pred_instances = []
            
            if (args.show or args.out_video) and len(vis_boxes):
                frame = mmcv.imshow_det_bboxes(frame,np.vstack(vis_boxes),
                                            np.array(vis_labels),vis_names,show=False)
                if args.draw_pose:
                    visualizer.add_datasample(
                        name=video_file,
                        image=frame,
                        data_sample=pred_instances,
                        hide_kpts=hide_kpts,
                        kpt_thr=config.vis.kpt_thr,
                        draw_gt=False,
                        draw_heatmap=False,
                        draw_bbox=False,
                        show_kpt_idx=False)
                    frame = visualizer.get_image()

            if args.show:
                mmcv.imshow(frame, video_file, args.wait_time)
            if args.out_video:
                video_writer.write(frame)
            frame_results_list.append(frame_results)

    #end loop
    cv2.destroyAllWindows()
    if args.out_video:
        video_writer.release()
        print('video have been saved')
    if args.out_preds:
        pd.DataFrame(frame_results_list).to_csv(args.out_preds, index=False)
        print('predictions have been saved')


if __name__ == '__main__':
    main()
