from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.signal import argrelextrema
from collections import deque

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        processed_positions = []
        print("BALL LIST")
        print(ball_positions)
        for frame_detections in ball_positions:

            # 如果是空，填空列表
            if not frame_detections:
                processed_positions.append([])
                continue

            if isinstance(frame_detections, dict):
                if len(frame_detections) > 0:
                    # 取字典里的第一个值作为球的位置
                    data = list(frame_detections.values())[0]
                else:
                    data = []
            else:
                data = frame_detections

            processed_positions.append(data)
        print(processed_positions)
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(processed_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        #take 2 frame right&left and cal current pos. ai suggest Savitzky-Golay but I can't understand, byebye~
        df_ball_positions['x1'] = df_ball_positions['x1'].rolling(window=2, min_periods=1, center=True).mean()
        df_ball_positions['y1'] = df_ball_positions['y1'].rolling(window=2, min_periods=1, center=True).mean()
        df_ball_positions['x2'] = df_ball_positions['x2'].rolling(window=2, min_periods=1, center=True).mean()
        df_ball_positions['y2'] = df_ball_positions['y2'].rolling(window=2, min_periods=1, center=True).mean()

        return df_ball_positions

    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1

                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions.loc[df_ball_positions.index[i], 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    frame_mid_pos=[0,0,0,0]#the mid point of the video frame, this is globe var, var given by detect_frames, would be used in detect_frames & detect_ball
    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        history = {}#data from history frames
        blacklist_areas = []#静止物品的区域
        ball_pos_log= []
        max_fixed_itm_radius=10#静止物品相对摄像头的最大浮动像素范围
        max_fixed_itm_track_frame=30
        self.frame_mid_pos=[len(frames[0]) / 2, len(frames) / 2, len(frames[0]) / 2, len(frames) / 2]
        for frame_num, ball_dict in enumerate(ball_detections):
            ball_pos_log.append(self.detect_ball(frame_num, ball_dict, blacklist_areas, history, ball_pos_log if len(ball_pos_log)>0 else [{1:self.frame_mid_pos}], max_fixed_itm_radius, max_fixed_itm_track_frame))
        return ball_pos_log

    def detect_ball(self,frame_num, ball_dict, blacklist_areas, history, ball_pos_log, max_fixed_itm_radius, max_fixed_itm_track_frame):
        fix_itm_flag = 0
        potential_boxes = ball_dict.get(1, [])  # get all box
        potential_boxes_sorted = []
        if not potential_boxes:
            return {1: []}
        chosen_box = None
        # sort box
        min_dist = float('inf')
        for box in potential_boxes:
            # remove box don't move (aka box in banned area)
            x1, y1, x2, y2, conf = box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            for bad_center in blacklist_areas:
                dist = np.linalg.norm(np.array(center) - np.array(bad_center))  # cal dist from cur box to nearest blacklist center
                if dist < max_fixed_itm_radius:
                    fix_itm_flag = 1  # mark as fix item (not moving)
                    break  # skip for bad_center in blacklist_areas:

            if fix_itm_flag == 1:  # fix item, not ball
                fix_itm_flag = 0
                continue

            # find cid nearest to cur box
            matched_id = None
            for cid, points in history.items():
                last_point = points[-1]
                dist = np.linalg.norm(np.array(center) - np.array(last_point))
                if dist < max_fixed_itm_radius:
                    matched_id = cid
                    break

            # not in any blacklist, handle it
            if matched_id is None:
                matched_id = len(history)
                history[matched_id] = deque(maxlen=max_fixed_itm_track_frame)  # create slot for new obj,only keep latest 30 data

            # add box pos to cid history
            history[matched_id].append(center)

            # obj reach 30 data, start cal
            if len(history[matched_id]) >= max_fixed_itm_track_frame:
                start_pt = np.array(history[matched_id][0])  # load data from start
                end_pt = np.array(history[matched_id][-1])  # to end
                total_movement = np.linalg.norm(end_pt - start_pt)  # cal obj move pixel in 1sec, OGLD dist
                if total_movement < max_fixed_itm_radius:
                    # not moving! not ball! ban now!
                    blacklist_areas.append(center)
                    del history[matched_id]
                    #re-determine latest 30(or ball_pos_log len if it's smaller) frame ball box, according to new ban rule
                    for frame_ptr in range(-min(max_fixed_itm_track_frame, len(ball_pos_log)),-1,1):
                        ball_pos_log[frame_ptr]=self.detect_ball(frame_ptr, ball_dict, blacklist_areas, history, ball_pos_log if len(ball_pos_log)>0 else [{1:self.frame_mid_pos}], max_fixed_itm_radius, max_fixed_itm_track_frame)
                    continue  # handle other box in potential_boxes now

            potential_boxes_sorted.append(box)  # more possible box with ball after cal

        # box nearest to last box(is ball) is ball
        ball_box = None
        for box in potential_boxes_sorted:
            x1, y1, x2, y2, conf = box

            for last_ball_pos in reversed(ball_pos_log[:frame_num]):#if last frame found no ball box, get it from more early frames
                if last_ball_pos[1]!=[]:
                    xx1, yy1, xx2, yy2 = last_ball_pos[1]
            else :#no valid box from start to end, set value as frame middle pos.
                xx1, yy1, xx2, yy2 = self.frame_mid_pos

            box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            last_ball_center = np.array([(xx1 + xx2) / 2, (yy1 + yy2) / 2])
            dist = np.linalg.norm(box_center - last_ball_center)
            if dist < min_dist:
                min_dist = dist
                ball_box = box

        current_frame_dict = {1: []}
        if ball_box is not None:
            current_frame_dict[1] = ball_box[:4]
        return current_frame_dict

    def detect_frame(self,frame):
        results = self.model.predict(frame,conf=0.15)[0]
        ball_dict = {}
        ball_dict[1] = []
        for box in results.boxes:
            pos = box.xyxy.tolist()[0]
            confidence = box.conf.tolist()[0]
            ball_dict[1].append(pos + [confidence])#add all box position and conf to list
        return ball_dict

    def ball_hits(self,df_ball_positions):
        @dataclass
        class MyStruct:
            frameNo: int
            signal: int
            #assum 1=ground, 2=player hit
        # convert the list into pandas dataframe
        #find bounce
        bounce_indices = argrelextrema(df_ball_positions['y1'].values, np.greater, order=10)[0]#find 局部最大时的帧 in 5 frame around
        print(bounce_indices)
        return bounce_indices

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []

        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                if bbox !=[]:#have box
                    x1, y1, x2, y2 = bbox
                    cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames

    def print_ball_hit_log(self,video_frames,ball_bounce_frame):
        output_video_frames = []
        for i, cur_frame in enumerate(video_frames):#work for every frame
            if(i in ball_bounce_frame):
                cv2.rectangle(cur_frame, (0, 610), (1280,630 ), (20, 20, 20), thickness=-1)
                cv2.putText(cur_frame,"The ball bounced.",(10,620),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 0, 255),3)
            output_video_frames.append(cur_frame)
        return output_video_frames
