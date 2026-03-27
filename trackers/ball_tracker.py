from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.signal import argrelextrema
import queue

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

            x1,y1,x2,y2,conf=frame_detections
            processed_positions.append([x1,y1,x2,y2,conf])

        print("processed_positions")
        print(processed_positions)
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(processed_positions,columns=['x1','y1','x2','y2','conf'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        #take 3 frame right&left and cal current pos. ai suggest Savitzky-Golay but I can't understand, byebye~
        df_ball_positions['x1'] = df_ball_positions['x1'].rolling(window=3, min_periods=1, center=True).mean()
        df_ball_positions['y1'] = df_ball_positions['y1'].rolling(window=3, min_periods=1, center=True).mean()
        df_ball_positions['x2'] = df_ball_positions['x2'].rolling(window=3, min_periods=1, center=True).mean()
        df_ball_positions['y2'] = df_ball_positions['y2'].rolling(window=3, min_periods=1, center=True).mean()

        print("df_ball_positions")
        print(df_ball_positions)
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

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []
        kf=kalman_filter()#init kf

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        box_queue = queue.Queue()
        frame_y,frame_x,channel= frames[0].shape
        heat_map=np.zeros((frame_y+1,frame_x+1),dtype=float)#create 720*480(if is) np list filled with 0; it start from 0, and I don't want to -1 every call
        for frame_no, frame in enumerate(frames):
            yolo_boxes = self.detect_frame(frame)
            box_queue.put(yolo_boxes)
            for posi_ball_box in yolo_boxes[1]:#get all yolo box
                x1,y1,x2,y2,conf=posi_ball_box
                heat_map[max(0,int(y1)-5):min(int(y2)+5,frame_y),max(0,int(x1)-5):min(int(x2)+5,frame_x)]+=1-conf+0.5#5px bloom, the last +0.5 is to offset -0.5 each for frame in frames loop

            if frame_no>30:#warm heat map with 30 frame
                cur_pick_box=self.pick_box(heat_map,box_queue)
                ball_detections.append(kf.kalman_filter_func(cur_pick_box))

            heat_map=heat_map-0.5#heat map cools down
            heat_map = np.clip(heat_map, 0, 30)#no <0 or >30 (2 sec to cool down a known black area)

        while not box_queue.empty():#warm up, so 30 fps behind, handle it
            cur_pick_box=self.pick_box(heat_map,box_queue)
            ball_detections.append(kf.kalman_filter_func(cur_pick_box))

        return ball_detections

    def pick_box(self,heat_map, box_queue):
        yolo_hist_boxes = box_queue.get()  # get yolo box 30 frame earlier
        coolest_temp = 15.00
        coolest_box = []
        if yolo_hist_boxes[1] == []:#yolo find no box
            return []
        for posi_ball_box in yolo_hist_boxes[1]:
            x1, y1, x2, y2, conf = posi_ball_box
            if heat_map[round((y1 + y2) / 2)][
                round((x1 + x2) / 2)] <= coolest_temp:  # box locate in the coolest pt wins
                coolest_temp = heat_map[round((y1 + y2) / 2)][round((x1 + x2) / 2)]
                coolest_box.append([x1, y1, x2, y2, conf])

        hi_conf = 0
        hi_conf_box = []
        if len(coolest_box) > 1:  # if more then 1 box with same temp
            for posi_ball_box in coolest_box:
                x1, y1, x2, y2, conf = posi_ball_box
                if conf > hi_conf:  # take highest conf box
                    hi_conf = conf#will have same conf?🤔
                    hi_conf_box=[x1, y1, x2, y2, conf]
            return hi_conf_box
        elif len(coolest_box) == 1:
            return coolest_box[0]
        else:#find no box(may be temp>15)
            return []

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
        print("bounce_indices")
        print(bounce_indices)
        return bounce_indices

    def draw_bboxes(self,video_frames, df_player_detections):
        output_video_frames = []

        player_detections = df_player_detections.values.tolist()#conv pandas to lsit
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            if ball_dict !=[]:#have box
                x1, y1, x2, y2, conf = ball_dict
                cv2.putText(frame, f"Conf: {conf:.3f}",(int(x1),int(y1 -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
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

class kalman_filter:
    trust_ball_radius=100#if in this area(50px), than box is trust
    last_ball_mid_pos=[-1,-1]

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)#4 alg: x,y,dx,dy, 2 opt: x,y
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],#new x=x+dx
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],#new dx=dx
                                             [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],[0, 1, 0, 0]], np.float32)#cares x,y only
        #过程噪音
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        #测量噪音
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        #误差
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)*1.5#加快收敛

    def update(self, x, y):#update kalman_filter alg
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)#merge with newest x,y
        smooth_x = self.kf.statePost[0][0]
        smooth_y = self.kf.statePost[1][0]

        return [smooth_x, smooth_y]

    def kalman_filter_func(self, ball_box_pos):
        pred = self.kf.predict()#predit ball pos with kalf
        pred_x = float(pred[0][0])
        pred_y = float(pred[1][0])
        trusted_ball = 1
        if ball_box_pos!=[]:#if yolo find a box
            x1,y1,x2,y2,conf=ball_box_pos
            mid_x=(x1+x2)/2
            mid_y=(y1+y2)/2
            if self.last_ball_mid_pos!=[-1,-1]  and (self.last_ball_mid_pos[0]-mid_x)**2+(self.last_ball_mid_pos[1]-mid_y)**2>self.trust_ball_radius**2:#the box is toooo far away from last box, than not the bakk
                trusted_ball=0#box not trusted, go predict
            else:
                smooth_pos=self.update(mid_x,mid_y)#upd with got ball pos
                final_x,final_y=smooth_pos[0],smooth_pos[1]
                box_h,box_w=x2-x1,y2-y1
                ball_box_pos_final=[int(final_x-(box_h/2)),int(final_y-(box_w/2)),int(final_x+(box_h/2)),int(final_y+(box_w/2)),conf]
                self.trust_ball_radius = 100# reset trust radius
                self.last_ball_mid_pos=[mid_x,mid_y]
                return ball_box_pos_final

        if ball_box_pos==[] or trusted_ball==0:#find no box, predict it
            self.trust_ball_radius = self.trust_ball_radius + 30  # time pass, ball may appear anywhere
            self.last_ball_mid_pos=[pred_x,pred_y]
            return [pred_x-10,pred_y-10,pred_x+10,pred_y+10,-1]

