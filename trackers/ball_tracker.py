from mpmath.math2 import sqrt2
from sympy.codegen.cnodes import sizeof
from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.signal import argrelextrema
from collections import deque
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from math import atan, pi
import queue

class BallTracker:
    def __init__(self,model_path):
        self.yolo_model = YOLO(model_path)
    def interpolate_ball_positions(self, ball_positions):
        print("Cal ball motion formular")
        v=0#x=vf+x0
        a=0; b=0; c=0#y=af2+bf+c
        cur_window_frame_no=[]
        cur_window_ball_x=[];cur_window_ball_y=[]
        ball_move_form_x=[];ball_move_form_y=[]
        formular_list=[];frame_of_list=[]#frame no. the formula belong
        frame_cnt=len(ball_positions)
        scaler = StandardScaler()
        db: DBSCAN = DBSCAN(eps=0.5, min_samples=3)#聚类 alg
        for frame_no in range (0,frame_cnt-13,1):#shift 1 frame/ step
            for frame_no_sub in range (frame_no,frame_no+11,2):#3 frame apart from each other, take 0,2,4,6,8,10 frame data
                if ball_positions[frame_no_sub] !=[]:#skip empty (yolo detect no ball)
                    x1,y1,x2,y2,conf=ball_positions[frame_no_sub]
                    cur_window_ball_x.append((x1+x2)/2)#append mid point of ball box
                    cur_window_ball_y.append((y1+y2)/2)
                    cur_window_frame_no.append(frame_no_sub)

            if len(cur_window_frame_no)>=5:#at least 5 valid
                ball_move_form_x.append(np.polyfit(cur_window_frame_no, cur_window_ball_x, deg=1))# cal x axis ball 线性拟合
                ball_move_form_y.append(np.polyfit(cur_window_frame_no, cur_window_ball_y, deg=2))#cal y axis ball 抛物线 formular
                formular_list.append([ball_move_form_x[-1][0],ball_move_form_x[-1][1],ball_move_form_y[-1][0],ball_move_form_y[-1][1],ball_move_form_y[-1][2]])#write down above data\
                frame_of_list.append(frame_no+6)
            cur_window_ball_x = [];cur_window_ball_y = [];cur_window_frame_no = []  # loop once, clean up

        scaled_form = scaler.fit_transform(formular_list)#standarlize all formulas
        db.fit(scaled_form)#load to DBSCAN
        labels= db.labels_

        chosen_form_dict = []
        if len(labels) > 0:
            for i, label in enumerate(labels):
                if label != -1:  #if not random noise than take it
                    chosen_form_dict.append({"frame no": frame_of_list[i], "formula": formular_list[i]})

        chosen_form_dict.sort(key=lambda x: x["frame no"])#sort by time aka frame no asec
        return chosen_form_dict

    def get_ball_form_switch_frames(self,ball_pos_form_list):#ball_pos_form = result of chosen_form_dict
        print("Splitting formulas")
        frame_no_switch_ball_form_final=0
        frame_no_switch_ball_form=[]
        frame_no_ball_hit=[]
        i=0
        for i in range(0,len(ball_pos_form_list)-1):
            class ball_pos_form_cur:
                a=ball_pos_form_list[i]["formula"][2]
                b = ball_pos_form_list[i]["formula"][3]
                c = ball_pos_form_list[i]["formula"][4]
            class ball_pos_form_next:
                a=ball_pos_form_list[i+1]["formula"][2]
                b = ball_pos_form_list[i+1]["formula"][3]
                c = ball_pos_form_list[i+1]["formula"][4]
            class ball_pos_form_mix:
                A=ball_pos_form_cur.a-ball_pos_form_next.a#(a1+a2)f2+(b1+b2)f+(c1+c2)=0
                B = ball_pos_form_cur.b - ball_pos_form_next.b
                C = ball_pos_form_cur.c - ball_pos_form_next.c
                delta=B**2-4*A*C
                if (A==0):#become linear
                    root_pos=C/B
                    delta=0#let program treat as 1 root
                elif (delta<0):
                    print("WTF?1")
                    root_pos = (ball_pos_form_list[i]["frame no"] + ball_pos_form_list[i + 1]["frame no"]) / 2#no intersect, sad move
                else:
                    root_pos=(-B+(B**2-4*A*C)**0.5)/(2*A)
                if delta > 0:
                    root_neg = (-B - (B ** 2 - 4 * A * C)**0.5) / (2 * A)

            if ball_pos_form_mix.delta <= 0:
                frame_no_switch_ball_form_final=ball_pos_form_mix.root_pos
            elif ball_pos_form_mix.delta>0:
                if ball_pos_form_mix.root_pos in range(ball_pos_form_list[i]["frame no"],ball_pos_form_list[i + 1]["frame no"]) and ball_pos_form_mix.root_neg not in range(ball_pos_form_list[i]["frame no"],ball_pos_form_list[i + 1]["frame no"]):#prevent time reverse
                    frame_no_switch_ball_form_final=ball_pos_form_mix.root_pos
                elif ball_pos_form_mix.root_neg in range(ball_pos_form_list[i]["frame no"],ball_pos_form_list[i + 1]["frame no"]):
                    frame_no_switch_ball_form_final=ball_pos_form_mix.root_neg
                #2 possible roots (both root in range), find the largest y root, ball from high 2 low and low 2 high, impossible low 2 high and hit ceiling and from high 2 low
                elif ball_pos_form_cur.a*(ball_pos_form_mix.root_pos**2)+ball_pos_form_cur.b*ball_pos_form_mix.root_pos+ball_pos_form_cur.c >ball_pos_form_cur.a*(ball_pos_form_mix.root_neg**2)+ball_pos_form_cur.b*ball_pos_form_mix.root_neg+ball_pos_form_cur.c:
                    frame_no_switch_ball_form_final=ball_pos_form_mix.root_pos
                else:
                    frame_no_switch_ball_form_final=ball_pos_form_mix.root_neg
            else:
                print("WTF?2")

            if frame_no_ball_hit !=[] and frame_no_switch_ball_form_final-frame_no_ball_hit[-1]<90:#form <3sec
                frame_no_switch_ball_form.append(frame_no_switch_ball_form_final)
            else:#two form >3sec is unreasonable, take 2 formulars mid frame
                frame_no_switch_ball_form.append((ball_pos_form_list[i]["frame no"]+ball_pos_form_list[i+1]["frame no"])/2)
            print(f"{i}: Append: {frame_no_switch_ball_form_final}")

            #is that ball bounce/ ball hit frame?
            form_cur_intersect_slop=2*ball_pos_form_cur.a*frame_no_switch_ball_form_final+ball_pos_form_cur.b#2af+b
            form_next_intersect_slop=2*ball_pos_form_next.a*frame_no_switch_ball_form_final+ball_pos_form_next.b
            form_intersect_angle_diff=(180/pi)*atan(abs((form_cur_intersect_slop-form_next_intersect_slop)/(1+form_cur_intersect_slop*form_next_intersect_slop)))
            if form_intersect_angle_diff>45:#2 formular intersect angle diff larger than 45 deg, treat as ball bounce/ ball hit
                frame_no_ball_hit.append(frame_no_switch_ball_form_final)

        return frame_no_switch_ball_form,frame_no_ball_hit

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

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
                ball_detections.append(self.pick_box(heat_map,box_queue))

            heat_map=heat_map-0.5#heat map cools down
            heat_map = np.clip(heat_map, 0, 30)#no <0 or >30 (2 sec to cool down a known black area)

        while not box_queue.empty():#warm up, so 30 fps behind, handle it
            ball_detections.append(self.pick_box(heat_map,box_queue))
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
        results = self.yolo_model.predict(frame,conf=0.15)[0]
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
        return bounce_indices

    def draw_bboxes(self,video_frames, ball_move_form, ball_form_switch_guide, ball_bounce_frame):
        output_video_frames = []
        using_form=0
        for frame_no in range(0,len(video_frames)):
            opt_frame = video_frames[frame_no]
            #Draw Bounding Boxes
            if(using_form+1<len(ball_form_switch_guide) and frame_no>=ball_form_switch_guide[using_form+1]):#change formular if reach
                using_form+=1
                print(f"{using_form}: Useing:")
                print(ball_move_form[using_form]["formula"])
                if frame_no in ball_bounce_frame:#cur frame ball bounce/ ball hit
                    cv2.rectangle(opt_frame, (0, 600), (1280, 630), (20, 20, 20), thickness=-1)#backgrond
                    cv2.putText(opt_frame,"The ball bounced.",(10,620),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 0, 255),3)
            #cal ball pos with ball motion formula
            x_mid=ball_move_form[using_form]["formula"][0]*frame_no+ball_move_form[using_form]["formula"][1]
            y_mid=ball_move_form[using_form]["formula"][2]*frame_no**2+ball_move_form[using_form]["formula"][3]*frame_no+ball_move_form[using_form]["formula"][4]
            cv2.rectangle(opt_frame,(round(x_mid)-10,round(y_mid)-10),(round(x_mid)+10,round(y_mid)+10), (0, 255, 255), 2)
            cv2.putText(opt_frame, f"Form", (round(x_mid)-10, round(y_mid)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(opt_frame,f"Form {using_form}: x={round(ball_move_form[using_form]['formula'][0])}f+{round(ball_move_form[using_form]['formula'][1])}f",(0,110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255,255), 2)
            cv2.putText(opt_frame, f"Form {using_form}: y={round(ball_move_form[using_form]['formula'][2])}f2+{round(ball_move_form[using_form]['formula'][3])}f+{round(ball_move_form[using_form]['formula'][4])}",(0,140), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0, 255, 255), 2)
            cv2.putText(opt_frame, f"x={x_mid} y={y_mid}",(0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255,), 2)
            output_video_frames.append(opt_frame)
        return output_video_frames


    def draw_bboxes_origin (self,video_frames, player_detections):
        output_video_frames=[]
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            if ball_dict !=[]:#have box
                x1, y1, x2, y2, conf = ball_dict
                cv2.putText(frame, f"Yolo",(int(x1),int(y1 -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            output_video_frames.append(frame)
        return output_video_frames
