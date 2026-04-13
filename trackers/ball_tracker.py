import sys
from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.signal import argrelextrema
import queue
import time

class BallTracker:
    def __init__(self,model_path,fine_fill_service):
        self.model = YOLO(model_path)
        self.ff=fine_fill_service

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

        #take 2 frame right&left and cal current pos. ai suggest Savitzky-Golay but I can't understand, byebye~
        df_ball_positions['x1'] = df_ball_positions['x1'].rolling(window=2, min_periods=1, center=True).mean()
        df_ball_positions['y1'] = df_ball_positions['y1'].rolling(window=2, min_periods=1, center=True).mean()
        df_ball_positions['x2'] = df_ball_positions['x2'].rolling(window=2, min_periods=1, center=True).mean()
        df_ball_positions['y2'] = df_ball_positions['y2'].rolling(window=2, min_periods=1, center=True).mean()

        print("df_ball_positions")
        print(df_ball_positions)
        return df_ball_positions

#    def get_ball_shot_frames(self,ball_positions):
#        ball_positions = [x.get(1,[]) for x in ball_positions]
#        # convert the list into pandas dataframe
#        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
#
#        df_ball_positions['ball_hit'] = 0
#
#        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
#        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
#        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
#        minimum_change_frames_for_hit = 10
#        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
#            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
#            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0
#
#            if negative_position_change or positive_position_change:
#                change_count = 0
#                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
#                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
#                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0
#
#                    if negative_position_change and negative_position_change_following_frame:
#                        change_count+=1
#                    elif positive_position_change and positive_position_change_following_frame:
#                        change_count+=1
#
#                if change_count>minimum_change_frames_for_hit-1:
#                    df_ball_positions.loc[df_ball_positions.index[i], 'ball_hit'] = 1
#
#        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()
#
#        return frame_nums_with_ball_hits

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []
        kf=kalman_filter()#init kf
        ff=fine_fill_gap()
        frame_nums_with_ball_hits=[]

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        box_queue = queue.Queue()
        frame_y,frame_x,channel= frames[0].shape
        heat_map=np.zeros((frame_y+1,frame_x+1),dtype=float)#create 720*480(if is) np list filled with 0; it start from 0, and I don't want to -1 every call
        essay_heat_up_map=np.zeros((frame_y+1,frame_x+1),dtype=float)
        ball_pos_buffer = queue.Queue()
        for frame_no, frame in enumerate(frames):
            yolo_boxes = self.detect_frame(frame)
            box_queue.put(yolo_boxes)
            for posi_ball_box in yolo_boxes[1]:#get all yolo box
                x1,y1,x2,y2,conf=posi_ball_box
                heat_map[max(0,int(y1)-5):min(int(y2)+5,frame_y),max(0,int(x1)-5):min(int(x2)+5,frame_x)]+=1-conf+0.5#5px bloom, the last +0.5 is to offset -0.5 each for frame in frames loop

            frame_no_shifted=frame_no-10
            if frame_no>10:#warm heat map with 10 frame
                cur_pick_box=self.pick_box(heat_map,box_queue)#warm up queue with 10 frame as well, total 20 frame delay
                print(f"{frame_no_shifted}:")
                ball_box, ball_hit=kf.kalman_filter_func(frame_no_shifted,cur_pick_box,self.ff)
                print(ball_hit)
                print("=============")
                ball_detections.append(ball_box)
                self.live_display(frame_no_shifted,frames[frame_no_shifted],ball_box)
                if ball_hit!=[0]:
                    print(f"Appened: {frame_no_shifted-ball_hit[0]}")
                    frame_nums_with_ball_hits.append(frame_no_shifted-ball_hit[0])#frame

            heat_map=heat_map-0.5#heat map cools down
            heat_map = np.clip(heat_map, 0, 30)#no <0 or >30 (2 sec to cool down a known black area)

        while not box_queue.empty():#warm up, so 10 fps behind, handle it
            frame_no_shifted=len(frames)-box_queue.qsize()
            cur_pick_box=self.pick_box(heat_map,box_queue)
            ball_box, ball_hit = kf.kalman_filter_func(frame_no_shifted,cur_pick_box,self.ff)
            ball_detections.append(ball_box)
            self.live_display(frame_no_shifted,frames[frame_no_shifted],ball_box)
            if ball_hit != [0]:
                frame_nums_with_ball_hits.append(frame_no_shifted-ball_hit[0])

        cv2.destroyAllWindows()
        return ball_detections, frame_nums_with_ball_hits

    def pick_box(self,heat_map, box_queue):
        yolo_hist_boxes = box_queue.get()  # get yolo box 10 frame earlier
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

    def live_display(self,frame_no,video_frame,bbox):#intel enough to display, but more acc data need further cal
        x1,y1,x2,y2,conf=bbox
        opt_video_frame=video_frame.copy()
        opt_video_frame=cv2.rectangle(opt_video_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        opt_video_frame=cv2.putText(opt_video_frame, f"Conf: {conf:.3f}" if conf>0 else ("Kalman Filter" if conf==-1 else "Fine fill"),(int(x1),int(y1 -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        opt_video_frame = cv2.resize(opt_video_frame, (0, 0), fx=0.5, fy=0.5)#50% frame size also window size
        opt_video_frame=cv2.putText(opt_video_frame, f"{frame_no}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0, 255, 255), 2)
        opt_video_frame=cv2.putText(opt_video_frame, "Working in progress, not final.", (250, 500), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
        cv2.imshow("Phased streaming", opt_video_frame)
        cv2.waitKey(1)

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
        i=0

        player_detections = df_player_detections.values.tolist()#conv pandas to lsit
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            if ball_dict !=[]:#have box
                x1, y1, x2, y2, conf = ball_dict
                i+=1
                cv2.putText(frame, f"{i}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0, 255, 255), 2)
                cv2.putText(frame, f"Conf: {conf:.3f}" if conf>0 else ("Kalman Filter" if conf==-1 else "Fine fill"),(int(x1),int(y1 -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
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
    def __init__(self):
        self.kf = cv2.KalmanFilter(5, 2)#4 alg: x,y,dx,dy,ddy 2 opt: x,y, ddy aka gravity
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0, 0],#new x=x+dx
                                             [0, 1, 0, 1, 0.1],#new y=y+dy+0.1ddy
                                             [0, 0, 1, 0, 0],#new dx=dx
                                             [0, 0, 0, 1, 1],
                                             [0, 0, 0, 0, 1]], np.float32)

        self.kf.processNoiseCov[4, 4] = 0.3#
        self.kf.statePost = np.array([0, 0, 0, 0, 0.5], np.float32).reshape(-1, 1)#assum ball gravity 0.5px/frame at first
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0,0],[0, 1, 0, 0, 0]], np.float32)#cares x,y only
        #过程噪音
        self.kf.processNoiseCov = np.eye(5, dtype=np.float32) * 1e-3
        #测量噪音
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        #误差
        self.kf.errorCovPost = np.eye(5, dtype=np.float32)*1.5#加快收敛

        self.trust_ball_radius = 80  # if in this area(50px), than box is trust
        self.last_ball_mid_pos = [-1, -1]
        self.ball_box_pos_final=[0,0]

        self.last_ball_motion_change_rate=0
        self.cur_ball_motion_after_hit = [[-1,-1], [-1, -1]]  # frameno,dx,dy

        self.ball_loose_start_frame=-1#first got []

    def update(self, x, y):#update kalman_filter alg
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)#merge with newest x,y
        smooth_x = self.kf.statePost[0][0]
        smooth_y = self.kf.statePost[1][0]

        return [smooth_x, smooth_y]

    def kalman_filter_func(self,frame_no, ball_box_pos, fine_fill_service):
        pred = self.kf.predict()#predit ball pos with kalf
        pred_x = float(pred[0][0])
        pred_y = float(pred[1][0])
        trusted_ball=1

        if ball_box_pos!=[]:#if yolo find a box
            x1,y1,x2,y2,conf=ball_box_pos
            mid_x=(x1+x2)/2
            mid_y=(y1+y2)/2
            ball_moved_dist=(self.last_ball_mid_pos[0]-mid_x)**2+(self.last_ball_mid_pos[1]-mid_y)**2
            if self.last_ball_mid_pos!=[-1,-1]  and ball_moved_dist>self.trust_ball_radius**2:#the box is toooo far away from last box, than not the ball
                trusted_ball=0#box not trusted, go predict
            else:
                smooth_pos=self.update(mid_x,mid_y)#upd with got ball pos
                final_x,final_y=smooth_pos[0],smooth_pos[1]
                self.last_ball_mid_pos=[mid_x,mid_y]#cur ball pos for next kalman_filter_func
                box_h,box_w=x2-x1,y2-y1
                ball_box_pos_final=[int(final_x-(box_h/2)),int(final_y-(box_w/2)),int(final_x+(box_h/2)),int(final_y+(box_w/2)),conf]
                self.trust_ball_radius = 80# reset trust radius
                if self.ball_loose_start_frame!=-1:#ball lost just end
                    fine_fill_service.open_fill_ticket(self.ball_loose_start_frame-1,frame_no)#open a ticket to fill loosing ball pos later
                    self.ball_loose_start_frame=-1#reset

                print(f"Hit flag:{self.cur_ball_motion_after_hit[0]}")
                ball_motion_change_rate=0.5*abs(self.cur_ball_motion_after_hit[1][0] - self.kf.statePost[2][0]) + abs(self.cur_ball_motion_after_hit[1][1] - self.kf.statePost[3][0])
                if self.cur_ball_motion_after_hit[0][0]==-1:
                    print(f"Delta0:{ball_motion_change_rate}")
                    if ball_motion_change_rate>8:#may be a hit, record down
                        self.cur_ball_motion_after_hit[0]=[1,0]#mark as hit
                    self.cur_ball_motion_after_hit[1]=[self.kf.statePost[2][0],self.kf.statePost[3][0]]#recoed cur dx&dy
                    self.last_ball_motion_change_rate = ball_motion_change_rate
                elif self.cur_ball_motion_after_hit[0][0] in range(1,3):# once mark as hit, continue track
                    print(f"Delta1:{ball_motion_change_rate}")
                    if ball_motion_change_rate<max(5,int(self.last_ball_motion_change_rate/2)):#if keep that delta, may be real hit
                        self.cur_ball_motion_after_hit[0][0]+=1#add conf that is real hit
                    else:#not keep delta, may lock logo char etc
                        print(f"Break Combo, lim:{int(self.last_ball_motion_change_rate/2)}")
                        if ball_motion_change_rate > 8:
                            self.cur_ball_motion_after_hit[0] = [1,0]#break combo
                        else:
                            self.cur_ball_motion_after_hit[0] = [-1,-1]
                        self.cur_ball_motion_after_hit[1] = [self.kf.statePost[2][0],self.kf.statePost[3][0]]#recoed cur dx&dy
                    self.last_ball_motion_change_rate = ball_motion_change_rate
                elif self.cur_ball_motion_after_hit[0][0]>=3:#3 real ball pos with simular delta, treat as real hit
                    self.cur_ball_motion_after_hit[0] = [-1,-1]  # reset hit mark
                    print("Hit")
                    return ball_box_pos_final, [-(self.cur_ball_motion_after_hit[0][0]+self.cur_ball_motion_after_hit[0][1])+1]# tell parent hit 3 frame before
                return ball_box_pos_final,[0]

        if ball_box_pos == [] or trusted_ball ==0:  # find no box, predict it
            self.trust_ball_radius = self.trust_ball_radius + 30  # time pass, ball may appear anywhere
            self.last_ball_mid_pos = [pred_x, pred_y]
            if self.ball_loose_start_frame==-1:
                self.ball_loose_start_frame=frame_no
            return [pred_x - 10, pred_y - 10, pred_x + 10, pred_y + 10, -1],[0]

class fine_fill_gap:#上下文填充
    @dataclass
    class Ticket:
        from_frame: int#from_frame+1 would be []
        to_frame: int#to_frame-1 is []

    def __init__(self):
        self.ticket_list=[]
        self.target_arr= np.array([#what we want to: pos_f=af**3+bf**2+cf**3+d 三次多项式
            [0, 0, 0, 1],#x/y start, f is 0
            [1, 1, 1, 1],#x/y end, f is 1, f standarlized
            [0, 0, 1, 0],#dx/dy start, 速率是对位置求导 (af**3+bf**2+cf**3+d)'=3af+2bf+c, f is 0
            [3, 2, 1, 0]#dx/dy end, (af**3+bf**2+cf**3+d)'=3af+2bf+c, f is 1
        ])
        self.route_arr=np.linalg.inv(self.target_arr)#variables to get what we want to

        self.weight= np.array([
            [1, 0, 0, 0],#WTF???
            [1,   0,   0,   0],#2 real length
            [0.7, 0.3, 0,   0],#3 real length
            [0.65, 0.2, 0.15, 0],#4 real length
            [0.62,0.18, 0.12,0.08]#5 real length
        ])

    def open_fill_ticket(self, from_f, to_f):#request fine fill later
        self.ticket_list.append(self.Ticket(from_f, to_f))

    def fine_filling_gap(self,ball_positions):
        ticket_list_index=0
        print(f"ticket_list{self.ticket_list}")
        while ticket_list_index < len(self.ticket_list):
            ticket=self.ticket_list[ticket_list_index]
            start_frame=ticket.from_frame
            end_frame=ticket.to_frame

            if start_frame<=2 or len(ball_positions)-end_frame<=2:
                ticket_list_index+=1
                continue#not enough data, may at start or end, do nothing

            #Merge nearby no ball pos
            ball_pos_from_end=[]
            ticket_shift=0
            for ticket_list_index_sub in range(ticket_list_index+1,len(self.ticket_list)):
                if self.ticket_list[ticket_list_index_sub].from_frame-end_frame<=2:#if 2 ticket apart <=2 frame, not enough data at end, merge later ticket
                    end_frame=self.ticket_list[ticket_list_index+1].to_frame
                    ticket_list_index+=1#move pointer to next ticket
                    ticket_shift+=1#moved pointer step
                else:#enough space at the end, process to cal
                    break

            if end_frame-start_frame<=2:#small gap, not worth, kalman_filter_func can do well
                ticket_list_index+=1
                continue

            total_length=end_frame-start_frame
            #cal dx dy for start_frame
            ball_pos_till_start=[]
            length=0
            for ball_positions_index in range(start_frame,max(0,start_frame-5, ( (self.ticket_list[ticket_list_index-1-ticket_shift].to_frame) if ticket_list_index>0 else -1) ),-1):#grab data from start to start-5 frame
                ball_pos_till_start.append([(ball_positions[ball_positions_index][0]+ball_positions[ball_positions_index][2])/2, (ball_positions[ball_positions_index][1]+ball_positions[ball_positions_index][3])/2])
                length+=1
            while len(ball_pos_till_start)<5:#fill[0,0] till len()==5
                ball_pos_till_start.append([0,0])
            start_x=ball_pos_till_start[0][0]
            start_y=ball_pos_till_start[0][1]
            start_dx=total_length*(self.weight[length-1][0]*(ball_pos_till_start[0][0]-ball_pos_till_start[1][0])+self.weight[length-1][1]*(ball_pos_till_start[1][0]-ball_pos_till_start[2][0])+self.weight[length-1][2]*(ball_pos_till_start[2][0]-ball_pos_till_start[3][0])+self.weight[length-1][3]*(ball_pos_till_start[3][0]-ball_pos_till_start[4][0]))
            start_dy=total_length*(self.weight[length-1][0]*(ball_pos_till_start[0][1]-ball_pos_till_start[1][1])+self.weight[length-1][1]*(ball_pos_till_start[1][1]-ball_pos_till_start[2][1])+self.weight[length-1][2]*(ball_pos_till_start[2][1]-ball_pos_till_start[3][1])+self.weight[length-1][3]*(ball_pos_till_start[3][1]-ball_pos_till_start[4][1]))

            #cal dx dy for end frame
            length=0
            for ball_positions_index in range(end_frame,min(len(ball_positions),end_frame+5, ( (self.ticket_list[ticket_list_index+1].from_frame) if ticket_list_index>0 else sys.maxsize) )):#grab data from end point to end+5 frame
                ball_pos_from_end.append([(ball_positions[ball_positions_index][0]+ball_positions[ball_positions_index][2])/2, (ball_positions[ball_positions_index][1]+ball_positions[ball_positions_index][3])/2])
                length+=1
            while len(ball_pos_from_end)<5:
                ball_pos_from_end.append([0,0])
            end_x=ball_pos_from_end[0][0]
            end_y=ball_pos_from_end[0][1]
            end_dx=total_length*(self.weight[length-1][0]*(ball_pos_from_end[1][0]-ball_pos_from_end[0][0])+self.weight[length-1][1]*(ball_pos_from_end[2][0]-ball_pos_from_end[1][0])+self.weight[length-1][2]*(ball_pos_from_end[3][0]-ball_pos_from_end[2][0])+self.weight[length-1][3]*(ball_pos_from_end[4][0]-ball_pos_from_end[3][0]))
            end_dy=total_length*(self.weight[length-1][0]*(ball_pos_from_end[1][1]-ball_pos_from_end[0][1])+self.weight[length-1][1]*(ball_pos_from_end[2][1]-ball_pos_from_end[1][1])+self.weight[length-1][2]*(ball_pos_from_end[3][1]-ball_pos_from_end[2][1])+self.weight[length-1][3]*(ball_pos_from_end[4][1]-ball_pos_from_end[3][1]))

            #cal confficient
            pred_x_a=self.route_arr[0][0]*start_x+self.route_arr[0][1]*end_x+self.route_arr[0][2]*start_dx+self.route_arr[0][3]*end_dx
            pred_x_b=self.route_arr[1][0]*start_x+self.route_arr[1][1]*end_x+self.route_arr[1][2]*start_dx+self.route_arr[1][3]*end_dx
            pred_x_c=self.route_arr[2][0]*start_x+self.route_arr[2][1]*end_x+self.route_arr[2][2]*start_dx+self.route_arr[2][3]*end_dx
            pred_x_d=self.route_arr[3][0]*start_x+self.route_arr[3][1]*end_x+self.route_arr[3][2]*start_dx+self.route_arr[3][3]*end_dx
            pred_y_a=self.route_arr[0][0]*start_y+self.route_arr[0][1]*end_y+self.route_arr[0][2]*start_dy+self.route_arr[0][3]*end_dy
            pred_y_b=self.route_arr[1][0]*start_y+self.route_arr[1][1]*end_y+self.route_arr[1][2]*start_dy+self.route_arr[1][3]*end_dy
            pred_y_c=self.route_arr[2][0]*start_y+self.route_arr[2][1]*end_y+self.route_arr[2][2]*start_dy+self.route_arr[2][3]*end_dy
            pred_y_d=self.route_arr[3][0]*start_y+self.route_arr[3][1]*end_y+self.route_arr[3][2]*start_dy+self.route_arr[3][3]*end_dy

            #predicting
            for ball_positions_index in range(1,total_length):
                pred_x=pred_x_a*(ball_positions_index/total_length)**3+pred_x_b*(ball_positions_index/total_length)**2+pred_x_c*(ball_positions_index/total_length)+pred_x_d
                pred_y=pred_y_a*(ball_positions_index/total_length)**3+pred_y_b*(ball_positions_index/total_length)**2+pred_y_c*(ball_positions_index/total_length)+pred_y_d
                ball_positions[start_frame+ball_positions_index]=[pred_x-10,pred_y-10,pred_x+10,pred_y+10,-2]
                print(f"frame:{start_frame+ball_positions_index} predict: [{pred_x}, {pred_y}]")

            ticket_list_index+=1

        return ball_positions
