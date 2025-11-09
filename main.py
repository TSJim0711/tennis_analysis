from utils import (read_img,
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants
from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy

def main():
    # 读取视频文件，确保输入视频在 input_videos 文件夹下
    input_video_path = "input_videos/feder6s.mp4"
    #mmexport1723797525791  wu_mei9s    input_video wu1920
    video_frames = read_img(input_video_path)

    # 初始化球员和网球的检测追踪器
    # 确保模型文件在 models 文件夹下
    # PlayerTracker 使用 YOLOv8 模型来检测球员
    # BallTracker 使用 YOLOv5 模型来检测网球
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')

    # 检测球员，优先从缓存文件读取检测结果，加快调试速度
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,  #False True
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    # 检测网球，优先从缓存文件读取检测结果
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,  #False True
                                                     stub_path="tracker_stubs/ball_detections.pkl"
                                                     )
    # 对网球轨迹进行插值，补全缺失的球位置
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    
    # 初始化场地线检测模型
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    # 预测第一帧的场地关键点
    court_keypoints = court_line_detector.predict(video_frames[0])

    # 根据场地关键点筛选和过滤球员检测结果，确保球员在场地内
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # 初始化小场地（MiniCourt）对象，用于后续坐标转换和可视化
    mini_court = MiniCourt(video_frames[0]) 

    # 检测每一次击球的帧编号
    ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)

    # 将球员和球的检测框坐标转换到小场地坐标系
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)

    # 初始化球员统计数据，第一帧为0
    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]
    
    # 遍历每一次击球，统计球速和球员移动速度等信息
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # 计算球在小场地上的像素距离
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        
        # 将像素距离转换为实际米数
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        # 计算本次击球的球速（单位：km/h）
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # 判断本次击球的球员（距离球最近的球员）
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))

        # 计算对方球员的移动速度
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                               player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           )

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6


        # 复制上一帧的统计数据，更新当前帧的统计数据
        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    # 将球员统计数据转换为DataFrame，方便后续处理
    player_stats_data_df = pd.DataFrame(player_stats_data)
    # 构建所有帧的编号DataFrame
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    # 合并统计数据，保证每一帧都有对应数据
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    # 用前向填充补全空值
    player_stats_data_df = player_stats_data_df.ffill()

    # 计算球员的平均击球速度和平均移动速度
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']



    # 绘制输出视频
    ## 绘制球员检测框
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
    ## 绘制网球检测框
    output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## 绘制场地关键点
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # 绘制小场地和球员、球的轨迹点
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

    # 绘制球员统计信息
    output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

    ## 在左上角绘制帧编号
    for i, frame in enumerate(output_video_frames):
         cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 保存输出视频到 output_videos 文件夹
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()