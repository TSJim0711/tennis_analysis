from utils import ( read_video,
                    save_video,
                    measure_distance,
                    draw_player_stats,
                    convert_pixel_distance_to_meters,
                    load_or_export_model
                   )
import constants
from court_line_detector import CourtLineDetector
from trackers import BallTracker
import cv2
import pandas as pd
from copy import deepcopy

def main():
    # 读取视频文件，确保输入视频在 input_videos 文件夹下
    input_video_path = "input_videos/input_video.mp4"
    #mmexport1723797525791  wu_mei9s    input_video wu1920
    video_frames = read_video(input_video_path)

    save_video(video_frames, "output_videos/output_video.avi")

    # BallTracker 使用 YOLOv5 模型来检测网球
    ball_tracker = BallTracker(model_path=load_or_export_model(device=0))#use device=0 aka gpu
    # 检测网球，优先从缓存文件读取检测结果
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,  #False True
                                                     stub_path="tracker_stubs/ball_detections.pkl"
                                                     )
    # 对网球轨迹进行插值，补全缺失的球位置
    df_ball_positions = ball_tracker.interpolate_ball_positions(ball_detections)
    #determind ball bounce/hit
    ball_bounce_frame=ball_tracker.ball_hits(df_ball_positions)

    # 初始化场地线检测模型
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    # 预测第一帧的场地关键点
    court_keypoints = court_line_detector.predict(video_frames[0])

    # 绘制输出视频
    output_video_frames  = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)
    ## 绘制网球检测框
    output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    ##print ball bounce/hit to video
    output_video_frames = ball_tracker.print_ball_hit_log(output_video_frames,ball_bounce_frame)
    # 保存输出视频到 output_videos 文件夹
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()


