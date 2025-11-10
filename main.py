from utils import (read_img,
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants
from court_line_detector import CourtLineDetector
import cv2
import pandas as pd
from copy import deepcopy

def main():
    # 读取视频文件，确保输入视频在 input_videos 文件夹下
    input_video_path = "input_videos/test.png"
    #mmexport1723797525791  wu_mei9s    input_video wu1920
    video_frames = read_img(input_video_path)
    
    # 初始化场地线检测模型
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    # 预测第一帧的场地关键点
    court_keypoints = court_line_detector.predict(video_frames[0])

    # 绘制输出视频
    output_video_frames  = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)

    # 保存输出视频到 output_videos 文件夹
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()