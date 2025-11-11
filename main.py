import constants
from court_line_detector import CourtLineDetector
import cv2
import pandas as pd
from copy import deepcopy

def main():
    # read selected image from input dir
    input_path = "input/test.png"
    img_proc = cv2.imread(input_path)
    if img_proc is None:
        print("Fetch image error. Quitting...")
        return 0
    
    # 初始化场地线检测模型
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    # 预测场地关键点
    court_keypoints = court_line_detector.predict(img_proc)

    # 绘制输出img
    img_opt  = court_line_detector.draw_keypoints(img_proc, court_keypoints)

    # place image to output dir
    cv2.imwrite("output/output.png", img_opt)

if __name__ == "__main__":
    main()