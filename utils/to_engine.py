import os
import numpy as np
from ultralytics import YOLO


def load_or_export_model(device=0):
    model = None
    need_train = False

    if os.path.exists("models/yolo11n.engine"):#check if model/xxx.engine
        print(f"Did found .engine, checking capability")
        try:  #try anything using .engine

            test_model = YOLO("models/yolo11n.engine", task='detect')
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)#blank img
            test_model.predict(dummy_frame, imgsz=640, device=device, verbose=False)
            print(f"File .engine good for this comp.")
            model = test_model

        except Exception as e:#.engine file can not use
            print(f"File .engine test failed: {e}")
            os.remove("models/yolo11n.engine")
            need_train = True
    else:
        print(f"Not found .engine file")
        need_train = True

    if need_train:
        print("Conv from yolo11n.pt to .engine Now.")
        pt_model = YOLO("models/yolo11n.pt")
        pt_model.export(format='engine',imgsz=[640, 640], device=device, half=True, simplify=True )#pt ->engine, train by 1280 reset to 640
        del pt_model#delete ptmodel from memory now, realse ram space
        model = YOLO("models/yolo11n.engine", task='detect')#load new .engine
        print("File .engine covert.")

    return model