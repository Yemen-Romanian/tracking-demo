import cv2
import logging
from pathlib import Path
import os
import argparse

def build_tracker(tracker_name: str):

    if tracker_name == "kcf":
        return cv2.legacy.TrackerKCF_create()
    elif tracker_name == "mil":
        return cv2.TrackerMIL_create()
    elif tracker_name == "mosse":
        return cv2.legacy.TrackerMOSSE_create()
    elif tracker_name == "nano":
        params = cv2.TrackerNano_Params()
        params.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        params.target = cv2.dnn.DNN_TARGET_CPU
        params.backbone = "../network_weights/nano/nanotrack_backbone_sim.onnx"
        params.neckhead = "../network_weights/nano/nanotrack_head_sim.onnx"
        return cv2.TrackerNano_create(params)
    elif tracker_name == "csrt":
        return cv2.legacy.TrackerCSRT_create()
    elif tracker_name == "medianflow":
        return cv2.legacy.TrackerMedianFlow_create()
    elif tracker_name == "tld":
        return cv2.legacy.TrackerTLD_create()
    else:
        raise ValueError(f"Could not create tracker with name {tracker_name}")
    
def build_video_capture(video_path, data_type):
    if video_path is None:
        return cv2.VideoCapture(0)

    video_root_folder = Path(video_path)
    
    if not video_root_folder.exists():
        raise ValueError(f"Could not find file or folder named {video_path}")
    
    if data_type == "single":
        return cv2.VideoCapture(video_root_folder)
    
    elif data_type == "vot":
        return cv2.VideoCapture(f"{video_path}/%08d.jpg")
    
    elif data_type == "uav":
        return cv2.VideoCapture(f"{video_path}/img%06d.jpg")

    
def run_demo(data_path, data_type, tracker_type):
    capture = build_video_capture(data_path, data_type)

    if not capture.isOpened():
        raise ValueError(f"Wrong video path {data_path}")
    
    tracker = build_tracker(tracker_type)

    status, frame = capture.read()

    bbox = cv2.selectROI(frame, printNotice=True)
    cv2.destroyAllWindows()
    bbox_color = (255, 0, 0)
    tracker.init(frame, bbox)

    while (True):
        status, frame = capture.read()

        if not status:
            logging.warning("Video sequency ends, ending program...")
            cv2.destroyAllWindows()
            return
        
        tracking_result, new_bbox = tracker.update(frame)

        x, y, w, h = map(int, new_bbox)

        if tracking_result:
            print('detected object')
            frame_to_display = cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 2, 1)
        else:
            print('could not detect object')
            frame_to_display = cv2.putText(frame, "Could not detect object", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255))

        cv2.imshow("Tracking demo", frame_to_display)

        key = cv2.waitKey(33)
        if key == ord('q'):
            logging.warning("Escaping demo...")
            break
        
        if key == ord('r'):
            cv2.destroyAllWindows()
            logging.warning("Re-initing tracker")
            bbox = cv2.selectROI(frame, printNotice=True)
            cv2.destroyAllWindows()
            tracker = build_tracker(tracker_type)
            tracker.init(frame, bbox)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='tracking-demo')
    parser.add_argument("--data_path", help="path to a single video file or a folder containing .jpg video frames")
    parser.add_argument("--data_type", default="single", help="supported types: vot, single, uav (single is default)")
    parser.add_argument("--tracker_type", default="kcf", help="supported trackers: mil, kcf, nano, mosse, csrt, medianflow")

    args = parser.parse_args()
    print(args)

    run_demo(args.data_path, args.data_type, args.tracker_type)
