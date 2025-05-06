import cv2
import logging
from pathlib import Path
import argparse

from trackers.kcf import KCFTracker, KCFParams
from utils import get_cv2_pattern_from_folder


def build_tracker(tracker_name: str):
    if tracker_name == "kcf":
        return KCFTracker(KCFParams(debug=False))
    elif tracker_name == "nano":
        params = cv2.TrackerNano_Params()
        params.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        params.target = cv2.dnn.DNN_TARGET_CPU
        params.backbone = "../network_weights/nano/nanotrack_backbone_sim.onnx"
        params.neckhead = "../network_weights/nano/nanotrack_head_sim.onnx"
        return cv2.TrackerNano_create(params)
    elif tracker_name == "tld":
        return cv2.legacy.TrackerTLD_create()
    else:
        raise ValueError(f"Could not create tracker with name {tracker_name}")


def build_video_capture(video_path):
    if video_path is None:
        return cv2.VideoCapture(0)

    video_root_folder = Path(video_path)

    if not video_root_folder.exists():
        raise ValueError(f"Could not find file or folder named {video_path}")

    if video_root_folder.is_file():
        return cv2.VideoCapture(video_root_folder)

    image_name_pattern = get_cv2_pattern_from_folder(video_root_folder)
    return cv2.VideoCapture(video_root_folder / image_name_pattern)


def run_demo(data_path, tracker_type):
    capture = build_video_capture(data_path)

    if not capture.isOpened():
        raise ValueError(f"Wrong video path {data_path}")

    tracker = build_tracker(tracker_type)

    status, frame = capture.read()

    bbox = cv2.selectROI(frame, printNotice=True)
    cv2.destroyAllWindows()
    bbox_color = (255, 0, 0)
    tracker.init(frame, bbox)

    while True:
        status, frame = capture.read()

        if not status:
            logging.warning("Video sequency ends, ending program...")
            cv2.destroyAllWindows()
            return

        tracking_result, new_bbox = tracker.update(frame)
        print(tracking_result)

        x, y, w, h = map(int, new_bbox)

        if tracking_result:
            print("detected object")
            frame_to_display = cv2.rectangle(
                frame, (x, y), (x + w, y + h), bbox_color, 2, 1
            )
        else:
            print("could not detect object")
            frame_to_display = cv2.putText(
                frame,
                "Could not detect object",
                (100, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
            )

        cv2.imshow("Tracking demo", frame_to_display)

        key = cv2.waitKey(33)
        if key == ord("q"):
            logging.warning("Escaping demo...")
            break

        if key == ord("r"):
            cv2.destroyAllWindows()
            logging.warning("Re-initing tracker")
            bbox = cv2.selectROI(frame, printNotice=True)
            cv2.destroyAllWindows()
            tracker = build_tracker(tracker_type)
            tracker.init(frame, bbox)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="tracking-demo")
    parser.add_argument(
        "--data_path",
        help="path to a single video file or a folder containing .jpg video frames",
    )
    parser.add_argument(
        "--tracker_type",
        default="kcf",
        help="supported trackers: kcf, nano, tld",
    )

    args = parser.parse_args()

    run_demo(args.data_path, args.tracker_type)
