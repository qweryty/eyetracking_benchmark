import argparse

import cv2
import numpy as np
import mediapipe as mp
from pupil_detectors import Detector2D
from time import sleep
from datetime import datetime
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
from eye_processor import EyeProcessor
from config import EyeTrackCameraConfig, EyeTrackSettingsConfig

def read_video(eye_video_path, width = 1080, height=1920):
    eye_video = cv2.VideoCapture(eye_video_path)
    result = []
    while eye_video.isOpened():
        frame_number = eye_video.get(cv2.CAP_PROP_POS_FRAMES)
        fps = eye_video.get(cv2.CAP_PROP_FPS)
        ret, eye_frame = eye_video.read()
        if ret:
            eye_frame = cv2.resize(eye_frame, (width, height), interpolation=cv2.INTER_NEAREST)
            result.append((frame_number / fps, eye_frame, cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)))
        else:
            break
    eye_video.release()
    return result


def analyze_pye3d(all_frames, width=1080, height=1920):
    # create 2D detector
    detector_2d = Detector2D()
    # create pye3D detector
    camera = CameraModel(focal_length=561.5, resolution=[width, height])
    detector_3d = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)

    start_time = datetime.now()
    # read each frame of video and run pupil detectors
    for timestamp, eye_frame, grayscale_array in all_frames:
        # read video frame as numpy array
        # eye_frame = cv2.resize(eye_frame, (width, height), interpolation=cv2.INTER_NEAREST)
        grayscale_array = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        # run 2D detector on video frame
        result_2d = detector_2d.detect(grayscale_array)
        result_2d["timestamp"] = timestamp
        # pass 2D detection result to 3D detector
        result_3d = detector_3d.update_and_detect(result_2d, grayscale_array)
        
        # ellipse_3d = result_3d["ellipse"]
        # # draw 3D detection result on eye frame
        # cv2.ellipse(
        #     eye_frame,
        #     tuple(int(v) for v in ellipse_3d["center"]),
        #     tuple(int(v / 2) for v in ellipse_3d["axes"]),
        #     ellipse_3d["angle"],
        #     0,
        #     360,  # start/end angle for drawing
        #     (0, 255, 0),  # color (BGR): red
        # )
        # # show frame
        # cv2.imshow("eye_frame", eye_frame)
        # # press esc to exit
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

    processing_time = datetime.now() - start_time
    print(f"Processed {len(all_frames)} frames in {processing_time} fps:{len(all_frames)/processing_time.seconds}")

    cv2.destroyAllWindows()

def analyze_mediapipe(all_frames, width=1080, height=1920):
    mp_face_mesh = mp.solutions.face_mesh
    start_time = datetime.now()
    landmarks_count = 0
    with mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=.5,
        min_tracking_confidence=.5
    ) as face_mesh:
        for timestamp, eye_frame, grayscale_array in all_frames:
            rgb_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                landmarks_count += 1

    processing_time = datetime.now() - start_time
    print(f"Processed {len(all_frames)} frames in {processing_time} fps:{len(all_frames)/processing_time.seconds} landmarks: {landmarks_count}")
    cv2.destroyAllWindows()

def analyze_tracker(all_frames, width=1080, height=1920):
    processor = EyeProcessor(
        config=EyeTrackCameraConfig(threshold=40, roi_window_w=width, roi_window_h=height), 
        settings=EyeTrackSettingsConfig(gui_blob_fallback=False),
        eye_id="EyeId.RIGHT"
    )
    processor.run(all_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eye_video_path")
    args = parser.parse_args()

    # width = 1080
    # height = 1920
    # width = 1920
    # height = 1080
    width = 640
    height = 480
    all_frames = read_video(args.eye_video_path, width=width, height=height)
    # print("Testing pye3d")
    # for i in range(20):
    #     # analyze_video(all_frames, width=480, height=640)
    #     analyze_pye3d(all_frames, width=width, height=height)

    print("Testing tracker")
    for i in range(20):
        analyze_tracker(all_frames, width=width, height=height)

    # print("Testing mediapipe")
    # for i in range(20):
    #     # analyze_video(all_frames, width=480, height=640)
    #     analyze_mediapipe(all_frames, width=width, height=height)
