import os
import cv2
import mediapipe as mp
import urllib.request
from collections import deque
import numpy as np
from ultralytics import YOLO
import time

# Dictionarul de emotii
EMOTION_MAPPING = {
    "Happiness": ["mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft", "cheekSquintRight"],
    "Sadness": ["browInnerUp", "mouthFrownLeft", "mouthFrownRight"],
    "Surprise": ["jawOpen", "browOuterUpLeft", "browOuterUpRight", "eyeWidenedLeft", "eyeWidenedRight"],
    "Anger": ["browDownLeft", "browDownRight", "mouthPressLeft", "mouthPressRight"],
    "Disgust": ["noseSneerLeft", "noseSneerRight", "mouthUpperUpLeft", "mouthUpperUpRight"],
    "Fear": ["mouthStretchLeft", "mouthStretchRight", "eyeWidenedLeft", "eyeWidenedRight", "browInnerUp"],
    "Contempt": ["mouthDimpleLeft", "mouthDimpleRight", "mouthLeft", "mouthRight"]
}

# Setari
BUFFER_SIZE = 30
POSE_VELOCITY_THRESH = 3
AU_VELOCITY_THRESH = 0.15


def porneste_detectia(model_path="best.pt"):
    # Verificam si descarcam landmarker-ul daca lipseste
    if not os.path.exists("face_landmarker.task"):
        print("Fisierul face_landmarker lipseste. Se descarca...")
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        urllib.request.urlretrieve(model_url, "face_landmarker.task")
        print("Downloaded face_landmarker succesfully!")

    blendshape_buffer = deque(maxlen=BUFFER_SIZE)
    last_pose = None

    if os.path.exists(model_path):
        print(f"\nModel găsit cu succes la: {model_path}")
        custom_model = YOLO(model_path)

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=VisionRunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True)

        with FaceLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(0)  # 0 pentru webcam-ul laptopului
            # Pentru fps
            prev_frame_time = 0
            new_frame_time = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                img_h, img_w = frame.shape[:2]

                YOLO_results = custom_model.predict(source=frame, save=False, conf=0.5, verbose=False)
                face_found = False

                for result in YOLO_results:
                    boxes = result.boxes
                    face_found = True
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        w, h = x2 - x1, y2 - y1
                        pad_x = w * 0.1
                        x1_nou = max(0, int(x1 - pad_x))
                        x2_nou = min(img_w, int(x2 + pad_x))
                        y2_nou = int(y2)
                        y1_nou = int(y1)
                        raport = w / h

                        if raport > 0.9:
                            offset_frunte = h * 0.30
                            y1_nou = max(0, int(y1 - offset_frunte))
                            color = (255, 0, 255)
                            label = f"Ajustat sus + latit: W/H {raport:.2f}"
                        else:
                            color = (0, 255, 0)
                            label = f"Doar latit: W/H {raport:.2f}"

                        cv2.rectangle(frame, (x1_nou, y1_nou), (x2_nou, y2_nou), color, 2)

                        cv2.putText(frame, label, (x1_nou, max(10, y1_nou - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


                        img_crop = frame[y1_nou:y2_nou, x1_nou:x2_nou]
                        if img_crop.size == 0:
                            continue

                        img_crop_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_crop_rgb)
                        MP_results = landmarker.detect(mp_img)
                        if MP_results.face_landmarks:
                          landmarks = MP_results.face_landmarks[0]
                          crop_h, crop_w, _ = img_crop.shape
                          for landmark in landmarks:
                            # Punem offset pentru a desena landmark-erii pe imaginea originala, nu pe cea cropped (e comentariul MEU nu al lui Gemini)
                            x = x1_nou + int(landmark.x * crop_w)
                            y = y1_nou + int(landmark.y * crop_h)
                            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

                        if not MP_results.face_blendshapes:
                            continue

                        is_moving_too_fast = False
                        if MP_results.facial_transformation_matrixes:
                            matrix = MP_results.facial_transformation_matrixes[0][:3, :3]
                            euler_angles, _, _, _, _, _, = cv2.RQDecomp3x3(matrix)
                            current_pose = np.array(euler_angles)

                            if last_pose is not None:
                                pose_velocity = np.abs(current_pose - last_pose)
                                if np.any(pose_velocity > POSE_VELOCITY_THRESH):
                                    is_moving_too_fast = True
                                    cv2.putText(frame, "MOTION GATE ACTIVE", (x1_nou, max(15, y1_nou - 30)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            last_pose = current_pose

                        current_aus = np.array([b.score for b in MP_results.face_blendshapes[0]])
                        au_names = [b.category_name for b in MP_results.face_blendshapes[0]]

                        if not is_moving_too_fast:
                            if len(blendshape_buffer) == BUFFER_SIZE:
                                baseline_aus = np.mean(blendshape_buffer, axis=0)
                                au_velocity = current_aus - baseline_aus
                                spike_indices = np.nonzero(au_velocity > AU_VELOCITY_THRESH)[0]
                                y_offset = y1_nou + 20

                                for idx in spike_indices:
                                    if au_names[idx] == "_neutral": continue
                                    detected_emotions = [emotion for emotion, blendshapes in EMOTION_MAPPING.items() if
                                                         au_names[idx] in blendshapes]

                                    if detected_emotions:
                                        emotion_str = ",".join(detected_emotions)
                                        cv2.putText(frame, f"{emotion_str} ({au_names[idx]})", (x2_nou + 10, y_offset),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                    else:
                                        cv2.putText(frame, f"Spike: {au_names[idx]}", (x2_nou + 10, y_offset),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    y_offset += 25
                            blendshape_buffer.append(current_aus)
                        break  # Procesam doar prima fata
                # Calculam si afisam fps-ul
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps = int(fps)
                fps = str(fps)
                cv2.putText(frame, f"{fps} FPS", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)    
                if not face_found:
                    blendshape_buffer.clear()
                    last_pose = None

                cv2.imshow("Micro-Expression Detector", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:  # ESC pentru a iesi
                    break
            cap.release()
            cv2.destroyAllWindows()
    else:
        print(f"Eroare: Nu am găsit modelul la calea: {model_path}")
        
