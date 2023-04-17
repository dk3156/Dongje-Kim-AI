
import cv2
import random
import numpy as np
from object_detection import ObjectDetection
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("ball.mp4")

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
video_writer = cv2.VideoWriter("./output/ball_output.avi", cv2.VideoWriter_fourcc('P','I','M','1'), fps, (width, height))

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

initial_error = np.array([10.0, 10.0])
motion_noise = np.array([20.0, 20.0])
measurement_noise = 200
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([[907.], [0.]])
kf.F = np.array([[1., 1.], [0., 1.]])
kf.H = np.array([[1., 0.]])
kf.P *= 1000.
kf.R = 5
kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
print(kf.x)

prev_box = [855, 233, 105, 85]
diff = 0
while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Point current frame
    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    if len(boxes) == 0:
        [x, y, w, h] = prev_box
        kf.predict()
        kf.update(x)
        x = int(kf.x[0, 0])
        prev_box[0] = x
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        for box in boxes:
            (x, y, w, h) = box
            prev_box = box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_cur_frame.append((cx, cy))
            #print("FRAME N°", count, " ", x, y, w, h)

            # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        # cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    video_writer.write(frame)
    
    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

video_writer.release()

