# ==============================================
# RGB-D YOLO Detection with Orientation
# ==============================================

CAMERA_TYPE = "realsense"   # "realsense" or "zed"

from ultralytics import YOLO
import numpy as np
import cv2
import math

model = YOLO("./weights/best.pt")

# ---------------- ORIENTATION FUNCTIONS ----------------

def estimate_yaw_from_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(cnt)
    angle = rect[2]

    if angle < -45:
        angle = 90 + angle

    yaw = np.deg2rad(angle)
    return yaw


def euler_to_quaternion(roll, pitch, yaw):
    qx = math.sin(roll/2)*math.cos(pitch/2)*math.cos(yaw/2) - math.cos(roll/2)*math.sin(pitch/2)*math.sin(yaw/2)
    qy = math.cos(roll/2)*math.sin(pitch/2)*math.cos(yaw/2) + math.sin(roll/2)*math.cos(pitch/2)*math.sin(yaw/2)
    qz = math.cos(roll/2)*math.cos(pitch/2)*math.sin(yaw/2) - math.sin(roll/2)*math.sin(pitch/2)*math.cos(yaw/2)
    qw = math.cos(roll/2)*math.cos(pitch/2)*math.cos(yaw/2) + math.sin(roll/2)*math.sin(pitch/2)*math.sin(yaw/2)
    return qx, qy, qz, qw


# ==============================================
# REALSENSE SETUP
# ==============================================
if CAMERA_TYPE == "realsense":
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy = intr.fx, intr.fy
    cx_cam, cy_cam = intr.ppx, intr.ppy


# ==============================================
# ZED SETUP
# ==============================================
if CAMERA_TYPE == "zed":
    import pyzed.sl as sl

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ ZED failed")
        exit()

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    fx = calib.fx
    fy = calib.fy
    cx_cam = calib.cx
    cy_cam = calib.cy


print("🎥 Running Detection with Orientation...")

while True:

    # -------- REALSENSE --------
    if CAMERA_TYPE == "realsense":
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

    # -------- ZED --------
    if CAMERA_TYPE == "zed":
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        frame = image.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    results = model(frame, conf=0.45)

    for r in results:
        for box, cls, conf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy()
        ):

            x1, y1, x2, y2 = map(int, box)
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)

            # -------- Depth --------
            if CAMERA_TYPE == "realsense":
                depth_value = depth_image[v, u] * depth_scale

            if CAMERA_TYPE == "zed":
                depth_value = depth.get_value(u, v)[1]

            if depth_value <= 0 or np.isnan(depth_value):
                continue

            # -------- 3D Position --------
            X = (u - cx_cam) * depth_value / fx
            Y = (v - cy_cam) * depth_value / fy
            Z = depth_value

            # Optical → ROS
            X_ros = Z
            Y_ros = -X
            Z_ros = -Y

            # -------- Orientation --------
            roi = frame[y1:y2, x1:x2]
            yaw = estimate_yaw_from_roi(roi)

            # Overhead camera:
            roll = math.pi    # gripper facing downward
            pitch = 0

            qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)

            print("--------------")
            print(f"Object: {model.names[int(cls)]}")
            print(f"Position (ROS frame): X={X_ros:.3f}, Y={Y_ros:.3f}, Z={Z_ros:.3f}")
            print(f"Orientation (quat): [{qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}]")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (u,v), 5, (0,0,255), -1)

    cv2.imshow("RGB-D YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


if CAMERA_TYPE == "realsense":
    pipeline.stop()

if CAMERA_TYPE == "zed":
    zed.close()

cv2.destroyAllWindows()
