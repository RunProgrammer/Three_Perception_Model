import pyzed.sl as sl
from ultralytics import YOLO
import numpy as np
import cv2

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("./weights/best.pt")

# -----------------------------
# Initialize ZED
# -----------------------------
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ Failed to open ZED")
    exit()

runtime = sl.RuntimeParameters()

image = sl.Mat()
depth = sl.Mat()

# Get camera intrinsics
calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
fx = calib.fx
fy = calib.fy
cx_cam = calib.cx
cy_cam = calib.cy

print("🎥 Starting detection... Press Q to quit")

while True:

    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        frame = image.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = model(frame, conf=0.5)

        for r in results:
            for box, cls, conf in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy()
            ):
                x1, y1, x2, y2 = map(int, box)

                # Pixel center
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                # Get depth at center pixel
                depth_value = depth.get_value(u, v)[1]

                if np.isnan(depth_value) or depth_value <= 0:
                    continue

                # -----------------------------
                # 3D Position in CAMERA OPTICAL FRAME
                # -----------------------------
                X_opt = (u - cx_cam) * depth_value / fx
                Y_opt = (v - cy_cam) * depth_value / fy
                Z_opt = depth_value

                # -----------------------------
                # Convert Optical → ROS camera_link frame
                #
                # Optical frame:
                #   X right
                #   Y down
                #   Z forward
                #
                # ROS camera_link:
                #   X forward
                #   Y left
                #   Z up
                #
                # Conversion:
                #   X_ros = Z_opt
                #   Y_ros = -X_opt
                #   Z_ros = -Y_opt
                # -----------------------------

                X_ros = Z_opt
                Y_ros = -X_opt
                Z_ros = -Y_opt

                print("--------------")
                print(f"Object: {model.names[int(cls)]}")
                print(f"ROS Frame Position:")
                print(f"X: {X_ros:.3f} m")
                print(f"Y: {Y_ros:.3f} m")
                print(f"Z: {Z_ros:.3f} m")

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.circle(frame, (u,v), 5, (0,0,255), -1)

        cv2.imshow("ZED YOLO 3D", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()
cv2.destroyAllWindows()
