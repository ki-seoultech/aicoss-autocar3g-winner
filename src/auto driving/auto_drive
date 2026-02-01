from autocar3g.absclient import AbstractPopClient
from autocar3g import AI
from autocar3g.camera import Camera
from autocar3g.driving import Driving
import cv2, time, math

# ===== IP 설정 =====
CAR_IP = "192.168.248.247"
AbstractPopClient.BROKER_DOMAIN = CAR_IP
Camera.SERVER_IP = CAR_IP

cam = Camera()
cam.start()

car = Driving()
throttle = 1
steer_limit = 0.9

TF = AI.Track_Follow_TF(cam)
TF.load_model("Track_Model.h5")

print("Running... Press 'q' on the video window / Ctrl+C")

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

try:
    while True:
        car.throttle = throttle

        # 화면 표시용 원본 프레임
        frame = cam.read()

        # 추론
        ret_tf = TF.run()

        steer = 0.0
        x = None

        if isinstance(ret_tf, dict) and ("x" in ret_tf):
            x = float(ret_tf["x"])
            if steer > 0.5:
                steer = (x - 0.5) * 2.6
            else:
                steer = (x - 0.5) * 3

            if steer > steer_limit: steer = steer_limit
            if steer < -steer_limit: steer = -steer_limit

            car.steering = steer
        else:
            car.steering = 0.0

        # 화면에 그리기
        if frame is not None:
            H, W = frame.shape[:2]

            # 1) 중앙 기준선 표시
            cv2.line(frame, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)

            # 2) 예측 점 표시
            if x is not None:
                px = int(x * W)
                py = int(H * 0.75)  # 점을 찍을 높이(원하는 위치로 조절)

                # 점(원) 그리기
                cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)

                # 텍스트 오버레이
                cv2.putText(frame, f"x={x:.3f} steer={steer:+.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "ret_tf invalid -> steer=0", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("AutoCar Camera", frame)

            # q로 종료(창 포커스 필요)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        time.sleep(0.02)

except KeyboardInterrupt:
    pass
finally:
    car.throttle = 0
    car.steering = 0
    try:
        cam.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("stopped safely.")
