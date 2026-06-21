import os
import cv2
import torch
from ultralytics import YOLO


# ============================================================
# 사용자 설정
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "assets", "models", "yolo26n.pt")

CAMERA_DEVICE = "/dev/video0"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

IMAGE_SIZE = 1024
CONFIDENCE = 0.45

# COCO 데이터셋의 person 클래스 번호
PERSON_CLASS_ID = 0

# Jetson GPU 번호
GPU_DEVICE = 0


# ============================================================
# USB 카메라 열기
# ============================================================

def open_usb_camera():
    print(f"[INFO] USB 카메라를 실행합니다: {CAMERA_DEVICE}")

    camera = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return camera


# ============================================================
# GPU 확인
# ============================================================

def check_gpu():
    print(f"[INFO] PyTorch 버전: {torch.__version__}")
    print(f"[INFO] CUDA 사용 가능 여부: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU를 사용할 수 없습니다.")
        print("[ERROR] 현재 PyTorch가 CPU 전용 버전일 가능성이 큽니다.")
        print("[INFO] GPU용 PyTorch를 먼저 설치해야 합니다.")
        return False

    gpu_name = torch.cuda.get_device_name(GPU_DEVICE)

    print(f"[INFO] 사용 GPU: {gpu_name}")
    print(f"[INFO] CUDA 버전: {torch.version.cuda}")

    return True


# ============================================================
# 메인 함수
# ============================================================

def main():
    # GPU가 없으면 프로그램 종료
    if not check_gpu():
        return

    print("[INFO] YOLO26n 모델을 GPU로 불러옵니다.")

    try:
        model = YOLO(MODEL_PATH)

    except Exception as error:
        print("[ERROR] YOLO 모델을 불러오지 못했습니다.")
        print(f"[ERROR] 상세 내용: {error}")
        return

    cap = open_usb_camera()

    if not cap.isOpened():
        print(f"[ERROR] {CAMERA_DEVICE} 카메라를 열 수 없습니다.")
        print("[INFO] /dev/video1도 확인해보세요.")
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("[INFO] USB 카메라 연결 성공")
    print(f"[INFO] 해상도: {actual_width} x {actual_height}")
    print(f"[INFO] 카메라 FPS: {actual_fps}")
    print("[INFO] GPU 사람 탐지를 시작합니다.")
    print("[INFO] 종료: q 또는 ESC")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[ERROR] 카메라 프레임을 읽지 못했습니다.")
            break

        try:
            results = model.predict(
                source=frame,

                # GPU 0번 사용
                device=GPU_DEVICE,

                # 사람 클래스만 탐지
                classes=[PERSON_CLASS_ID],

                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,

                # FP16 연산 사용
                half=True,

                verbose=False
            )

        except Exception as error:
            print("[ERROR] GPU 추론 중 오류가 발생했습니다.")
            print(f"[ERROR] 상세 내용: {error}")
            break

        result = results[0]
        output_frame = result.plot()

        person_count = len(result.boxes)

        frame_height, frame_width = frame.shape[:2]

        screen_center_x = frame_width // 2
        screen_center_y = frame_height // 2

        # 화면 중심
        cv2.circle(
            output_frame,
            (screen_center_x, screen_center_y),
            6,
            (255, 0, 0),
            -1
        )

        cv2.putText(
            output_frame,
            f"Person: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output_frame,
            "Device: GPU",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if person_count > 0:
            confidences = result.boxes.conf.cpu().numpy()

            # 신뢰도가 가장 높은 사람 선택
            best_index = confidences.argmax()

            box = result.boxes.xyxy[best_index].cpu().numpy()
            confidence = float(result.boxes.conf[best_index].cpu())

            x1, y1, x2, y2 = map(int, box)

            person_center_x = (x1 + x2) // 2
            person_center_y = (y1 + y2) // 2

            error_x = person_center_x - screen_center_x
            error_y = person_center_y - screen_center_y

            cv2.circle(
                output_frame,
                (person_center_x, person_center_y),
                7,
                (0, 0, 255),
                -1
            )

            cv2.line(
                output_frame,
                (screen_center_x, screen_center_y),
                (person_center_x, person_center_y),
                (0, 255, 255),
                2
            )

            cv2.putText(
                output_frame,
                f"Target: ({person_center_x}, {person_center_y})",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                output_frame,
                f"Error: ({error_x}, {error_y})",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                output_frame,
                f"Confidence: {confidence:.2f}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

        cv2.imshow(
            "YOLO26n GPU Person Detection",
            output_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("[INFO] 프로그램을 종료했습니다.")


if __name__ == "__main__":
    main()
