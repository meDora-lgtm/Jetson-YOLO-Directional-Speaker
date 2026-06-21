import time
import cv2

from ultralytics import YOLO
from adafruit_extended_bus import ExtendedI2C
from adafruit_servokit import ServoKit
from adafruit_bus_device import i2c_device


# ============================================================
# 사용자 설정
# ============================================================

# ------------------------------------------------------------
# YOLO 모델 설정
# ------------------------------------------------------------

MODEL_PATH = "yolo26n.engine"

INFERENCE_SIZE = 960

# 탐지 신뢰도
# 낮추면 컵을 더 잘 찾지만 오탐지가 증가할 수 있습니다.
CONFIDENCE = 0.30

# COCO 데이터셋 클래스
TARGET_CLASS_ID = 41
TARGET_CLASS_NAME = "cup"


# ------------------------------------------------------------
# 카메라 설정
# ------------------------------------------------------------

CAMERA_DEVICE = "/dev/video0"

# 카메라 화면은 HD로 받아서 표시합니다.
# YOLO에는 내부적으로 640×640 크기로 변환되어 들어갑니다.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# USB 카메라에서 HD 30FPS를 위해 MJPG 사용
USE_MJPG = True

# 카메라 버퍼 크기
# 낮을수록 화면 지연이 감소합니다.
CAMERA_BUFFER_SIZE = 1


# ------------------------------------------------------------
# PCA9685 및 서보 설정
# ------------------------------------------------------------

I2C_BUS_NUMBER = 1
PCA9685_ADDRESS = 0x41
SERVO_CHANNEL = 0

# 서보 PWM 범위
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500

# 물리적인 서보 안전 범위
#
# 125도 미만:
# 장치의 턱에 걸리므로 이동하지 않음
#
# 180도 초과:
# 장치가 너무 높아 정상 작동하지 않으므로 이동하지 않음
MIN_ANGLE = 125.0
MAX_ANGLE = 180.0

# 프로그램 시작 및 컵 분실 후 복귀할 위치
INITIAL_ANGLE = 125.0

# 실제 장치의 움직임 방향
#
# -1:
# 화면에서 컵이 위에 있으면 각도가 증가하도록 제어
#
# 1:
# 화면에서 컵이 위에 있으면 각도가 감소하도록 제어
#
# 현재 방향이 반대로 움직이면 -1과 1을 서로 바꾸십시오.
SERVO_DIRECTION = -1.0


# ------------------------------------------------------------
# 화면 중앙 제어 범위
# ------------------------------------------------------------

# 컵 중심이 화면 중앙에서 이 범위 안에 있으면 정지
DEAD_ZONE_Y = 30

# 중앙에 가까워질수록 감속하는 범위
SLOW_ZONE_Y = 120


# ------------------------------------------------------------
# PD 제어 설정
# ------------------------------------------------------------

# P 게인
#
# 오차가 클수록 서보 속도를 크게 만드는 값입니다.
# 값을 올리면 반응이 빨라지지만 진동할 수 있습니다.
PD_KP = 0.065

# D 게인
#
# 오차 변화 속도에 반응하여 중앙을 지나치는 현상을 줄입니다.
# 너무 높으면 탐지 좌표 노이즈에 민감해질 수 있습니다.
PD_KD = 0.012

# D항 저역통과 필터
#
# 0에 가까움:
# 매우 부드럽지만 반응이 느림
#
# 1에 가까움:
# 빠르게 반응하지만 좌표 노이즈에 민감함
DERIVATIVE_FILTER_ALPHA = 0.25

# D항이 갑자기 커지는 것을 방지하는 제한값
# 단위: degree/second
MAX_D_TERM = 8.0


# ------------------------------------------------------------
# 서보 움직임 설정
# ------------------------------------------------------------

# 추적 중 최대 서보 속도
# 단위: degree/second
MAX_SERVO_SPEED = 20.0

# 추적 중 최대 가속도
# 단위: degree/second²
MAX_SERVO_ACCELERATION = 65.0

# 서보 명령 최소 간격
SERVO_UPDATE_INTERVAL = 0.015

# 중앙에 들어왔을 때 남은 속도를 줄이는 비율
#
# 작을수록 빠르게 정지합니다.
CENTER_VELOCITY_DAMPING = 0.55


# ------------------------------------------------------------
# 컵 좌표 필터 설정
# ------------------------------------------------------------

# 컵 중심 좌표 필터
#
# 낮을수록:
# 부드럽지만 반응이 느림
#
# 높을수록:
# 반응은 빠르지만 흔들림이 커질 수 있음
TARGET_FILTER_ALPHA = 0.35


# ------------------------------------------------------------
# 컵 분실 및 초기 위치 복귀 설정
# ------------------------------------------------------------

# 컵이 사라진 뒤 초기 위치 복귀를 시작하는 시간
LOST_RETURN_TIME = 2.0

# 초기 위치 복귀 최대 속도
RETURN_MAX_SPEED = 14.0

# 초기 위치 복귀 최대 가속도
RETURN_ACCELERATION = 45.0

# 초기 위치 복귀 서보 명령 간격
RETURN_INTERVAL = 0.015

# 초기 위치에 도착했다고 판단할 각도 오차
RETURN_ANGLE_TOLERANCE = 0.08

# 복귀 완료로 판단할 속도
RETURN_VELOCITY_TOLERANCE = 0.35


# ============================================================
# 사용자 설정 검증
# ============================================================

def validate_settings():
    """잘못된 사용자 설정을 실행 전에 검사합니다."""

    if MIN_ANGLE >= MAX_ANGLE:
        raise ValueError(
            f"MIN_ANGLE({MIN_ANGLE})은 "
            f"MAX_ANGLE({MAX_ANGLE})보다 작아야 합니다."
        )

    if not MIN_ANGLE <= INITIAL_ANGLE <= MAX_ANGLE:
        raise ValueError(
            f"INITIAL_ANGLE({INITIAL_ANGLE})은 "
            f"{MIN_ANGLE}~{MAX_ANGLE}도 범위 안이어야 합니다."
        )

    if INFERENCE_SIZE <= 0:
        raise ValueError(
            "INFERENCE_SIZE는 0보다 커야 합니다."
        )

    if not 0.0 <= CONFIDENCE <= 1.0:
        raise ValueError(
            "CONFIDENCE는 0.0~1.0 범위여야 합니다."
        )

    if DEAD_ZONE_Y < 0:
        raise ValueError(
            "DEAD_ZONE_Y는 0 이상이어야 합니다."
        )

    if SLOW_ZONE_Y <= DEAD_ZONE_Y:
        raise ValueError(
            "SLOW_ZONE_Y는 DEAD_ZONE_Y보다 커야 합니다."
        )

    if SERVO_DIRECTION not in (-1.0, 1.0):
        raise ValueError(
            "SERVO_DIRECTION은 -1 또는 1이어야 합니다."
        )


validate_settings()


# ============================================================
# 현재 사용자 설정 출력
# ============================================================

def print_current_settings():
    """코드의 실제 변수 값을 이용하여 설정 정보를 출력합니다."""

    direction_text = (
        "반전(-1)"
        if SERVO_DIRECTION == -1
        else "정방향(+1)"
    )

    print("")
    print("=" * 62)
    print("[현재 사용자 설정]")
    print("=" * 62)

    print(f"YOLO 모델               : {MODEL_PATH}")
    print(f"YOLO 작업                : detect")
    print(f"탐지 대상                : {TARGET_CLASS_NAME}")
    print(f"탐지 클래스 ID           : {TARGET_CLASS_ID}")
    print(f"탐지 신뢰도              : {CONFIDENCE:.2f}")
    print(
        f"YOLO 추론 크기           : "
        f"{INFERENCE_SIZE} x {INFERENCE_SIZE}"
    )

    print("-" * 62)

    print(f"카메라 장치              : {CAMERA_DEVICE}")
    print(
        f"요청 카메라 해상도       : "
        f"{CAMERA_WIDTH} x {CAMERA_HEIGHT}"
    )
    print(f"요청 카메라 FPS          : {CAMERA_FPS}")
    print(f"MJPG 사용                : {USE_MJPG}")
    print(f"카메라 버퍼              : {CAMERA_BUFFER_SIZE}")

    print("-" * 62)

    print(f"I2C 버스                 : {I2C_BUS_NUMBER}")
    print(f"PCA9685 주소             : 0x{PCA9685_ADDRESS:02X}")
    print(f"서보 채널                : {SERVO_CHANNEL}")
    print(
        f"서보 PWM 범위            : "
        f"{SERVO_MIN_PULSE}~{SERVO_MAX_PULSE} us"
    )
    print(
        f"서보 안전 범위           : "
        f"{MIN_ANGLE:.1f}~{MAX_ANGLE:.1f}도"
    )
    print(f"서보 초기 위치           : {INITIAL_ANGLE:.1f}도")
    print(f"서보 방향                : {direction_text}")

    print("-" * 62)

    print(f"중앙 데드존              : ±{DEAD_ZONE_Y}px")
    print(f"중앙 감속 범위           : ±{SLOW_ZONE_Y}px")
    print(f"P 게인                   : {PD_KP}")
    print(f"D 게인                   : {PD_KD}")
    print(f"D 필터                   : {DERIVATIVE_FILTER_ALPHA}")
    print(f"D항 최대 제한            : ±{MAX_D_TERM} deg/s")
    print(f"서보 최대 속도           : {MAX_SERVO_SPEED} deg/s")
    print(
        f"서보 최대 가속도         : "
        f"{MAX_SERVO_ACCELERATION} deg/s²"
    )
    print(f"컵 좌표 필터             : {TARGET_FILTER_ALPHA}")
    print(f"컵 분실 복귀 시간        : {LOST_RETURN_TIME}초")

    print("=" * 62)
    print("")


print_current_settings()


# ============================================================
# I2C 장치 탐지 우회
# ============================================================

original_init = i2c_device.I2CDevice.__init__


def no_probe_init(self, i2c, device_address, probe=True):
    original_init(
        self,
        i2c,
        device_address,
        probe=False
    )


i2c_device.I2CDevice.__init__ = no_probe_init


# ============================================================
# PCA9685 초기화
# ============================================================

print("[INFO] PCA9685와 서보를 초기화합니다.")

i2c = ExtendedI2C(I2C_BUS_NUMBER)

kit = ServoKit(
    channels=16,
    i2c=i2c,
    address=PCA9685_ADDRESS
)

kit.servo[SERVO_CHANNEL].set_pulse_width_range(
    SERVO_MIN_PULSE,
    SERVO_MAX_PULSE
)


# ============================================================
# YOLO 모델 불러오기
# ============================================================

print(f"[INFO] YOLO 모델을 불러옵니다: {MODEL_PATH}")

# task를 명시하여
# "Unable to automatically guess model task" 경고를 줄입니다.
model = YOLO(
    MODEL_PATH,
    task="detect"
)

print(
    f"[INFO] 탐지 대상: "
    f"{TARGET_CLASS_NAME} (class {TARGET_CLASS_ID})"
)

print(
    f"[INFO] YOLO 추론 크기: "
    f"{INFERENCE_SIZE} x {INFERENCE_SIZE}"
)


# ============================================================
# 카메라 열기
# ============================================================

print(f"[INFO] 카메라를 실행합니다: {CAMERA_DEVICE}")

camera = cv2.VideoCapture(
    CAMERA_DEVICE,
    cv2.CAP_V4L2
)

if USE_MJPG:
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

camera.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)

camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    CAMERA_BUFFER_SIZE
)

if not camera.isOpened():
    raise RuntimeError(
        f"카메라를 열 수 없습니다: {CAMERA_DEVICE}"
    )


actual_width = int(
    camera.get(cv2.CAP_PROP_FRAME_WIDTH)
)

actual_height = int(
    camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

actual_fps = camera.get(
    cv2.CAP_PROP_FPS
)

actual_fourcc_value = int(
    camera.get(cv2.CAP_PROP_FOURCC)
)

actual_fourcc = "".join(
    chr((actual_fourcc_value >> (8 * i)) & 0xFF)
    for i in range(4)
)

print(
    f"[INFO] 실제 카메라 설정: "
    f"{actual_width} x {actual_height}, "
    f"{actual_fps:.1f} FPS, "
    f"FOURCC={actual_fourcc}"
)


# ============================================================
# 공통 함수
# ============================================================

def clamp(value, minimum, maximum):
    """값을 지정한 최소·최대 범위로 제한합니다."""

    return max(
        minimum,
        min(float(value), maximum)
    )


def clamp_angle(angle):
    """서보 각도를 안전 범위로 제한합니다."""

    return clamp(
        angle,
        MIN_ANGLE,
        MAX_ANGLE
    )


def set_servo_angle_safe(angle):
    """서보에 전달하기 전에 각도를 안전 범위로 제한합니다."""

    safe_angle = clamp_angle(angle)

    kit.servo[SERVO_CHANNEL].angle = safe_angle

    return safe_angle


def update_velocity_smoothly(
    current_velocity,
    target_velocity,
    max_acceleration,
    delta_time
):
    """
    가속도를 제한하여 현재 속도를 목표 속도로
    부드럽게 변화시킵니다.
    """

    velocity_difference = (
        target_velocity - current_velocity
    )

    max_velocity_change = (
        max_acceleration * delta_time
    )

    velocity_change = clamp(
        velocity_difference,
        -max_velocity_change,
        max_velocity_change
    )

    return current_velocity + velocity_change


def select_largest_target(boxes):
    """
    탐지된 대상 중 화면에서 가장 크게 보이는
    대상 하나를 선택합니다.
    """

    if boxes is None:
        return None

    selected_target = None
    largest_area = 0.0

    for box in boxes:

        class_id = int(
            box.cls[0].item()
        )

        if class_id != TARGET_CLASS_ID:
            continue

        confidence = float(
            box.conf[0].item()
        )

        x1, y1, x2, y2 = (
            box.xyxy[0].cpu().tolist()
        )

        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height

        if area > largest_area:

            largest_area = area

            selected_target = (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                confidence
            )

    return selected_target


# ============================================================
# 서보 초기 위치 설정
# ============================================================

current_angle = set_servo_angle_safe(
    INITIAL_ANGLE
)

print(
    f"[INFO] 서보 초기 위치: "
    f"{current_angle:.1f}도"
)

time.sleep(1.0)


# ============================================================
# 상태 변수
# ============================================================

current_time = time.monotonic()

last_detection_time = current_time
last_servo_update_time = current_time

returning_to_initial = False
servo_limit_status = None

# 현재 서보 속도
servo_velocity = 0.0

# 필터링된 컵 중심 Y 좌표
filtered_target_y = None

# PD 제어용 이전 오차
previous_error_y = None

# 필터링된 오차 변화율
filtered_error_derivative = 0.0

# FPS 계산
fps = 0.0
fps_frame_count = 0
fps_measure_start = time.monotonic()


# ============================================================
# 메인 루프
# ============================================================

try:

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "[WARNING] 카메라 프레임을 "
                "읽지 못했습니다."
            )

            time.sleep(0.05)
            continue


        frame_height, frame_width = frame.shape[:2]

        screen_center_x = frame_width // 2
        screen_center_y = frame_height // 2

        current_time = time.monotonic()

        delta_time = (
            current_time - last_servo_update_time
        )

        delta_time = clamp(
            delta_time,
            0.001,
            0.1
        )


        # ----------------------------------------------------
        # FPS 계산
        # ----------------------------------------------------

        fps_frame_count += 1

        fps_elapsed = (
            current_time - fps_measure_start
        )

        if fps_elapsed >= 1.0:

            fps = (
                fps_frame_count / fps_elapsed
            )

            fps_frame_count = 0
            fps_measure_start = current_time


        # ----------------------------------------------------
        # YOLO 대상 탐지
        # ----------------------------------------------------

        try:

            results = model.predict(
                source=frame,
                imgsz=INFERENCE_SIZE,
                conf=CONFIDENCE,
                classes=[TARGET_CLASS_ID],
                verbose=False
            )

        except AssertionError as error:

            error_text = str(error)

            if "input size" in error_text:

                raise RuntimeError(
                    "\nTensorRT 엔진 입력 크기와 "
                    "INFERENCE_SIZE가 일치하지 않습니다.\n"
                    f"현재 INFERENCE_SIZE: {INFERENCE_SIZE}\n"
                    f"원본 오류: {error_text}\n\n"
                    "현재 엔진이 640×640 고정 엔진이라면 "
                    "INFERENCE_SIZE=640으로 설정하십시오."
                ) from error

            raise


        target = select_largest_target(
            results[0].boxes
        )


        # ----------------------------------------------------
        # 화면 기준선 표시
        # ----------------------------------------------------

        cv2.line(
            frame,
            (0, screen_center_y),
            (frame_width, screen_center_y),
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (0, screen_center_y - DEAD_ZONE_Y),
            (frame_width, screen_center_y - DEAD_ZONE_Y),
            (100, 100, 100),
            1
        )

        cv2.line(
            frame,
            (0, screen_center_y + DEAD_ZONE_Y),
            (frame_width, screen_center_y + DEAD_ZONE_Y),
            (100, 100, 100),
            1
        )

        cv2.circle(
            frame,
            (screen_center_x, screen_center_y),
            7,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # 대상 탐지 성공
        # ----------------------------------------------------

        if target is not None:

            # 복귀 중 대상이 다시 나타나면 즉시 추적 재개
            if returning_to_initial:

                returning_to_initial = False
                servo_velocity = 0.0

                previous_error_y = None
                filtered_error_derivative = 0.0

                print(
                    f"[INFO] 복귀 중 {TARGET_CLASS_NAME} 재탐지 - "
                    "복귀 취소 후 추적 재시작"
                )


            x1, y1, x2, y2, confidence = target

            target_center_x = (
                x1 + x2
            ) // 2

            raw_target_center_y = (
                y1 + y2
            ) / 2.0


            # ------------------------------------------------
            # 대상 중심 좌표 필터링
            # ------------------------------------------------

            if filtered_target_y is None:

                filtered_target_y = (
                    raw_target_center_y
                )

            else:

                filtered_target_y = (
                    TARGET_FILTER_ALPHA
                    * raw_target_center_y
                    + (
                        1.0
                        - TARGET_FILTER_ALPHA
                    )
                    * filtered_target_y
                )


            filtered_center_y_int = int(
                filtered_target_y
            )

            # 양수: 대상이 화면 중앙보다 아래
            # 음수: 대상이 화면 중앙보다 위
            error_y = (
                filtered_target_y
                - screen_center_y
            )

            last_detection_time = current_time


            # ------------------------------------------------
            # 대상 표시
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # 원본 중심점
            cv2.circle(
                frame,
                (
                    target_center_x,
                    int(raw_target_center_y)
                ),
                6,
                (0, 0, 255),
                -1
            )

            # 필터링된 중심점
            cv2.circle(
                frame,
                (
                    target_center_x,
                    filtered_center_y_int
                ),
                8,
                (255, 0, 255),
                2
            )

            cv2.line(
                frame,
                (
                    screen_center_x,
                    screen_center_y
                ),
                (
                    target_center_x,
                    filtered_center_y_int
                ),
                (0, 255, 255),
                2
            )


            # ------------------------------------------------
            # PD 서보 제어
            # ------------------------------------------------

            p_term = 0.0
            d_term = 0.0

            if (
                current_time - last_servo_update_time
                >= SERVO_UPDATE_INTERVAL
            ):

                if abs(error_y) <= DEAD_ZONE_Y:

                    # 중앙 데드존에서는 정지
                    target_velocity = 0.0

                    servo_velocity *= (
                        CENTER_VELOCITY_DAMPING
                    )

                    if abs(servo_velocity) < 0.08:
                        servo_velocity = 0.0

                    previous_error_y = error_y
                    filtered_error_derivative = 0.0

                else:

                    # 데드존 크기를 제외한 실제 제어 오차
                    if error_y > 0:
                        effective_error = (
                            error_y - DEAD_ZONE_Y
                        )
                    else:
                        effective_error = (
                            error_y + DEAD_ZONE_Y
                        )


                    # P항
                    p_term = (
                        PD_KP * effective_error
                    )


                    # D항
                    if previous_error_y is None:

                        raw_error_derivative = 0.0

                    else:

                        raw_error_derivative = (
                            error_y - previous_error_y
                        ) / delta_time


                    filtered_error_derivative = (
                        DERIVATIVE_FILTER_ALPHA
                        * raw_error_derivative
                        + (
                            1.0
                            - DERIVATIVE_FILTER_ALPHA
                        )
                        * filtered_error_derivative
                    )

                    d_term = (
                        PD_KD
                        * filtered_error_derivative
                    )

                    d_term = clamp(
                        d_term,
                        -MAX_D_TERM,
                        MAX_D_TERM
                    )


                    # 화면 오차를 서보 방향에 맞게 변환
                    target_velocity = (
                        p_term + d_term
                    ) * SERVO_DIRECTION


                    # 중앙 근처에서 속도 감속
                    if abs(error_y) < SLOW_ZONE_Y:

                        slow_ratio = (
                            abs(error_y) - DEAD_ZONE_Y
                        ) / (
                            SLOW_ZONE_Y - DEAD_ZONE_Y
                        )

                        slow_ratio = clamp(
                            slow_ratio,
                            0.20,
                            1.0
                        )

                        target_velocity *= slow_ratio


                    target_velocity = clamp(
                        target_velocity,
                        -MAX_SERVO_SPEED,
                        MAX_SERVO_SPEED
                    )

                    previous_error_y = error_y


                # 가속도 제한
                servo_velocity = update_velocity_smoothly(
                    current_velocity=servo_velocity,
                    target_velocity=target_velocity,
                    max_acceleration=MAX_SERVO_ACCELERATION,
                    delta_time=delta_time
                )


                requested_angle = (
                    current_angle
                    + servo_velocity * delta_time
                )


                # --------------------------------------------
                # 서보 각도 제한
                #
                # 제한에 도달해도 초기 위치 복귀 상태로
                # 전환하지 않고 해당 한계에서 정지합니다.
                #
                # 이후 오차 방향이 반대로 바뀌면 즉시
                # 안전 범위 안쪽으로 다시 움직일 수 있습니다.
                # --------------------------------------------

                servo_limit_status = None

                if requested_angle <= MIN_ANGLE:

                    current_angle = set_servo_angle_safe(
                        MIN_ANGLE
                    )

                    # 더 낮은 각도로 움직이려는 속도만 제거
                    if servo_velocity < 0:
                        servo_velocity = 0.0

                    servo_limit_status = "MIN LIMIT"

                elif requested_angle >= MAX_ANGLE:

                    current_angle = set_servo_angle_safe(
                        MAX_ANGLE
                    )

                    # 더 높은 각도로 움직이려는 속도만 제거
                    if servo_velocity > 0:
                        servo_velocity = 0.0

                    servo_limit_status = "MAX LIMIT"

                else:

                    current_angle = set_servo_angle_safe(
                        requested_angle
                    )


                last_servo_update_time = current_time


            # ------------------------------------------------
            # 탐지 정보 표시
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"{TARGET_CLASS_NAME}: {confidence:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Error Y: {error_y:.1f}px",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Servo speed: {servo_velocity:.2f} deg/s",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"P: {p_term:.2f}  D: {d_term:.2f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 150, 0),
                2
            )


            if abs(error_y) <= DEAD_ZONE_Y:

                cv2.putText(
                    frame,
                    f"{TARGET_CLASS_NAME.upper()} CENTERED",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


            if servo_limit_status is not None:

                cv2.putText(
                    frame,
                    servo_limit_status,
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )


        # ----------------------------------------------------
        # 초기 위치 복귀 중
        # ----------------------------------------------------

        elif returning_to_initial:

            filtered_target_y = None
            previous_error_y = None
            filtered_error_derivative = 0.0
            servo_limit_status = None

            if (
                current_time - last_servo_update_time
                >= RETURN_INTERVAL
            ):

                return_error = (
                    INITIAL_ANGLE - current_angle
                )

                target_return_velocity = clamp(
                    return_error * 3.5,
                    -RETURN_MAX_SPEED,
                    RETURN_MAX_SPEED
                )

                servo_velocity = update_velocity_smoothly(
                    current_velocity=servo_velocity,
                    target_velocity=target_return_velocity,
                    max_acceleration=RETURN_ACCELERATION,
                    delta_time=delta_time
                )

                next_angle = (
                    current_angle
                    + servo_velocity * delta_time
                )


                # 초기 위치를 지나치지 않도록 제한
                if (
                    current_angle > INITIAL_ANGLE
                    and next_angle < INITIAL_ANGLE
                ):
                    next_angle = INITIAL_ANGLE

                elif (
                    current_angle < INITIAL_ANGLE
                    and next_angle > INITIAL_ANGLE
                ):
                    next_angle = INITIAL_ANGLE


                current_angle = set_servo_angle_safe(
                    next_angle
                )

                last_servo_update_time = current_time


            if (
                abs(current_angle - INITIAL_ANGLE)
                <= RETURN_ANGLE_TOLERANCE
                and abs(servo_velocity)
                <= RETURN_VELOCITY_TOLERANCE
            ):

                current_angle = set_servo_angle_safe(
                    INITIAL_ANGLE
                )

                servo_velocity = 0.0
                returning_to_initial = False
                last_detection_time = current_time

                print(
                    f"[INFO] 초기 위치 "
                    f"{INITIAL_ANGLE:.1f}도 복귀 완료"
                )


        # ----------------------------------------------------
        # 대상이 탐지되지 않은 상태
        # ----------------------------------------------------

        else:

            filtered_target_y = None
            previous_error_y = None
            filtered_error_derivative = 0.0
            servo_limit_status = None

            elapsed_time = (
                current_time - last_detection_time
            )


            # 대상이 사라지면 현재 속도를 부드럽게 감소
            if (
                current_time - last_servo_update_time
                >= SERVO_UPDATE_INTERVAL
            ):

                servo_velocity = update_velocity_smoothly(
                    current_velocity=servo_velocity,
                    target_velocity=0.0,
                    max_acceleration=MAX_SERVO_ACCELERATION,
                    delta_time=delta_time
                )

                if abs(servo_velocity) < 0.05:
                    servo_velocity = 0.0

                last_servo_update_time = current_time


            cv2.putText(
                frame,
                (
                    f"{TARGET_CLASS_NAME.capitalize()} lost: "
                    f"{elapsed_time:.1f}s"
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )


            if elapsed_time >= LOST_RETURN_TIME:

                servo_velocity = 0.0
                returning_to_initial = True

                print(
                    f"[INFO] {TARGET_CLASS_NAME} 분실 "
                    f"{elapsed_time:.1f}초 - "
                    f"{INITIAL_ANGLE:.1f}도로 복귀 시작"
                )


        # ----------------------------------------------------
        # 현재 상태 문자열
        # ----------------------------------------------------

        if returning_to_initial:

            status_text = (
                f"{TARGET_CLASS_NAME.upper()} LOST - RETURNING"
            )

        elif target is not None:

            if servo_limit_status is not None:
                status_text = servo_limit_status
            else:
                status_text = (
                    f"TRACKING {TARGET_CLASS_NAME.upper()}"
                )

        else:

            status_text = (
                f"SEARCHING {TARGET_CLASS_NAME.upper()}"
            )


        # ----------------------------------------------------
        # 공통 화면 정보
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Camera: {frame_width} x {frame_height}",
            (frame_width - 360, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"YOLO: {INFERENCE_SIZE} x "
                f"{INFERENCE_SIZE}"
            ),
            (frame_width - 360, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (frame_width - 360, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Servo: {current_angle:.2f} deg",
            (10, frame_height - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            (
                f"Safe range: "
                f"{MIN_ANGLE:.0f} - {MAX_ANGLE:.0f} deg"
            ),
            (10, frame_height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Status: {status_text}",
            (10, frame_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 0),
            2
        )


        window_title = (
            f"YOLO {TARGET_CLASS_NAME.capitalize()} Tracking "
            f"- Camera {CAMERA_WIDTH}x{CAMERA_HEIGHT} "
            f"- Inference {INFERENCE_SIZE}"
        )

        cv2.imshow(
            window_title,
            frame
        )


        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break


except KeyboardInterrupt:

    print(
        "\n[INFO] 사용자가 프로그램을 중단했습니다."
    )


finally:

    print(
        f"[INFO] 서보를 초기 위치 "
        f"{INITIAL_ANGLE:.1f}도로 복귀합니다."
    )

    return_velocity = 0.0
    previous_time = time.monotonic()

    while (
        abs(current_angle - INITIAL_ANGLE)
        > 0.05
    ):

        now = time.monotonic()

        delta_time = clamp(
            now - previous_time,
            0.001,
            0.1
        )

        previous_time = now

        return_error = (
            INITIAL_ANGLE - current_angle
        )

        target_velocity = clamp(
            return_error * 3.5,
            -RETURN_MAX_SPEED,
            RETURN_MAX_SPEED
        )

        return_velocity = update_velocity_smoothly(
            current_velocity=return_velocity,
            target_velocity=target_velocity,
            max_acceleration=RETURN_ACCELERATION,
            delta_time=delta_time
        )

        next_angle = (
            current_angle
            + return_velocity * delta_time
        )


        if (
            current_angle > INITIAL_ANGLE
            and next_angle < INITIAL_ANGLE
        ):
            next_angle = INITIAL_ANGLE

        elif (
            current_angle < INITIAL_ANGLE
            and next_angle > INITIAL_ANGLE
        ):
            next_angle = INITIAL_ANGLE


        current_angle = set_servo_angle_safe(
            next_angle
        )

        time.sleep(RETURN_INTERVAL)


    current_angle = set_servo_angle_safe(
        INITIAL_ANGLE
    )

    camera.release()
    cv2.destroyAllWindows()

    print(
        f"[INFO] 서보 "
        f"{current_angle:.1f}도 복귀 완료"
    )

    print("[INFO] 카메라를 종료했습니다.")
    print("[INFO] 프로그램을 종료했습니다.")
