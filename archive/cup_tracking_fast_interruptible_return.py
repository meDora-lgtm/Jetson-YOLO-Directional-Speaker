import time
import cv2

from ultralytics import YOLO
from adafruit_extended_bus import ExtendedI2C
from adafruit_servokit import ServoKit
from adafruit_bus_device import i2c_device


# ============================================================
# 사용자 설정
# ============================================================

MODEL_PATH = "yolo26n.engine"
CAMERA_DEVICE = "/dev/video0"

# HD 카메라 설정
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# 화면은 HD로 표시하고 YOLO 추론은 640으로 수행
IMAGE_SIZE = 640
CONFIDENCE = 0.45

# COCO 데이터셋 cup 클래스 번호
TARGET_CLASS_ID = 41
TARGET_CLASS_NAME = "cup"


# ============================================================
# PCA9685 및 서보 설정
# ============================================================

PCA9685_ADDRESS = 0x41
SERVO_CHANNEL = 0

# 물리적인 서보 안전 범위
#
# 125도 미만:
# 장치의 턱에 걸리므로 절대로 이동하지 않음
#
# 180도 초과:
# 장치가 너무 높아 정상 작동하지 않으므로 이동하지 않음
INITIAL_ANGLE = 125.0
MIN_ANGLE = 125.0
MAX_ANGLE = 180.0

# 컵 중심이 중앙에서 ±35픽셀 이내면 정지
DEAD_ZONE_Y = 30

# 중앙 근처에서 감속하기 시작하는 범위
SLOW_ZONE_Y = 120

# 화면 오차를 서보 속도로 변환하는 비율
KP_SPEED = 0.070

# 서보 최대 속도: degree/second
MAX_SERVO_SPEED = 20.0

# 서보 최대 가속도: degree/second²
MAX_SERVO_ACCELERATION = 65.0

# 컵 중심좌표 필터 강도
# 낮을수록 부드럽지만 반응이 느림
TARGET_FILTER_ALPHA = 0.35

# 서보 명령 최소 간격
SERVO_UPDATE_INTERVAL = 0.015

# 컵이 중앙에 들어왔을 때 속도 감쇠 비율
CENTER_VELOCITY_DAMPING = 0.55

# 컵이 사라진 후 초기 위치 복귀 시간
LOST_RETURN_TIME = 2.0

# 초기 위치 복귀 설정
RETURN_MAX_SPEED = 14.0
RETURN_ACCELERATION = 45.0
RETURN_INTERVAL = 0.015

# 현재 장치의 실제 움직임 방향을 반대로 수정
#
# 컵의 빨간 점이 화면 위에 있으면 카메라가 위로 이동
# 컵의 빨간 점이 화면 아래에 있으면 카메라가 아래로 이동
SERVO_DIRECTION = -1


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

i2c = ExtendedI2C(1)

kit = ServoKit(
    channels=16,
    i2c=i2c,
    address=PCA9685_ADDRESS
)

kit.servo[SERVO_CHANNEL].set_pulse_width_range(
    500,
    2500
)


# ============================================================
# YOLO 모델 불러오기
# ============================================================

print("[INFO] YOLO 모델을 불러옵니다.")

model = YOLO(MODEL_PATH)

print("[INFO] 탐지 대상: cup")
print("[INFO] 카메라 설정: 1280 x 720")
print("[INFO] YOLO 추론 크기: 640")
print("[INFO] 서보 안전 범위: 125도 ~ 180도")
print("[INFO] 서보 방향 반전 설정: -1")


# ============================================================
# 카메라 열기
# ============================================================

print("[INFO] HD 카메라를 실행합니다.")

camera = cv2.VideoCapture(
    CAMERA_DEVICE,
    cv2.CAP_V4L2
)

# USB 카메라의 HD 30FPS 사용을 위해 MJPG 요청
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
    1
)

if not camera.isOpened():
    raise RuntimeError(
        f"카메라를 열 수 없습니다: {CAMERA_DEVICE}"
    )


# 실제 카메라 적용값 확인
actual_width = int(
    camera.get(cv2.CAP_PROP_FRAME_WIDTH)
)

actual_height = int(
    camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

actual_fps = camera.get(
    cv2.CAP_PROP_FPS
)

print(
    f"[INFO] 실제 카메라 설정: "
    f"{actual_width} x {actual_height}, "
    f"{actual_fps:.1f} FPS"
)


# ============================================================
# 값 제한 함수
# ============================================================

def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(float(value), maximum)
    )


def clamp_angle(angle):
    """
    서보 각도를 반드시 125도 이상,
    180도 이하로 제한합니다.
    """

    return clamp(
        angle,
        MIN_ANGLE,
        MAX_ANGLE
    )


# ============================================================
# 안전하게 서보 각도 적용
# ============================================================

def set_servo_angle_safe(angle):
    """
    서보에 전달하기 전에 반드시 안전 범위로 제한합니다.
    """

    safe_angle = clamp_angle(angle)

    kit.servo[SERVO_CHANNEL].angle = safe_angle

    return safe_angle


# ============================================================
# 서보 속도 및 가속도 제한
# ============================================================

def update_velocity_smoothly(
    current_velocity,
    target_velocity,
    max_acceleration,
    delta_time
):
    """
    현재 속도를 목표 속도 방향으로 천천히 변화시켜
    갑작스러운 출발과 정지를 방지합니다.
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


# ============================================================
# 가장 크게 탐지된 컵 선택
# ============================================================

def select_largest_cup(boxes):
    """
    컵이 여러 개 탐지되면 화면에서 가장 크게 보이는
    컵 한 개를 선택합니다.

    반환값:
        x1, y1, x2, y2, confidence

    컵이 없으면:
        None
    """

    if boxes is None:
        return None

    selected_cup = None
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

        width = x2 - x1
        height = y2 - y1
        area = width * height

        if area > largest_area:

            largest_area = area

            selected_cup = (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                confidence
            )

    return selected_cup


# ============================================================
# 초기 위치 설정
# ============================================================

current_angle = set_servo_angle_safe(
    INITIAL_ANGLE
)

print(
    f"[INFO] 서보 초기 위치: "
    f"{current_angle:.1f}도"
)

time.sleep(1)


# ============================================================
# 상태 변수
# ============================================================

last_detection_time = time.monotonic()
last_servo_update_time = time.monotonic()

returning_to_initial = False
limit_exceeded = False

# 현재 서보 이동 속도
servo_velocity = 0.0

# 필터링된 컵 중심 Y 좌표
filtered_target_y = None


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

            time.sleep(0.1)
            continue


        frame_height, frame_width = frame.shape[:2]

        screen_center_x = frame_width // 2
        screen_center_y = frame_height // 2

        current_time = time.monotonic()

        delta_time = (
            current_time - last_servo_update_time
        )

        # 순간적으로 시간이 너무 크게 계산되는 것 방지
        delta_time = clamp(
            delta_time,
            0.001,
            0.1
        )


        # ----------------------------------------------------
        # YOLO 컵 탐지
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            classes=[TARGET_CLASS_ID],
            verbose=False
        )

        cup = select_largest_cup(
            results[0].boxes
        )


        # ----------------------------------------------------
        # 화면 중앙 가로선 표시
        # ----------------------------------------------------

        # 정확한 화면 중앙선
        cv2.line(
            frame,
            (0, screen_center_y),
            (frame_width, screen_center_y),
            (255, 255, 255),
            2
        )

        # 중앙 위쪽 데드존
        cv2.line(
            frame,
            (0, screen_center_y - DEAD_ZONE_Y),
            (frame_width, screen_center_y - DEAD_ZONE_Y),
            (100, 100, 100),
            1
        )

        # 중앙 아래쪽 데드존
        cv2.line(
            frame,
            (0, screen_center_y + DEAD_ZONE_Y),
            (frame_width, screen_center_y + DEAD_ZONE_Y),
            (100, 100, 100),
            1
        )

        # 화면 정중앙 표시
        cv2.circle(
            frame,
            (screen_center_x, screen_center_y),
            7,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # 컵이 탐지되면 복귀 중이어도 즉시 추적 재시작
        # ----------------------------------------------------

        if cup is not None:

            # 초기 위치 복귀 중 컵을 다시 발견한 경우,
            # 복귀를 즉시 취소하고 컵 추적으로 전환합니다.
            if returning_to_initial:

                returning_to_initial = False
                limit_exceeded = False

                # 복귀 방향의 관성을 제거하여 반대 방향 급회전을 방지
                servo_velocity = 0.0
                filtered_target_y = None

                print(
                    "[INFO] 복귀 중 컵 재탐지 - "
                    "복귀 취소 후 추적 재시작"
                )


            x1, y1, x2, y2, confidence = cup

            cup_center_x = (
                x1 + x2
            ) // 2

            raw_cup_center_y = (
                y1 + y2
            ) / 2.0


            # ------------------------------------------------
            # 컵 중심 좌표 필터링
            # ------------------------------------------------

            if filtered_target_y is None:

                filtered_target_y = (
                    raw_cup_center_y
                )

            else:

                filtered_target_y = (
                    TARGET_FILTER_ALPHA
                    * raw_cup_center_y
                    + (
                        1.0
                        - TARGET_FILTER_ALPHA
                    )
                    * filtered_target_y
                )


            filtered_center_y_int = int(
                filtered_target_y
            )


            # 세로 오차
            #
            # 음수:
            # 컵의 빨간 점이 화면 중앙보다 위에 있음
            #
            # 양수:
            # 컵의 빨간 점이 화면 중앙보다 아래에 있음
            error_y = (
                filtered_target_y
                - screen_center_y
            )

            last_detection_time = current_time


            # ------------------------------------------------
            # 컵 표시
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (
                    cup_center_x,
                    int(raw_cup_center_y)
                ),
                6,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                (
                    cup_center_x,
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
                    cup_center_x,
                    filtered_center_y_int
                ),
                (0, 255, 255),
                2
            )


            # ------------------------------------------------
            # 고속·부드러운 서보 속도 제어
            # ------------------------------------------------

            if (
                current_time - last_servo_update_time
                >= SERVO_UPDATE_INTERVAL
            ):

                if abs(error_y) <= DEAD_ZONE_Y:

                    target_velocity = 0.0

                    servo_velocity *= (
                        CENTER_VELOCITY_DAMPING
                    )

                    if abs(servo_velocity) < 0.08:
                        servo_velocity = 0.0

                else:

                    effective_error = error_y

                    if effective_error > 0:
                        effective_error -= DEAD_ZONE_Y
                    else:
                        effective_error += DEAD_ZONE_Y

                    target_velocity = (
                        KP_SPEED
                        * effective_error
                        * SERVO_DIRECTION
                    )

                    target_velocity = clamp(
                        target_velocity,
                        -MAX_SERVO_SPEED,
                        MAX_SERVO_SPEED
                    )

                    if abs(error_y) < SLOW_ZONE_Y:

                        slow_ratio = (
                            abs(error_y) - DEAD_ZONE_Y
                        ) / (
                            SLOW_ZONE_Y - DEAD_ZONE_Y
                        )

                        slow_ratio = clamp(
                            slow_ratio,
                            0.25,
                            1.0
                        )

                        target_velocity *= slow_ratio


                servo_velocity = update_velocity_smoothly(
                    current_velocity=servo_velocity,
                    target_velocity=target_velocity,
                    max_acceleration=MAX_SERVO_ACCELERATION,
                    delta_time=delta_time
                )

                angle_change = (
                    servo_velocity * delta_time
                )

                requested_angle = (
                    current_angle + angle_change
                )


                # 125도 미만 또는 180도 초과 요청 시
                # 현재 안전 한계에서 멈추고 복귀 상태로 전환
                if requested_angle < MIN_ANGLE:

                    print(
                        "[SAFETY] 125도 미만 명령 차단"
                    )

                    current_angle = set_servo_angle_safe(
                        MIN_ANGLE
                    )

                    servo_velocity = 0.0
                    returning_to_initial = True
                    limit_exceeded = True

                elif requested_angle > MAX_ANGLE:

                    print(
                        "[SAFETY] 180도 초과 명령 차단"
                    )

                    current_angle = set_servo_angle_safe(
                        MAX_ANGLE
                    )

                    servo_velocity = 0.0
                    returning_to_initial = True
                    limit_exceeded = True

                else:

                    current_angle = set_servo_angle_safe(
                        requested_angle
                    )

                last_servo_update_time = current_time


            if abs(error_y) <= DEAD_ZONE_Y:

                cv2.putText(
                    frame,
                    "CUP CENTERED",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


            cv2.putText(
                frame,
                f"Cup: {confidence:.2f}",
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


        # ----------------------------------------------------
        # 컵이 없고 초기 위치로 복귀 중
        # ----------------------------------------------------

        elif returning_to_initial:

            filtered_target_y = None

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

                if (
                    current_angle > INITIAL_ANGLE
                    and next_angle < INITIAL_ANGLE
                ):
                    next_angle = INITIAL_ANGLE

                current_angle = set_servo_angle_safe(
                    next_angle
                )

                last_servo_update_time = current_time


            if (
                abs(current_angle - INITIAL_ANGLE) <= 0.08
                and abs(servo_velocity) <= 0.35
            ):

                current_angle = set_servo_angle_safe(
                    INITIAL_ANGLE
                )

                servo_velocity = 0.0
                returning_to_initial = False
                limit_exceeded = False
                last_detection_time = current_time

                print(
                    "[INFO] 초기 위치 125도 복귀 완료"
                )


        # ----------------------------------------------------
        # 컵이 탐지되지 않은 경우
        # ----------------------------------------------------

        else:

            filtered_target_y = None

            elapsed_time = (
                current_time - last_detection_time
            )

            # 컵이 사라지면 서보 속도를 천천히 감소
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
                f"Cup lost: {elapsed_time:.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )


            # 3초 동안 컵이 없으면 초기 위치로 복귀
            if elapsed_time >= LOST_RETURN_TIME:

                servo_velocity = 0.0
                returning_to_initial = True
                limit_exceeded = False


        # ----------------------------------------------------
        # 상태 표시
        # ----------------------------------------------------

        if returning_to_initial:

            if limit_exceeded:
                status_text = "LIMIT BLOCKED - RETURNING"
            else:
                status_text = "CUP LOST - RETURNING"

        elif cup is not None:
            status_text = "SMOOTH VERTICAL CUP TRACKING"

        else:
            status_text = "SEARCHING CUP"


        cv2.putText(
            frame,
            f"Resolution: {frame_width} x {frame_height}",
            (frame_width - 330, 30),
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
            "Safe range: 125 - 180 deg",
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


        cv2.imshow(
            "YOLO HD Smooth Vertical Cup Tracking",
            frame
        )


        # q 또는 ESC를 누르면 종료
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break


except KeyboardInterrupt:

    print(
        "\n[INFO] 사용자가 프로그램을 중단했습니다."
    )


finally:

    print("[INFO] 초기 위치로 복귀합니다.")

    return_velocity = 0.0
    previous_time = time.monotonic()

    # 종료 시에도 부드럽게 125도로 복귀
    while abs(current_angle - INITIAL_ANGLE) > 0.05:

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

        current_angle = set_servo_angle_safe(
            next_angle
        )

        time.sleep(RETURN_INTERVAL)


    current_angle = set_servo_angle_safe(
        INITIAL_ANGLE
    )

    camera.release()
    cv2.destroyAllWindows()

    print("[INFO] 서보 초기 위치 복귀 완료")
    print("[INFO] 카메라를 종료했습니다.")
    print("[INFO] 프로그램을 종료했습니다.")