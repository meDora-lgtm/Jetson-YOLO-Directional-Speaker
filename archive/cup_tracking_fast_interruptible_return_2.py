import time
import cv2

from ultralytics import YOLO
from adafruit_extended_bus import ExtendedI2C
from adafruit_servokit import ServoKit
from adafruit_bus_device import i2c_device


# ============================================================
# 사용자 설정
# ============================================================

MODEL_PATH = "yolo26n_960.engine"
CAMERA_DEVICE = "/dev/video0"

# USB 카메라 설정
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# YOLO 추론 크기
#
IMAGE_SIZE = 960

# 컵 탐지 신뢰도
CONFIDENCE = 0.30

# COCO 데이터셋 cup 클래스
TARGET_CLASS_ID = 41
TARGET_CLASS_NAME = "cup"


# ============================================================
# PCA9685 및 서보 설정
# ============================================================

PCA9685_ADDRESS = 0x41
SERVO_CHANNEL = 0

# ------------------------------------------------------------
# 서보 물리적 안전 범위
#
# 125도 미만:
# 장치 턱에 걸릴 수 있으므로 절대 이동하지 않습니다.
#
# 180도 초과:
# 장치가 지나치게 올라가므로 이동하지 않습니다.
# ------------------------------------------------------------

MIN_ANGLE = 125.0
MAX_ANGLE = 180.0

# 양방향 추적이 가능하도록 안전범위 중앙값을 사용합니다.
#
# 실제 장치의 카메라 정중앙 위치가 다르면
# 이 값만 조금씩 조절하십시오.
INITIAL_ANGLE = 130.5


# ============================================================
# 화면 중심 제어 설정
# ============================================================

# 컵 중심이 화면 중앙에서 ±30픽셀 이내면 정지
DEAD_ZONE_Y = 30

# 중앙 근처에서 감속하기 시작하는 범위
SLOW_ZONE_Y = 120

# 중앙에 들어온 뒤 남아 있는 서보 속도 감쇠 비율
CENTER_VELOCITY_DAMPING = 0.55


# ============================================================
# PD 제어 설정
# ============================================================

# P 게인
#
# 컵이 화면 중앙에서 멀리 있을수록
# 서보 목표 속도를 크게 만듭니다.
PD_KP = 0.065

# D 게인
#
# 오차 변화 속도를 이용하여 중앙 통과와 진동을 줄입니다.
PD_KD = 0.012

# D항 저역통과 필터 강도
#
# 작을수록 더 부드럽지만 반응이 느립니다.
# 클수록 빠르게 반응하지만 탐지 흔들림에 민감합니다.
DERIVATIVE_FILTER_ALPHA = 0.25

# D항 최대 크기 제한
#
# 프레임 지연이나 탐지 위치 변화로 D항이 순간적으로
# 지나치게 커지는 것을 방지합니다.
MAX_D_TERM = 8.0


# ============================================================
# 서보 움직임 설정
# ============================================================

# 추적 중 최대 서보 속도: degree/second
MAX_SERVO_SPEED = 20.0

# 추적 중 최대 서보 가속도: degree/second²
MAX_SERVO_ACCELERATION = 65.0

# 컵 중심좌표 필터 강도
#
# 작을수록 영상 흔들림에는 강하지만 반응이 느립니다.
# 클수록 빠르게 반응하지만 탐지 좌표 변화에 민감합니다.
TARGET_FILTER_ALPHA = 0.35

# 서보 명령 최소 간격
SERVO_UPDATE_INTERVAL = 0.015


# ============================================================
# 컵 소실 및 초기 위치 복귀 설정
# ============================================================

# 컵이 보이지 않은 뒤 초기 위치 복귀를 시작할 시간
LOST_RETURN_TIME = 2.0

# 초기 위치 복귀 최대 속도
RETURN_MAX_SPEED = 14.0

# 초기 위치 복귀 최대 가속도
RETURN_ACCELERATION = 45.0

# 초기 위치 복귀 명령 간격
RETURN_INTERVAL = 0.015

# 초기 위치 복귀 속도 비례 게인
RETURN_KP = 3.5


# ============================================================
# 서보 방향 설정
# ============================================================

# 장치 움직임 방향
#
# 현재 설명 기준:
# 컵이 화면 위에 있으면 카메라가 위로 움직이고
# 컵이 화면 아래에 있으면 카메라가 아래로 움직이도록 -1 사용
#
# 실제 움직임이 반대로 나오면 1로 변경하십시오.
SERVO_DIRECTION = -1


# ============================================================
# I2C 장치 탐지 우회
# ============================================================

original_i2c_device_init = i2c_device.I2CDevice.__init__


def no_probe_init(self, i2c, device_address, probe=True):
    """
    일부 Jetson 환경에서 PCA9685 초기 탐지가 실패하는 문제를
    피하기 위해 I2C probe를 비활성화합니다.
    """

    original_i2c_device_init(
        self,
        i2c,
        device_address,
        probe=False
    )


i2c_device.I2CDevice.__init__ = no_probe_init


# ============================================================
# 값 제한 함수
# ============================================================

def clamp(value, minimum, maximum):
    """
    value를 minimum과 maximum 사이로 제한합니다.
    """

    return max(
        minimum,
        min(float(value), maximum)
    )


def clamp_angle(angle):
    """
    서보 각도를 물리적 안전범위 안으로 제한합니다.
    """

    return clamp(
        angle,
        MIN_ANGLE,
        MAX_ANGLE
    )


# ============================================================
# PCA9685 초기화
# ============================================================

print("[INFO] PCA9685를 초기화합니다.")

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
# 안전하게 서보 각도 적용
# ============================================================

def set_servo_angle_safe(angle):
    """
    서보에 명령을 보내기 전에 반드시 안전범위로 제한합니다.
    """

    safe_angle = clamp_angle(angle)

    kit.servo[SERVO_CHANNEL].angle = safe_angle

    return safe_angle


# ============================================================
# 속도 및 가속도 제한 함수
# ============================================================

def update_velocity_smoothly(
    current_velocity,
    target_velocity,
    max_acceleration,
    delta_time
):
    """
    현재 속도를 목표 속도 방향으로 서서히 변화시킵니다.

    서보의 갑작스러운 출발, 정지 및 방향 전환을 줄입니다.
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
    컵이 여러 개 탐지된 경우 화면에서 가장 크게 보이는
    컵 한 개를 추적 대상으로 선택합니다.

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

        box_width = max(
            0.0,
            x2 - x1
        )

        box_height = max(
            0.0,
            y2 - y1
        )

        box_area = (
            box_width * box_height
        )

        if box_area > largest_area:

            largest_area = box_area

            selected_cup = (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                confidence
            )

    return selected_cup


# ============================================================
# YOLO 모델 불러오기
# ============================================================

print("[INFO] YOLO 모델을 불러옵니다.")

model = YOLO(
    MODEL_PATH,
    task="detect"
)

print(f"[INFO] 탐지 대상: {TARGET_CLASS_NAME}")
print(
    f"[INFO] 카메라 요청 설정: "
    f"{CAMERA_WIDTH} x {CAMERA_HEIGHT}, "
    f"{CAMERA_FPS} FPS"
)
print(f"[INFO] YOLO 추론 크기: {IMAGE_SIZE}")
print(
    f"[INFO] 서보 안전범위: "
    f"{MIN_ANGLE:.1f}도 ~ {MAX_ANGLE:.1f}도"
)
print(f"[INFO] 초기 위치: {INITIAL_ANGLE:.1f}도")
print(f"[INFO] 서보 방향 설정: {SERVO_DIRECTION}")


# ============================================================
# 카메라 열기
# ============================================================

print("[INFO] HD 카메라를 실행합니다.")

camera = cv2.VideoCapture(
    CAMERA_DEVICE,
    cv2.CAP_V4L2
)

# HD 30FPS 출력을 위해 MJPG 요청
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

# 오래된 프레임이 쌓이는 현상 감소
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
# 서보 초기 위치 설정
# ============================================================

current_angle = set_servo_angle_safe(
    INITIAL_ANGLE
)

print(
    f"[INFO] 서보 초기 위치 설정 완료: "
    f"{current_angle:.1f}도"
)

time.sleep(1.0)


# ============================================================
# 상태 변수
# ============================================================

current_time = time.monotonic()

last_detection_time = current_time
last_servo_update_time = current_time

# 현재 초기 위치 복귀 중인지 여부
returning_to_initial = False

# 현재 서보 이동 속도
servo_velocity = 0.0

# 필터링된 컵 중심 Y 좌표
filtered_target_y = None

# 이전 PD 오차
previous_error_y = None

# 필터링된 오차 변화율
filtered_derivative_y = 0.0

# 현재 각도 제한에 도달했는지 여부
angle_limit_active = False

# 제한 메시지가 반복 출력되지 않도록 저장
previous_limit_state = None


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

        # 프레임 중단 등으로 시간이 지나치게 커지는 것을 방지
        delta_time = clamp(
            delta_time,
            0.001,
            0.1
        )


        # ====================================================
        # YOLO 컵 탐지
        # ====================================================

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


        # ====================================================
        # 화면 중앙선 및 데드존 표시
        # ====================================================

        # 정확한 중앙선
        cv2.line(
            frame,
            (0, screen_center_y),
            (frame_width, screen_center_y),
            (255, 255, 255),
            2
        )

        # 데드존 상단
        cv2.line(
            frame,
            (0, screen_center_y - DEAD_ZONE_Y),
            (
                frame_width,
                screen_center_y - DEAD_ZONE_Y
            ),
            (100, 100, 100),
            1
        )

        # 데드존 하단
        cv2.line(
            frame,
            (0, screen_center_y + DEAD_ZONE_Y),
            (
                frame_width,
                screen_center_y + DEAD_ZONE_Y
            ),
            (100, 100, 100),
            1
        )

        # 화면 정중앙
        cv2.circle(
            frame,
            (
                screen_center_x,
                screen_center_y
            ),
            7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # 컵 탐지 상태
        # ====================================================

        if cup is not None:

            # 복귀 중 컵이 다시 탐지되면 즉시 추적 재개
            if returning_to_initial:

                returning_to_initial = False

                # 복귀 방향 속도를 제거하여 급격한 반전을 방지
                servo_velocity = 0.0

                # 이전 탐지 정보 초기화
                filtered_target_y = None
                previous_error_y = None
                filtered_derivative_y = 0.0

                print(
                    "[INFO] 복귀 중 컵 재탐지 - "
                    "복귀를 취소하고 추적을 재개합니다."
                )


            x1, y1, x2, y2, confidence = cup

            cup_center_x = (
                x1 + x2
            ) // 2

            raw_cup_center_y = (
                y1 + y2
            ) / 2.0


            # ------------------------------------------------
            # 컵 중심좌표 저역통과 필터
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


            # ------------------------------------------------
            # 세로 오차 계산
            #
            # 음수:
            # 컵이 화면 중앙보다 위에 있음
            #
            # 양수:
            # 컵이 화면 중앙보다 아래에 있음
            # ------------------------------------------------

            error_y = (
                filtered_target_y
                - screen_center_y
            )

            last_detection_time = current_time


            # ------------------------------------------------
            # 탐지 결과 화면 표시
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # 원본 컵 중심
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

            # 필터링된 컵 중심
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

            # 중앙에서 컵까지 연결선
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


            # =================================================
            # PD 기반 서보 속도 제어
            # =================================================

            p_term = 0.0
            d_term = 0.0
            target_velocity = 0.0
            angle_limit_active = False

            if (
                current_time - last_servo_update_time
                >= SERVO_UPDATE_INTERVAL
            ):

                # ---------------------------------------------
                # 중앙 데드존에 들어온 경우
                # ---------------------------------------------

                if abs(error_y) <= DEAD_ZONE_Y:

                    target_velocity = 0.0

                    # 남아 있는 관성 속도를 빠르게 감쇠
                    servo_velocity *= (
                        CENTER_VELOCITY_DAMPING
                    )

                    if abs(servo_velocity) < 0.08:
                        servo_velocity = 0.0

                    # D항도 중앙에서 서서히 제거
                    filtered_derivative_y *= 0.5

                    if abs(filtered_derivative_y) < 0.1:
                        filtered_derivative_y = 0.0


                # ---------------------------------------------
                # 데드존 밖에 있는 경우
                # ---------------------------------------------

                else:

                    # 데드존만큼은 제어 오차에서 제외합니다.
                    if error_y > 0:
                        effective_error = (
                            error_y - DEAD_ZONE_Y
                        )
                    else:
                        effective_error = (
                            error_y + DEAD_ZONE_Y
                        )


                    # -----------------------------------------
                    # 오차 변화율 계산
                    # -----------------------------------------

                    if previous_error_y is None:

                        raw_derivative_y = 0.0

                    else:

                        raw_derivative_y = (
                            error_y
                            - previous_error_y
                        ) / delta_time


                    # D항 입력 저역통과 필터
                    filtered_derivative_y = (
                        DERIVATIVE_FILTER_ALPHA
                        * raw_derivative_y
                        + (
                            1.0
                            - DERIVATIVE_FILTER_ALPHA
                        )
                        * filtered_derivative_y
                    )


                    # -----------------------------------------
                    # P항 계산
                    # -----------------------------------------

                    p_term = (
                        PD_KP
                        * effective_error
                    )


                    # -----------------------------------------
                    # D항 계산 및 제한
                    # -----------------------------------------

                    d_term = (
                        PD_KD
                        * filtered_derivative_y
                    )

                    d_term = clamp(
                        d_term,
                        -MAX_D_TERM,
                        MAX_D_TERM
                    )


                    # -----------------------------------------
                    # 최종 목표 속도 계산
                    # -----------------------------------------

                    target_velocity = (
                        (p_term + d_term)
                        * SERVO_DIRECTION
                    )

                    target_velocity = clamp(
                        target_velocity,
                        -MAX_SERVO_SPEED,
                        MAX_SERVO_SPEED
                    )


                    # -----------------------------------------
                    # 중앙 근처 감속
                    # -----------------------------------------

                    if abs(error_y) < SLOW_ZONE_Y:

                        slow_ratio = (
                            abs(error_y)
                            - DEAD_ZONE_Y
                        ) / (
                            SLOW_ZONE_Y
                            - DEAD_ZONE_Y
                        )

                        slow_ratio = clamp(
                            slow_ratio,
                            0.25,
                            1.0
                        )

                        target_velocity *= slow_ratio


                # ---------------------------------------------
                # 최대 가속도를 적용하여 속도를 부드럽게 변경
                # ---------------------------------------------

                servo_velocity = update_velocity_smoothly(
                    current_velocity=servo_velocity,
                    target_velocity=target_velocity,
                    max_acceleration=MAX_SERVO_ACCELERATION,
                    delta_time=delta_time
                )


                # ---------------------------------------------
                # 다음 서보 각도 계산
                # ---------------------------------------------

                angle_change = (
                    servo_velocity
                    * delta_time
                )

                requested_angle = (
                    current_angle
                    + angle_change
                )


                # ---------------------------------------------
                # 최소각도 제한
                # ---------------------------------------------

                if requested_angle < MIN_ANGLE:

                    current_angle = set_servo_angle_safe(
                        MIN_ANGLE
                    )

                    angle_limit_active = True

                    # 최소각도보다 더 낮게 가려는 속도만 제거
                    if servo_velocity < 0.0:
                        servo_velocity = 0.0

                    if previous_limit_state != "MIN":

                        print(
                            f"[SAFETY] 최소각도 "
                            f"{MIN_ANGLE:.1f}도에서 "
                            "아래 방향 이동을 차단합니다."
                        )

                        previous_limit_state = "MIN"


                # ---------------------------------------------
                # 최대각도 제한
                # ---------------------------------------------

                elif requested_angle > MAX_ANGLE:

                    current_angle = set_servo_angle_safe(
                        MAX_ANGLE
                    )

                    angle_limit_active = True

                    # 최대각도보다 더 높게 가려는 속도만 제거
                    if servo_velocity > 0.0:
                        servo_velocity = 0.0

                    if previous_limit_state != "MAX":

                        print(
                            f"[SAFETY] 최대각도 "
                            f"{MAX_ANGLE:.1f}도에서 "
                            "위 방향 이동을 차단합니다."
                        )

                        previous_limit_state = "MAX"


                # ---------------------------------------------
                # 안전범위 안이면 정상 이동
                # ---------------------------------------------

                else:

                    current_angle = set_servo_angle_safe(
                        requested_angle
                    )

                    previous_limit_state = None


                previous_error_y = error_y
                last_servo_update_time = current_time


            # ------------------------------------------------
            # 중앙 정렬 상태 표시
            # ------------------------------------------------

            if abs(error_y) <= DEAD_ZONE_Y:

                cv2.putText(
                    frame,
                    "CUP CENTERED",
                    (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


            # ------------------------------------------------
            # 컵 탐지 정보 표시
            # ------------------------------------------------

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

            cv2.putText(
                frame,
                f"P: {p_term:.2f} / D: {d_term:.2f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 200, 0),
                2
            )


        # ====================================================
        # 초기 위치 복귀 상태
        # ====================================================

        elif returning_to_initial:

            # 컵이 없으므로 탐지 및 PD 상태 초기화
            filtered_target_y = None
            previous_error_y = None
            filtered_derivative_y = 0.0
            angle_limit_active = False
            previous_limit_state = None

            if (
                current_time - last_servo_update_time
                >= RETURN_INTERVAL
            ):

                return_error = (
                    INITIAL_ANGLE
                    - current_angle
                )

                target_return_velocity = clamp(
                    return_error * RETURN_KP,
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
                    + servo_velocity
                    * delta_time
                )


                # 초기각도를 넘어가지 않도록 제한
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


            # 초기 위치 도착 여부 확인
            if (
                abs(
                    current_angle
                    - INITIAL_ANGLE
                ) <= 0.08
                and abs(servo_velocity) <= 0.35
            ):

                current_angle = set_servo_angle_safe(
                    INITIAL_ANGLE
                )

                servo_velocity = 0.0
                returning_to_initial = False

                # 복귀 직후 즉시 다시 복귀 상태가 되는 것을 방지
                last_detection_time = current_time

                print(
                    f"[INFO] 초기 위치 "
                    f"{INITIAL_ANGLE:.1f}도 복귀 완료"
                )


            cv2.putText(
                frame,
                "RETURNING TO INITIAL POSITION",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2
            )


        # ====================================================
        # 컵이 탐지되지 않은 상태
        # ====================================================

        else:

            filtered_target_y = None
            previous_error_y = None
            filtered_derivative_y = 0.0
            angle_limit_active = False
            previous_limit_state = None

            elapsed_time = (
                current_time
                - last_detection_time
            )


            # ------------------------------------------------
            # 컵이 사라지면 현재 속도를 부드럽게 줄입니다.
            # ------------------------------------------------

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


            # ------------------------------------------------
            # 컵이 일정 시간 이상 없고,
            # 현재 위치가 초기 위치와 다를 때만 복귀합니다.
            # ------------------------------------------------

            if (
                elapsed_time >= LOST_RETURN_TIME
                and abs(
                    current_angle
                    - INITIAL_ANGLE
                ) > 0.08
            ):

                servo_velocity = 0.0
                returning_to_initial = True

                print(
                    "[INFO] 컵이 일정 시간 탐지되지 않아 "
                    "초기 위치로 복귀합니다."
                )


        # ====================================================
        # 상태 문자열 결정
        # ====================================================

        if returning_to_initial:

            status_text = "CUP LOST - RETURNING"

        elif cup is not None:

            if angle_limit_active:
                status_text = "ANGLE LIMIT BLOCKED"
            elif abs(error_y) <= DEAD_ZONE_Y:
                status_text = "CUP CENTERED"
            else:
                status_text = "PD CUP TRACKING"

        else:

            status_text = "SEARCHING CUP"


        # ====================================================
        # 공통 상태 화면 표시
        # ====================================================

        cv2.putText(
            frame,
            f"Resolution: {frame_width} x {frame_height}",
            (
                max(10, frame_width - 340),
                30
            ),
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


        # ====================================================
        # 영상 표시
        # ====================================================

        cv2.imshow(
            "YOLO HD PD Vertical Cup Tracking",
            frame
        )


        # q 또는 ESC로 종료
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break


except KeyboardInterrupt:

    print(
        "\n[INFO] 사용자가 프로그램을 중단했습니다."
    )


except Exception as main_error:

    print(
        f"\n[ERROR] 프로그램 실행 중 오류 발생: "
        f"{main_error}"
    )

    raise


finally:

    # ========================================================
    # 프로그램 종료 시 초기 위치 복귀
    # ========================================================

    try:

        print(
            f"[INFO] 서보를 초기 위치 "
            f"{INITIAL_ANGLE:.1f}도로 복귀합니다."
        )

        return_velocity = 0.0
        previous_time = time.monotonic()

        while (
            abs(
                current_angle
                - INITIAL_ANGLE
            ) > 0.05
        ):

            now = time.monotonic()

            delta_time = clamp(
                now - previous_time,
                0.001,
                0.1
            )

            previous_time = now

            return_error = (
                INITIAL_ANGLE
                - current_angle
            )

            target_velocity = clamp(
                return_error * RETURN_KP,
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
                + return_velocity
                * delta_time
            )


            # 초기 위치를 넘어가지 않도록 제한
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

            time.sleep(
                RETURN_INTERVAL
            )


        current_angle = set_servo_angle_safe(
            INITIAL_ANGLE
        )

        print(
            "[INFO] 서보 초기 위치 복귀 완료"
        )


    except Exception as servo_return_error:

        print(
            f"[WARNING] 종료 중 서보 복귀 실패: "
            f"{servo_return_error}"
        )


    finally:

        camera.release()
        cv2.destroyAllWindows()

        print("[INFO] 카메라를 종료했습니다.")
        print("[INFO] 프로그램을 종료했습니다.")
