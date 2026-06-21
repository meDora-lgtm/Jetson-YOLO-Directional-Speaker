import time
import math
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

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

IMAGE_SIZE = 1024
CONFIDENCE = 0.25

# COCO 클래스 번호
PERSON_CLASS_ID = 0
CUP_CLASS_ID = 41

PERSON_CLASS_NAME = "person"
CUP_CLASS_NAME = "cup"

# 컵이 순간적으로 탐지되지 않아도 마지막 위치를 유지하는 시간
CUP_HOLD_TIME = 0.4


# ============================================================
# 얼굴 추정 설정
# ============================================================

# 사람 바운딩박스 상단에서 아래쪽으로 16% 지점
FACE_CENTER_RATIO_Y = 0.16

FACE_BOX_WIDTH_RATIO = 0.42
FACE_BOX_HEIGHT_RATIO = 0.24

# 추적 중 사람이 잠깐 사라져도 마지막 위치를 유지하는 시간
PERSON_HOLD_TIME = 0.50


# ============================================================
# 컵과 사람 관계 판단 설정
# ============================================================

# 컵과 사람 중심 사이의 거리를 사람 키로 나눈 값입니다.
#
# 예:
# 거리 250px, 사람 높이 500px
# normalized distance = 250 / 500 = 0.5
#
# 이 값 이하이면 컵과 사람이 붙어 있다고 판단합니다.
CUP_HELD_DISTANCE_RATIO = 0.60

# 등록된 사람이 컵에서 이 값 이상 멀어지면 사람 추적 시작
PERSON_SEPARATION_DISTANCE_RATIO = 1.10

# 컵과 사람이 붙어 있는 상태가 몇 프레임 지속되어야
# 컵을 들고 있던 사람으로 등록할지 결정합니다.
HOLDER_CONFIRM_FRAMES = 10

# 사람이 컵에서 떨어진 상태가 몇 프레임 지속되어야
# 실제 분리된 것으로 판단할지 결정합니다.
SEPARATION_CONFIRM_FRAMES = 10

# 추적 중인 사람이 다시 컵 가까이 돌아온 상태가
# 몇 프레임 지속되어야 컵을 다시 가져간 것으로 판단할지 결정합니다.
RETURN_TO_CUP_CONFIRM_FRAMES = 4

# 등록한 사람을 다음 프레임에서 찾을 때 허용하는 이동 거리
# 이전 사람 중심과 현재 사람 중심 거리 / 이전 사람 높이
PERSON_MATCH_DISTANCE_RATIO = 0.75

# 추적 대상 사람이 잠깐 사라져도 기억하는 시간
TARGET_PERSON_MEMORY_TIME = 1.0

# 추적 완료 후 자동 초기화 여부
# False이면 한 번 선택한 사람을 계속 추적합니다.
AUTO_RESET_TRACKING = False


# ============================================================
# PCA9685 및 서보 설정
# ============================================================

PCA9685_ADDRESS = 0x41
SERVO_CHANNEL = 0

MIN_ANGLE = 125.0
MAX_ANGLE = 180.0

INITIAL_ANGLE = 125.5


# ============================================================
# 화면 중심 제어
# ============================================================

DEAD_ZONE_Y = 10
SLOW_ZONE_Y = 120

CENTER_VELOCITY_DAMPING = 0.50


# ============================================================
# PD 제어 설정
# ============================================================

PD_KP = 0.05

# 기존 0.002에서 약간 증가
PD_KD = 0.04

# 낮을수록 D값이 더 부드럽게 변합니다.
# 기존 0.12에서 0.08로 변경
DERIVATIVE_FILTER_ALPHA = 0.08

# 순간적으로 큰 D항이 발생하는 것을 제한
MAX_D_TERM = 6.0

# 한 프레임에서 미분값이 비정상적으로 튀는 것을 제한
MAX_RAW_DERIVATIVE = 1800.0


# ============================================================
# 서보 움직임 설정
# ============================================================

# 기존 20도/s에서 30% 증가
MAX_SERVO_SPEED = 26.0

# 기존 65도/s²에서 약 30% 증가
MAX_SERVO_ACCELERATION = 84.5

TARGET_FILTER_ALPHA = 0.32

SERVO_UPDATE_INTERVAL = 0.015

HELD_TARGET_SPEED_RATIO = 0.25


# ============================================================
# 사람 소실 및 초기 위치 복귀
# ============================================================

LOST_RETURN_TIME = 2.0

# 기존 14에서 30% 증가
RETURN_MAX_SPEED = 18.2

# 기존 45에서 30% 증가
RETURN_ACCELERATION = 58.5

RETURN_INTERVAL = 0.015
RETURN_KP = 3.5


# ============================================================
# 서보 방향
# ============================================================

# 얼굴이 화면 아래에 있을 때 카메라가 반대로 움직이면
# -1을 1로 변경하십시오.
SERVO_DIRECTION = -1


# ============================================================
# I2C 장치 탐지 우회
# ============================================================

original_i2c_device_init = i2c_device.I2CDevice.__init__


def no_probe_init(self, i2c, device_address, probe=True):
    original_i2c_device_init(
        self,
        i2c,
        device_address,
        probe=False
    )


i2c_device.I2CDevice.__init__ = no_probe_init


# ============================================================
# 공통 함수
# ============================================================

def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(float(value), maximum)
    )


def clamp_angle(angle):
    return clamp(
        angle,
        MIN_ANGLE,
        MAX_ANGLE
    )


def update_velocity_smoothly(
    current_velocity,
    target_velocity,
    max_acceleration,
    delta_time
):
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


def box_center(detection):
    x1, y1, x2, y2, confidence = detection

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def box_width(detection):
    return max(
        1.0,
        detection[2] - detection[0]
    )


def box_height(detection):
    return max(
        1.0,
        detection[3] - detection[1]
    )


def detection_area(detection):
    return (
        box_width(detection)
        * box_height(detection)
    )


def center_distance(first_detection, second_detection):
    first_x, first_y = box_center(
        first_detection
    )

    second_x, second_y = box_center(
        second_detection
    )

    return math.hypot(
        first_x - second_x,
        first_y - second_y
    )


def normalized_person_cup_distance(person, cup):
    """
    사람과 컵 중심 거리를 사람 키로 나눈 값입니다.

    카메라와의 거리가 바뀌어도 픽셀 거리만 사용하는 것보다
    비교적 일정하게 판단할 수 있습니다.
    """

    distance = center_distance(
        person,
        cup
    )

    person_height = box_height(
        person
    )

    return distance / person_height


def normalized_person_match_distance(
    previous_person,
    current_person
):
    """
    이전 프레임 사람과 현재 프레임 사람의 중심 거리입니다.
    이전 사람 키를 기준으로 정규화합니다.
    """

    distance = center_distance(
        previous_person,
        current_person
    )

    reference_height = box_height(
        previous_person
    )

    return distance / reference_height


def get_detections(boxes):
    persons = []
    cups = []

    if boxes is None:
        return persons, cups

    for box in boxes:
        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        x1, y1, x2, y2 = (
            box.xyxy[0].cpu().tolist()
        )

        detection = (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
            confidence
        )

        if class_id == PERSON_CLASS_ID:
            persons.append(
                detection
            )

        elif class_id == CUP_CLASS_ID:
            cups.append(
                detection
            )

    return persons, cups


def find_closest_person_cup_pair(persons, cups):
    """
    화면에 있는 모든 사람-컵 조합 중 가장 가까운 조합을 찾습니다.
    """

    selected_person = None
    selected_cup = None
    selected_distance_ratio = None

    for person in persons:
        for cup in cups:
            distance_ratio = (
                normalized_person_cup_distance(
                    person,
                    cup
                )
            )

            if (
                selected_distance_ratio is None
                or distance_ratio
                < selected_distance_ratio
            ):
                selected_person = person
                selected_cup = cup
                selected_distance_ratio = (
                    distance_ratio
                )

    return (
        selected_person,
        selected_cup,
        selected_distance_ratio
    )


def find_matching_person(
    previous_person,
    current_persons
):
    """
    이전 위치와 가장 가까운 사람을 찾아 동일 인물로 유지합니다.
    """

    if previous_person is None:
        return None, None

    best_person = None
    best_distance_ratio = None

    for person in current_persons:
        distance_ratio = (
            normalized_person_match_distance(
                previous_person,
                person
            )
        )

        if (
            best_distance_ratio is None
            or distance_ratio
            < best_distance_ratio
        ):
            best_person = person
            best_distance_ratio = (
                distance_ratio
            )

    if (
        best_person is not None
        and best_distance_ratio
        <= PERSON_MATCH_DISTANCE_RATIO
    ):
        return (
            best_person,
            best_distance_ratio
        )

    return None, best_distance_ratio


def find_nearest_cup_to_person(person, cups):
    if person is None or not cups:
        return None, None

    nearest_cup = None
    nearest_distance_ratio = None

    for cup in cups:
        distance_ratio = (
            normalized_person_cup_distance(
                person,
                cup
            )
        )

        if (
            nearest_distance_ratio is None
            or distance_ratio
            < nearest_distance_ratio
        ):
            nearest_cup = cup
            nearest_distance_ratio = (
                distance_ratio
            )

    return nearest_cup, nearest_distance_ratio


def get_face_target(person):
    x1, y1, x2, y2, confidence = person

    person_width = max(
        1,
        x2 - x1
    )

    person_height = max(
        1,
        y2 - y1
    )

    face_center_x = (
        x1 + x2
    ) / 2.0

    face_center_y = (
        y1
        + person_height
        * FACE_CENTER_RATIO_Y
    )

    face_box_width = (
        person_width
        * FACE_BOX_WIDTH_RATIO
    )

    face_box_height = (
        person_height
        * FACE_BOX_HEIGHT_RATIO
    )

    face_x1 = int(
        face_center_x
        - face_box_width / 2
    )

    face_x2 = int(
        face_center_x
        + face_box_width / 2
    )

    face_y1 = int(
        face_center_y
        - face_box_height / 2
    )

    face_y2 = int(
        face_center_y
        + face_box_height / 2
    )

    return {
        "center_x": face_center_x,
        "center_y": face_center_y,
        "box": (
            face_x1,
            face_y1,
            face_x2,
            face_y2
        ),
        "confidence": confidence
    }


def draw_detection(
    frame,
    detection,
    label,
    color,
    thickness=2
):
    x1, y1, x2, y2, confidence = detection

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    cv2.putText(
        frame,
        f"{label}: {confidence:.2f}",
        (
            x1,
            max(25, y1 - 8)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        color,
        2
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


def set_servo_angle_safe(angle):
    safe_angle = clamp_angle(
        angle
    )

    kit.servo[SERVO_CHANNEL].angle = (
        safe_angle
    )

    return safe_angle


# ============================================================
# YOLO 모델
# ============================================================

print("[INFO] YOLO 모델을 불러옵니다.")

model = YOLO(
    MODEL_PATH,
    task="detect"
)

print("[INFO] 탐지 대상: PERSON + CUP")
print("[INFO] 컵에서 멀어진 컵 소유자를 추적합니다.")
print(
    f"[INFO] 분리 판단 거리 비율: "
    f"{PERSON_SEPARATION_DISTANCE_RATIO:.2f}"
)


# ============================================================
# 카메라
# ============================================================

camera = cv2.VideoCapture(
    CAMERA_DEVICE,
    cv2.CAP_V4L2
)

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
        f"카메라를 열 수 없습니다: "
        f"{CAMERA_DEVICE}"
    )


actual_width = int(
    camera.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

actual_height = int(
    camera.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
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
# 초기 상태
# ============================================================

current_angle = set_servo_angle_safe(
    INITIAL_ANGLE
)

time.sleep(1.0)

current_time = time.monotonic()

last_servo_update_time = current_time
last_target_detection_time = current_time

servo_velocity = 0.0

filtered_target_y = None
previous_error_y = None
filtered_derivative_y = 0.0

returning_to_initial = False
angle_limit_active = False
previous_limit_state = None

last_valid_face = None

# 마지막으로 정상 탐지된 컵 정보
last_valid_cup = None
last_cup_detection_time = 0.0
cup_target_held = False

# 상태 종류:
# WAITING_PAIR      : 컵과 사람이 가까워지기를 기다림
# MONITORING_HOLDER : 컵 소유자로 등록하고 거리 감시
# TRACKING_PERSON   : 사람이 컵에서 멀어진 뒤 얼굴 추적
tracking_state = "WAITING_PAIR"

holder_confirm_count = 0
separation_confirm_count = 0
return_to_cup_confirm_count = 0

registered_person = None
registered_cup = None

last_registered_person_time = 0.0

current_distance_ratio = None
person_match_ratio = None


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

        frame_height, frame_width = (
            frame.shape[:2]
        )

        screen_center_x = (
            frame_width // 2
        )

        screen_center_y = (
            frame_height // 2
        )

        current_time = time.monotonic()

        delta_time = clamp(
            current_time
            - last_servo_update_time,
            0.001,
            0.1
        )


        # ====================================================
        # 사람과 컵 탐지
        # ====================================================

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            classes=[
                PERSON_CLASS_ID,
                CUP_CLASS_ID
            ],
            verbose=False
        )

        persons, cups = get_detections(
            results[0].boxes
        )


        # ====================================================
        # 컵 탐지 결과 0.5초 유지
        # ====================================================

        cup_target_held = False

        if cups:
            # 여러 컵 중 신뢰도가 가장 높은 컵을 마지막 유효 컵으로 저장
            best_detected_cup = max(
                cups,
                key=lambda detection: detection[4]
            )

            last_valid_cup = best_detected_cup
            last_cup_detection_time = current_time

        elif (
            last_valid_cup is not None
            and current_time - last_cup_detection_time
            <= CUP_HOLD_TIME
        ):
            # 이번 프레임에서 컵이 사라져도 0.5초 동안
            # 마지막 위치를 현재 탐지 결과처럼 사용
            cups = [last_valid_cup]
            cup_target_held = True

        else:
            # 유지 시간이 지난 뒤에는 컵 소실 처리
            last_valid_cup = None


        # 모든 탐지 결과 표시
        for person in persons:
            draw_detection(
                frame,
                person,
                "PERSON",
                (255, 120, 0),
                2
            )

        for cup in cups:
            if cup_target_held:
                cup_label = "CUP HELD"
                cup_color = (0, 165, 255)
            else:
                cup_label = "CUP"
                cup_color = (0, 255, 0)

            draw_detection(
                frame,
                cup,
                cup_label,
                cup_color,
                2
            )


        # ====================================================
        # 상태 1: 컵과 가까운 사람 찾기
        # ====================================================

        if tracking_state == "WAITING_PAIR":
            (
                candidate_person,
                candidate_cup,
                candidate_distance_ratio
            ) = find_closest_person_cup_pair(
                persons,
                cups
            )

            current_distance_ratio = (
                candidate_distance_ratio
            )

            if (
                candidate_person is not None
                and candidate_cup is not None
                and candidate_distance_ratio
                <= CUP_HELD_DISTANCE_RATIO
            ):
                holder_confirm_count += 1

                registered_person = (
                    candidate_person
                )

                registered_cup = (
                    candidate_cup
                )

                last_registered_person_time = (
                    current_time
                )

                if (
                    holder_confirm_count
                    >= HOLDER_CONFIRM_FRAMES
                ):
                    tracking_state = (
                        "MONITORING_HOLDER"
                    )

                    holder_confirm_count = 0
                    separation_confirm_count = 0

                    print(
                        "[INFO] 컵을 들고 있던 "
                        "사람을 등록했습니다."
                    )

            else:
                holder_confirm_count = 0


        # ====================================================
        # 상태 2: 등록된 사람과 컵의 거리 감시
        # ====================================================

        elif tracking_state == "MONITORING_HOLDER":
            matched_person, person_match_ratio = (
                find_matching_person(
                    registered_person,
                    persons
                )
            )

            if matched_person is not None:
                registered_person = (
                    matched_person
                )

                last_registered_person_time = (
                    current_time
                )

                nearest_cup, cup_distance_ratio = (
                    find_nearest_cup_to_person(
                        registered_person,
                        cups
                    )
                )

                if nearest_cup is not None:
                    registered_cup = nearest_cup
                    current_distance_ratio = (
                        cup_distance_ratio
                    )

                    if (
                        cup_distance_ratio
                        >= PERSON_SEPARATION_DISTANCE_RATIO
                    ):
                        separation_confirm_count += 1

                    else:
                        separation_confirm_count = 0

                else:
                    # 컵이 잠깐 탐지되지 않았다고 바로
                    # 분리된 것으로 판단하지 않습니다.
                    current_distance_ratio = None
                    separation_confirm_count = max(
                        0,
                        separation_confirm_count - 1
                    )

                if (
                    separation_confirm_count
                    >= SEPARATION_CONFIRM_FRAMES
                ):
                    tracking_state = (
                        "TRACKING_PERSON"
                    )

                    separation_confirm_count = 0
                    return_to_cup_confirm_count = 0

                    last_target_detection_time = (
                        current_time
                    )

                    returning_to_initial = False

                    print(
                        "[INFO] 사람이 컵에서 "
                        "멀어졌습니다."
                    )

                    print(
                        "[INFO] 등록된 사람의 "
                        "얼굴 추적을 시작합니다."
                    )

            else:
                # 등록 대상이 잠깐 가려진 경우 기억
                if (
                    current_time
                    - last_registered_person_time
                    > TARGET_PERSON_MEMORY_TIME
                ):
                    print(
                        "[INFO] 등록된 사람을 "
                        "놓쳤습니다. 다시 탐색합니다."
                    )

                    tracking_state = (
                        "WAITING_PAIR"
                    )

                    registered_person = None
                    registered_cup = None

                    holder_confirm_count = 0
                    separation_confirm_count = 0
                    current_distance_ratio = None


        # ====================================================
        # 상태 3: 컵에서 멀어진 사람 추적
        # ====================================================

        elif tracking_state == "TRACKING_PERSON":
            matched_person, person_match_ratio = (
                find_matching_person(
                    registered_person,
                    persons
                )
            )

            if matched_person is not None:
                registered_person = matched_person
                last_registered_person_time = current_time

                # 추적 중인 사람이 다시 컵 가까이 돌아왔는지 확인
                nearest_cup, cup_distance_ratio = (
                    find_nearest_cup_to_person(
                        registered_person,
                        cups
                    )
                )

                if nearest_cup is not None:
                    registered_cup = nearest_cup
                    current_distance_ratio = cup_distance_ratio

                    if (
                        cup_distance_ratio
                        <= CUP_HELD_DISTANCE_RATIO
                    ):
                        return_to_cup_confirm_count += 1
                    else:
                        return_to_cup_confirm_count = 0

                else:
                    return_to_cup_confirm_count = 0
                    current_distance_ratio = None

                # 사람이 다시 컵을 들고 있다고 판단되면
                # 얼굴 추적을 멈추고 거리 감시 상태로 복귀
                if (
                    return_to_cup_confirm_count
                    >= RETURN_TO_CUP_CONFIRM_FRAMES
                ):
                    tracking_state = "MONITORING_HOLDER"

                    return_to_cup_confirm_count = 0
                    separation_confirm_count = 0

                    last_valid_face = None
                    filtered_target_y = None
                    previous_error_y = None
                    filtered_derivative_y = 0.0

                    servo_velocity = 0.0
                    returning_to_initial = False

                    print(
                        "[INFO] 추적 대상이 다시 컵 가까이 "
                        "돌아왔습니다."
                    )

                    print(
                        "[INFO] 컵을 다시 가져간 것으로 판단하여 "
                        "얼굴 추적을 중지합니다."
                    )

            else:
                return_to_cup_confirm_count = 0


        # ====================================================
        # 추적할 얼굴 목표 생성
        # ====================================================

        face_target = None
        target_held = False

        if (
            tracking_state == "TRACKING_PERSON"
            and registered_person is not None
        ):
            matched_person, person_match_ratio = (
                find_matching_person(
                    registered_person,
                    persons
                )
            )

            if matched_person is not None:
                registered_person = matched_person

                face_target = get_face_target(
                    matched_person
                )

                last_valid_face = face_target

                last_target_detection_time = (
                    current_time
                )

                last_registered_person_time = (
                    current_time
                )

                returning_to_initial = False

            elif (
                last_valid_face is not None
                and current_time
                - last_target_detection_time
                <= PERSON_HOLD_TIME
            ):
                face_target = last_valid_face
                target_held = True

            else:
                last_valid_face = None


        # ====================================================
        # 화면 중앙선
        # ====================================================

        cv2.line(
            frame,
            (0, screen_center_y),
            (frame_width, screen_center_y),
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (
                0,
                screen_center_y
                - DEAD_ZONE_Y
            ),
            (
                frame_width,
                screen_center_y
                - DEAD_ZONE_Y
            ),
            (100, 100, 100),
            1
        )

        cv2.line(
            frame,
            (
                0,
                screen_center_y
                + DEAD_ZONE_Y
            ),
            (
                frame_width,
                screen_center_y
                + DEAD_ZONE_Y
            ),
            (100, 100, 100),
            1
        )

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


        error_y = None
        p_term = 0.0
        d_term = 0.0


        # ====================================================
        # 얼굴 PD 추적
        # ====================================================

        if face_target is not None:
            angle_limit_active = False

            face_center_x = (
                face_target["center_x"]
            )

            raw_face_center_y = (
                face_target["center_y"]
            )

            face_x1, face_y1, face_x2, face_y2 = (
                face_target["box"]
            )

            face_color = (
                (0, 165, 255)
                if target_held
                else (0, 0, 255)
            )

            cv2.rectangle(
                frame,
                (
                    max(0, face_x1),
                    max(0, face_y1)
                ),
                (
                    min(
                        frame_width - 1,
                        face_x2
                    ),
                    min(
                        frame_height - 1,
                        face_y2
                    )
                ),
                face_color,
                2
            )

            cv2.circle(
                frame,
                (
                    int(face_center_x),
                    int(raw_face_center_y)
                ),
                9,
                face_color,
                -1
            )

            cv2.putText(
                frame,
                (
                    "TARGET HELD"
                    if target_held
                    else "TARGET PERSON FACE"
                ),
                (
                    max(0, face_x1),
                    max(25, face_y1 - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                face_color,
                2
            )


            # -----------------------------------------------
            # 목표 위치 저역 통과 필터
            # -----------------------------------------------

            if filtered_target_y is None:
                filtered_target_y = (
                    raw_face_center_y
                )

            elif not target_held:
                filtered_target_y = (
                    TARGET_FILTER_ALPHA
                    * raw_face_center_y
                    + (
                        1.0
                        - TARGET_FILTER_ALPHA
                    )
                    * filtered_target_y
                )


            error_y = (
                filtered_target_y
                - screen_center_y
            )

            cv2.line(
                frame,
                (
                    screen_center_x,
                    screen_center_y
                ),
                (
                    int(face_center_x),
                    int(filtered_target_y)
                ),
                (255, 0, 255),
                2
            )


            # -----------------------------------------------
            # PD 제어
            # -----------------------------------------------

            if (
                current_time
                - last_servo_update_time
                >= SERVO_UPDATE_INTERVAL
            ):
                target_velocity = 0.0

                if abs(error_y) <= DEAD_ZONE_Y:
                    servo_velocity *= (
                        CENTER_VELOCITY_DAMPING
                    )

                    filtered_derivative_y *= 0.45

                    if abs(servo_velocity) < 0.08:
                        servo_velocity = 0.0

                else:
                    if error_y > 0:
                        effective_error = (
                            error_y
                            - DEAD_ZONE_Y
                        )

                    else:
                        effective_error = (
                            error_y
                            + DEAD_ZONE_Y
                        )

                    if (
                        target_held
                        or previous_error_y is None
                    ):
                        raw_derivative_y = 0.0

                    else:
                        raw_derivative_y = (
                            error_y
                            - previous_error_y
                        ) / delta_time

                    raw_derivative_y = clamp(
                        raw_derivative_y,
                        -MAX_RAW_DERIVATIVE,
                        MAX_RAW_DERIVATIVE
                    )

                    if not target_held:
                        filtered_derivative_y = (
                            DERIVATIVE_FILTER_ALPHA
                            * raw_derivative_y
                            + (
                                1.0
                                - DERIVATIVE_FILTER_ALPHA
                            )
                            * filtered_derivative_y
                        )

                    p_term = (
                        PD_KP
                        * effective_error
                    )

                    d_term = clamp(
                        PD_KD
                        * filtered_derivative_y,
                        -MAX_D_TERM,
                        MAX_D_TERM
                    )

                    if target_held:
                        d_term = 0.0

                    target_velocity = (
                        (p_term + d_term)
                        * SERVO_DIRECTION
                    )

                    target_velocity = clamp(
                        target_velocity,
                        -MAX_SERVO_SPEED,
                        MAX_SERVO_SPEED
                    )


                    # 중앙 근처 감속
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
                            0.22,
                            1.0
                        )

                        target_velocity *= (
                            slow_ratio
                        )


                    if target_held:
                        target_velocity *= (
                            HELD_TARGET_SPEED_RATIO
                        )


                servo_velocity = (
                    update_velocity_smoothly(
                        current_velocity=(
                            servo_velocity
                        ),
                        target_velocity=(
                            target_velocity
                        ),
                        max_acceleration=(
                            MAX_SERVO_ACCELERATION
                        ),
                        delta_time=delta_time
                    )
                )

                requested_angle = (
                    current_angle
                    + servo_velocity
                    * delta_time
                )


                # -------------------------------------------
                # 안전각 제한
                # -------------------------------------------

                if requested_angle < MIN_ANGLE:
                    current_angle = (
                        set_servo_angle_safe(
                            MIN_ANGLE
                        )
                    )

                    angle_limit_active = True

                    if servo_velocity < 0:
                        servo_velocity = 0.0

                    if previous_limit_state != "MIN":
                        print(
                            f"[SAFETY] 최소각도 "
                            f"{MIN_ANGLE:.1f}도 제한"
                        )

                        previous_limit_state = "MIN"

                elif requested_angle > MAX_ANGLE:
                    current_angle = (
                        set_servo_angle_safe(
                            MAX_ANGLE
                        )
                    )

                    angle_limit_active = True

                    if servo_velocity > 0:
                        servo_velocity = 0.0

                    if previous_limit_state != "MAX":
                        print(
                            f"[SAFETY] 최대각도 "
                            f"{MAX_ANGLE:.1f}도 제한"
                        )

                        previous_limit_state = "MAX"

                else:
                    current_angle = (
                        set_servo_angle_safe(
                            requested_angle
                        )
                    )

                    previous_limit_state = None

                if not target_held:
                    previous_error_y = error_y

                last_servo_update_time = (
                    current_time
                )


        # ====================================================
        # 얼굴 추적 대상 없음
        # ====================================================

        else:
            filtered_target_y = None
            previous_error_y = None
            filtered_derivative_y = 0.0

            angle_limit_active = False
            previous_limit_state = None

            if (
                current_time
                - last_servo_update_time
                >= SERVO_UPDATE_INTERVAL
            ):
                servo_velocity = (
                    update_velocity_smoothly(
                        current_velocity=(
                            servo_velocity
                        ),
                        target_velocity=0.0,
                        max_acceleration=(
                            MAX_SERVO_ACCELERATION
                        ),
                        delta_time=delta_time
                    )
                )

                if abs(servo_velocity) < 0.05:
                    servo_velocity = 0.0

                last_servo_update_time = (
                    current_time
                )


            # 추적을 시작한 뒤 사람을 놓친 경우에만 복귀
            if tracking_state == "TRACKING_PERSON":
                elapsed_lost_time = (
                    current_time
                    - last_target_detection_time
                )

                if (
                    elapsed_lost_time
                    >= LOST_RETURN_TIME
                    and abs(
                        current_angle
                        - INITIAL_ANGLE
                    ) > 0.08
                ):
                    returning_to_initial = True


        # ====================================================
        # 초기 위치 복귀
        # ====================================================

        if (
            returning_to_initial
            and face_target is None
        ):
            if (
                current_time
                - last_servo_update_time
                >= RETURN_INTERVAL
            ):
                return_error = (
                    INITIAL_ANGLE
                    - current_angle
                )

                target_return_velocity = clamp(
                    return_error
                    * RETURN_KP,
                    -RETURN_MAX_SPEED,
                    RETURN_MAX_SPEED
                )

                servo_velocity = (
                    update_velocity_smoothly(
                        current_velocity=(
                            servo_velocity
                        ),
                        target_velocity=(
                            target_return_velocity
                        ),
                        max_acceleration=(
                            RETURN_ACCELERATION
                        ),
                        delta_time=delta_time
                    )
                )

                next_angle = (
                    current_angle
                    + servo_velocity
                    * delta_time
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

                current_angle = (
                    set_servo_angle_safe(
                        next_angle
                    )
                )

                last_servo_update_time = (
                    current_time
                )

            if (
                abs(
                    current_angle
                    - INITIAL_ANGLE
                ) <= 0.08
                and abs(
                    servo_velocity
                ) <= 0.35
            ):
                current_angle = (
                    set_servo_angle_safe(
                        INITIAL_ANGLE
                    )
                )

                servo_velocity = 0.0
                returning_to_initial = False

                if AUTO_RESET_TRACKING:
                    tracking_state = (
                        "WAITING_PAIR"
                    )

                    registered_person = None
                    registered_cup = None
                    last_valid_face = None


        # ====================================================
        # 등록된 사람과 컵 연결선
        # ====================================================

        if (
            registered_person is not None
            and registered_cup is not None
        ):
            person_center = box_center(
                registered_person
            )

            cup_center = box_center(
                registered_cup
            )

            cv2.line(
                frame,
                (
                    int(person_center[0]),
                    int(person_center[1])
                ),
                (
                    int(cup_center[0]),
                    int(cup_center[1])
                ),
                (0, 255, 255),
                2
            )

            cv2.circle(
                frame,
                (
                    int(person_center[0]),
                    int(person_center[1])
                ),
                6,
                (0, 255, 255),
                -1
            )

            cv2.circle(
                frame,
                (
                    int(cup_center[0]),
                    int(cup_center[1])
                ),
                6,
                (0, 255, 255),
                -1
            )


        # ====================================================
        # 상태 표시
        # ====================================================

        if tracking_state == "WAITING_PAIR":
            status_text = (
                "WAITING PERSON + CUP"
            )

        elif tracking_state == "MONITORING_HOLDER":
            status_text = (
                "TRACKING STOPPED - MONITORING HOLDER"
            )

        elif returning_to_initial:
            status_text = (
                "TARGET LOST - RETURNING"
            )

        elif face_target is not None and target_held:
            status_text = (
                "PERSON TEMPORARILY HELD"
            )

        elif face_target is not None:
            if angle_limit_active:
                status_text = (
                    "TRACKING PERSON - ANGLE LIMIT"
                )

            elif (
                error_y is not None
                and abs(error_y)
                <= DEAD_ZONE_Y
            ):
                status_text = (
                    "PERSON FACE CENTERED"
                )

            else:
                status_text = (
                    "TRACKING CUP OWNER"
                )

        else:
            status_text = (
                "SEARCHING REGISTERED PERSON"
            )


        cv2.putText(
            frame,
            f"Status: {status_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            (
                f"Persons: {len(persons)} / "
                f"Cups: {len(cups)}"
            ),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (0, 255, 255),
            2
        )

        if current_distance_ratio is not None:
            cv2.putText(
                frame,
                (
                    f"Person-Cup distance ratio: "
                    f"{current_distance_ratio:.2f}"
                ),
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                (
                    f"Separation threshold: "
                    f"{PERSON_SEPARATION_DISTANCE_RATIO:.2f}"
                ),
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (0, 255, 255),
                2
            )

        if error_y is not None:
            cv2.putText(
                frame,
                (
                    f"Face error Y: "
                    f"{error_y:.1f}px"
                ),
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (255, 0, 255),
                2
            )

            cv2.putText(
                frame,
                (
                    f"Servo speed: "
                    f"{servo_velocity:.2f} deg/s"
                ),
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (255, 0, 255),
                2
            )

            cv2.putText(
                frame,
                (
                    f"P: {p_term:.2f} / "
                    f"D: {d_term:.2f}"
                ),
                (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (255, 200, 0),
                2
            )

        cv2.putText(
            frame,
            (
                f"Servo: "
                f"{current_angle:.2f} deg"
            ),
            (
                10,
                frame_height - 65
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            (
                f"Safe range: "
                f"{MIN_ANGLE:.0f} - "
                f"{MAX_ANGLE:.0f} deg"
            ),
            (
                10,
                frame_height - 40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            (
                f"Resolution: "
                f"{frame_width} x "
                f"{frame_height}"
            ),
            (
                10,
                frame_height - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 0),
            2
        )


        cv2.imshow(
            "Cup Owner Person Tracking",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("r"):
            tracking_state = "WAITING_PAIR"

            holder_confirm_count = 0
            separation_confirm_count = 0
            return_to_cup_confirm_count = 0

            registered_person = None
            registered_cup = None
            last_valid_face = None
            last_valid_cup = None
            last_cup_detection_time = 0.0
            cup_target_held = False

            current_distance_ratio = None
            person_match_ratio = None

            returning_to_initial = False

            print(
                "[INFO] 추적 상태를 "
                "수동 초기화했습니다."
            )

        if (
            key == ord("q")
            or key == 27
        ):
            break


except KeyboardInterrupt:
    print(
        "\n[INFO] 사용자가 "
        "프로그램을 중단했습니다."
    )


except Exception as main_error:
    print(
        f"\n[ERROR] 프로그램 실행 중 "
        f"오류 발생: {main_error}"
    )

    raise


finally:
    try:
        print(
            f"[INFO] 서보를 초기 위치 "
            f"{INITIAL_ANGLE:.1f}도로 "
            "복귀합니다."
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
                return_error
                * RETURN_KP,
                -RETURN_MAX_SPEED,
                RETURN_MAX_SPEED
            )

            return_velocity = (
                update_velocity_smoothly(
                    current_velocity=(
                        return_velocity
                    ),
                    target_velocity=(
                        target_velocity
                    ),
                    max_acceleration=(
                        RETURN_ACCELERATION
                    ),
                    delta_time=delta_time
                )
            )

            next_angle = (
                current_angle
                + return_velocity
                * delta_time
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

            current_angle = (
                set_servo_angle_safe(
                    next_angle
                )
            )

            time.sleep(
                RETURN_INTERVAL
            )

        current_angle = (
            set_servo_angle_safe(
                INITIAL_ANGLE
            )
        )

        print(
            "[INFO] 서보 초기 위치 "
            "복귀 완료"
        )

    except Exception as servo_return_error:
        print(
            f"[WARNING] 종료 중 "
            f"서보 복귀 실패: "
            f"{servo_return_error}"
        )

    finally:
        camera.release()
        cv2.destroyAllWindows()

        print(
            "[INFO] 카메라를 종료했습니다."
        )

        print(
            "[INFO] 프로그램을 종료했습니다."
        )
