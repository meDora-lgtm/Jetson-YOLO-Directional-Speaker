import time
import math
import os
import glob
import shutil
import subprocess
import threading
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


# ============================================================
# 맥북 UDP 영상 송출 설정
# ============================================================

# 비워 두면 SSH_CLIENT/SSH_CONNECTION에서 접속한 맥북 IP를
# 자동으로 가져옵니다.
#
# 자동 인식이 되지 않으면 다음처럼 직접 입력하십시오.
# MACBOOK_IP = "192.168.0.15"
MACBOOK_IP = ""

STREAM_PORT = 5000
STREAM_FPS = 30
STREAM_JPEG_QUALITY = 80

# False: 젯슨에서는 창을 띄우지 않고 맥북으로만 송출
# True : 젯슨에서도 cv2.imshow 창을 함께 표시
ENABLE_LOCAL_PREVIEW = False

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
# 오디오 방송 설정
# ============================================================

# 스크립트와 같은 폴더에서 오디오 파일을 검색합니다.
AUDIO_FILE = "warning.wav"

# 실제 재생 테스트에서 정상 확인된 AB13X USB Audio 출력 장치
#
# 터미널 확인 명령:
# aplay -D plughw:CARD=Audio,DEV=0 -vv warning.wav
AUDIO_OUTPUT_DEVICE = "plughw:CARD=Audio,DEV=0"

AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.ogg", "*.m4a", "*.aac")

# 추적 대상이 보이는 동안 오디오를 다시 시작하는 간격
AUDIO_REPEAT_INTERVAL = 2.0

# 추적 대상이 이 시간 동안 실제로 탐지되지 않으면
# 팬/틸트를 초기 위치로 복귀하고 추적 상태를 초기화합니다.
NO_ACTIVITY_RESET_TIME = 4.0

# 프로그램 시작 시 오디오 파일과 ALSA 출력 장치를 실제로 열어 봅니다.
# True이면 warning.wav 앞부분이 약 1초 동안 시험 재생됩니다.
AUDIO_STARTUP_TEST = True
AUDIO_STARTUP_TEST_SECONDS = 1.0


# ============================================================
# 얼굴 추정 설정
# ============================================================

# 사람 바운딩박스 상단에서 아래쪽으로 16% 지점을 얼굴 중심으로 사용
FACE_CENTER_RATIO_Y = 0.16

FACE_BOX_WIDTH_RATIO = 0.42
FACE_BOX_HEIGHT_RATIO = 0.24

# 추적 중 사람이 잠깐 사라져도 마지막 얼굴 위치를 유지하는 시간
PERSON_HOLD_TIME = 0.50


# ============================================================
# 컵과 사람 관계 판단 설정
# ============================================================

# 사람-컵 중심 거리 / 사람 바운딩박스 높이
CUP_HELD_DISTANCE_RATIO = 0.60

# 등록된 사람이 컵에서 이 값 이상 멀어지면 얼굴 추적 시작
PERSON_SEPARATION_DISTANCE_RATIO = 1.10

HOLDER_CONFIRM_FRAMES = 10
SEPARATION_CONFIRM_FRAMES = 10
RETURN_TO_CUP_CONFIRM_FRAMES = 4

# 이전 사람과 현재 사람을 동일인으로 판단하는 최대 이동 거리 비율
PERSON_MATCH_DISTANCE_RATIO = 0.75

# 등록한 사람이 잠깐 사라져도 기억하는 시간
TARGET_PERSON_MEMORY_TIME = 1.0

# False: 컵을 다시 가져가면 거리 감시 상태로 돌아감
AUTO_RESET_TRACKING = False


# ============================================================
# PCA9685 및 팬-틸트 서보 설정
# ============================================================

PCA9685_ADDRESS = 0x41

# 0번 채널: 틸트(위/아래)
# 1번 채널: 팬(좌/우)
TILT_SERVO_CHANNEL = 0
PAN_SERVO_CHANNEL = 1

# 0번 틸트 서보 안전 범위
TILT_MIN_ANGLE = 125.0
TILT_MAX_ANGLE = 180.0
TILT_INITIAL_ANGLE = 125.5

# 1번 팬 서보 안전 범위
PAN_MIN_ANGLE = 70.0
PAN_MAX_ANGLE = 230.0
PAN_INITIAL_ANGLE = 150.0

# 230도 명령을 사용하기 위한 서보 전체 동작 범위
SERVO_ACTUATION_RANGE = 270

MIN_PULSE = 500
MAX_PULSE = 2500


# ============================================================
# 화면 중심 제어
# ============================================================

DEAD_ZONE_X = 12
DEAD_ZONE_Y = 12

SLOW_ZONE_X = 120
SLOW_ZONE_Y = 120

# 중앙에 들어왔을 때 속도를 한 프레임 만에 급격히 끊지 않고
# 여러 번에 걸쳐 부드럽게 줄입니다.
CENTER_VELOCITY_DAMPING = 0.72


# ============================================================
# PD 제어 설정
# ============================================================

# 팬과 틸트가 같은 속도 형식을 사용합니다.
PD_KP = 0.8
PD_KD = 0.01

# 낮을수록 미분값 변화가 더 부드럽습니다.
DERIVATIVE_FILTER_ALPHA = 0.001

MAX_D_TERM = 6.0
MAX_RAW_DERIVATIVE = 1800.0


# ============================================================
# 서보 움직임 설정
# ============================================================

# 영상 속 목표를 따라가는 논리적 최대 속도입니다.
# 기존 26 deg/s보다 빠르게 설정하되 가속도 제한을 유지합니다.
MAX_SERVO_SPEED = 42.0
MAX_SERVO_ACCELERATION = 160.0

# 낮을수록 검출 좌표의 흔들림을 더 많이 제거합니다.
# 속도를 올렸을 때 발생하기 쉬운 좌우 떨림을 줄이기 위해 0.20 사용.
TARGET_FILTER_ALPHA = 0.20

# 메인 추론 루프에서 목표 속도를 계산하는 최소 간격입니다.
# 실제 모터 미세 출력 주기는 아래 SERVO_OUTPUT_INTERVAL이 담당합니다.
SERVO_UPDATE_INTERVAL = 0.010

# 마지막 위치를 유지 중일 때 속도 비율
HELD_TARGET_SPEED_RATIO = 0.25


# ============================================================
# 서보 출력 미세 보간 설정
# ============================================================

# YOLO 추론 속도와 관계없이 별도 스레드가 100 Hz로 서보 명령을 갱신합니다.
SERVO_OUTPUT_INTERVAL = 0.010

# 메인 제어가 만든 목표 각도를 실제 출력이 따라가는 최대 속도/가속도입니다.
# 논리적 MAX_SERVO_SPEED보다 약간 높게 두어 출력 지연이 누적되지 않게 합니다.
SERVO_OUTPUT_MAX_SPEED = 65.0
SERVO_OUTPUT_MAX_ACCELERATION = 420.0
SERVO_OUTPUT_POSITION_KP = 14.0

# 지나치게 작은 중복 I2C 쓰기를 줄이는 값입니다.
SERVO_OUTPUT_WRITE_EPSILON = 0.015
SERVO_OUTPUT_SETTLE_ERROR = 0.06


# ============================================================
# 대상 소실 및 초기 위치 복귀 설정
# ============================================================

RETURN_MAX_SPEED = 28.0
RETURN_ACCELERATION = 120.0

RETURN_INTERVAL = 0.010
RETURN_KP = 3.5


# ============================================================
# 서보 방향 설정
# ============================================================

# 얼굴이 화면 아래쪽에 있는데 틸트가 반대로 움직이면
# -1을 1로 변경하십시오.
TILT_SERVO_DIRECTION = -1

# 얼굴이 화면 오른쪽에 있는데 팬이 반대로 움직이면
# 1을 -1로 변경하십시오.
PAN_SERVO_DIRECTION = -1


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


def clamp_tilt_angle(angle):
    return clamp(
        angle,
        TILT_MIN_ANGLE,
        TILT_MAX_ANGLE
    )


def clamp_pan_angle(angle):
    return clamp(
        angle,
        PAN_MIN_ANGLE,
        PAN_MAX_ANGLE
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
                or distance_ratio < selected_distance_ratio
            ):
                selected_person = person
                selected_cup = cup
                selected_distance_ratio = distance_ratio

    return (
        selected_person,
        selected_cup,
        selected_distance_ratio
    )


def find_matching_person(
    previous_person,
    current_persons
):
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
            or distance_ratio < best_distance_ratio
        ):
            best_person = person
            best_distance_ratio = distance_ratio

    if (
        best_person is not None
        and best_distance_ratio <= PERSON_MATCH_DISTANCE_RATIO
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
            or distance_ratio < nearest_distance_ratio
        ):
            nearest_cup = cup
            nearest_distance_ratio = distance_ratio

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


def calculate_axis_target_velocity(
    error,
    previous_error,
    filtered_derivative,
    dead_zone,
    slow_zone,
    direction,
    delta_time,
    target_held
):
    """
    한 축의 PD 목표 속도를 계산합니다.

    반환값:
        target_velocity
        new_filtered_derivative
        p_term
        d_term
    """

    target_velocity = 0.0
    p_term = 0.0
    d_term = 0.0

    if abs(error) <= dead_zone:
        filtered_derivative *= 0.45

        return (
            target_velocity,
            filtered_derivative,
            p_term,
            d_term
        )

    if error > 0:
        effective_error = error - dead_zone
    else:
        effective_error = error + dead_zone

    if target_held or previous_error is None:
        raw_derivative = 0.0
    else:
        raw_derivative = (
            error - previous_error
        ) / delta_time

    raw_derivative = clamp(
        raw_derivative,
        -MAX_RAW_DERIVATIVE,
        MAX_RAW_DERIVATIVE
    )

    if not target_held:
        filtered_derivative = (
            DERIVATIVE_FILTER_ALPHA
            * raw_derivative
            + (
                1.0
                - DERIVATIVE_FILTER_ALPHA
            )
            * filtered_derivative
        )

    p_term = (
        PD_KP
        * effective_error
    )

    d_term = clamp(
        PD_KD
        * filtered_derivative,
        -MAX_D_TERM,
        MAX_D_TERM
    )

    if target_held:
        d_term = 0.0

    target_velocity = (
        (p_term + d_term)
        * direction
    )

    target_velocity = clamp(
        target_velocity,
        -MAX_SERVO_SPEED,
        MAX_SERVO_SPEED
    )

    if abs(error) < slow_zone:
        slow_ratio = (
            abs(error) - dead_zone
        ) / (
            slow_zone - dead_zone
        )

        slow_ratio = clamp(
            slow_ratio,
            0.22,
            1.0
        )

        target_velocity *= slow_ratio

    if target_held:
        target_velocity *= (
            HELD_TARGET_SPEED_RATIO
        )

    return (
        target_velocity,
        filtered_derivative,
        p_term,
        d_term
    )


# ============================================================
# 오디오 재생 함수
# ============================================================

# 최근 오디오 오류를 화면과 터미널에 표시하기 위한 상태입니다.
audio_last_error = None


def find_audio_file():
    script_directory = os.path.dirname(
        os.path.abspath(__file__)
    )

    if AUDIO_FILE:
        candidate = AUDIO_FILE

        if not os.path.isabs(candidate):
            candidate = os.path.join(
                script_directory,
                candidate
            )

        if os.path.isfile(candidate):
            return candidate

        print(
            f"[WARNING] 지정한 오디오 파일을 찾지 못했습니다: "
            f"{candidate}"
        )
        return None

    for pattern in AUDIO_EXTENSIONS:
        matches = sorted(
            glob.glob(
                os.path.join(
                    script_directory,
                    pattern
                )
            )
        )

        if matches:
            return matches[0]

    return None


def build_audio_command(audio_path):
    extension = os.path.splitext(
        audio_path
    )[1].lower()

    if extension == ".wav" and shutil.which("aplay"):
        return [
            "aplay",
            "-D",
            AUDIO_OUTPUT_DEVICE,
            "-q",
            audio_path
        ]

    if shutil.which("ffplay"):
        return [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            audio_path
        ]

    if shutil.which("paplay"):
        return [
            "paplay",
            audio_path
        ]

    return None


def start_audio_playback(audio_path):
    global audio_last_error

    audio_last_error = None

    if audio_path is None:
        audio_last_error = "오디오 파일을 찾지 못했습니다."
        print(
            f"[WARNING] 오디오 재생 실패: {audio_last_error}"
        )
        return None

    command = build_audio_command(
        audio_path
    )

    if command is None:
        audio_last_error = (
            "오디오 재생 프로그램을 찾지 못했습니다. "
            "WAV는 aplay 설치를 권장합니다."
        )

        print(
            f"[WARNING] {audio_last_error}"
        )
        return None

    try:
        audio_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        # 장치를 열지 못한 경우 aplay가 즉시 종료되므로 잠깐 확인합니다.
        time.sleep(0.05)

        return_code = audio_process.poll()

        if (
            return_code is not None
            and return_code != 0
        ):
            error_output = ""

            if audio_process.stderr is not None:
                error_output = (
                    audio_process.stderr.read().strip()
                )

            audio_last_error = (
                error_output
                or f"재생 프로세스 종료 코드 {return_code}"
            )

            print(
                f"[WARNING] 오디오 재생 실패: "
                f"{audio_last_error}"
            )

            return None

        return audio_process

    except Exception as audio_error:
        audio_last_error = str(audio_error)

        print(
            f"[WARNING] 오디오 재생 실패: {audio_last_error}"
        )
        return None


def stop_audio_playback(audio_process):
    if audio_process is None:
        return None

    if audio_process.poll() is None:
        try:
            audio_process.terminate()
            audio_process.wait(timeout=0.3)

        except subprocess.TimeoutExpired:
            audio_process.kill()

        except Exception:
            pass

    return None


def print_alsa_playback_devices():
    """
    오디오 장치 열기에 실패했을 때 ALSA가 인식한 재생 장치를 출력합니다.
    """

    if shutil.which("aplay") is None:
        print(
            "[AUDIO CHECK] aplay 명령을 찾지 못했습니다."
        )
        return

    try:
        result = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False
        )

        device_list = (
            result.stdout.strip()
            or result.stderr.strip()
            or "재생 장치 정보 없음"
        )

        print("[AUDIO CHECK] ALSA 재생 장치 목록:")
        print(device_list)

    except Exception as device_error:
        print(
            f"[AUDIO CHECK] 장치 목록 확인 실패: "
            f"{device_error}"
        )


def check_audio_configuration(audio_path):
    """
    파일 존재 여부, aplay 설치 여부, 지정 ALSA 장치 열기 여부를 확인합니다.

    성공은 운영체제가 출력 장치를 열었다는 뜻입니다.
    스피커 전원, AUX 케이블, 볼륨과 같은 물리 연결은 별도로 확인해야 합니다.
    """

    global audio_last_error

    audio_last_error = None

    print("[AUDIO CHECK] 오디오 연결 상태를 확인합니다.")

    if audio_path is None:
        audio_last_error = "warning.wav 파일을 찾지 못했습니다."

        print(
            f"[AUDIO CHECK][FAIL] {audio_last_error}"
        )
        return False

    if not os.path.isfile(audio_path):
        audio_last_error = (
            f"오디오 파일이 존재하지 않습니다: {audio_path}"
        )

        print(
            f"[AUDIO CHECK][FAIL] {audio_last_error}"
        )
        return False

    if not os.access(audio_path, os.R_OK):
        audio_last_error = (
            f"오디오 파일 읽기 권한이 없습니다: {audio_path}"
        )

        print(
            f"[AUDIO CHECK][FAIL] {audio_last_error}"
        )
        return False

    if shutil.which("aplay") is None:
        audio_last_error = "aplay 명령을 찾지 못했습니다."

        print(
            f"[AUDIO CHECK][FAIL] {audio_last_error}"
        )
        return False

    command = build_audio_command(
        audio_path
    )

    if command is None:
        audio_last_error = "오디오 재생 명령을 만들지 못했습니다."

        print(
            f"[AUDIO CHECK][FAIL] {audio_last_error}"
        )
        return False

    print(
        f"[AUDIO CHECK] 파일 확인 완료: {audio_path}"
    )

    print(
        f"[AUDIO CHECK] 출력 장치 확인 중: "
        f"{AUDIO_OUTPUT_DEVICE}"
    )

    try:
        test_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            return_code = test_process.wait(
                timeout=AUDIO_STARTUP_TEST_SECONDS
            )

            if return_code != 0:
                error_output = ""

                if test_process.stderr is not None:
                    error_output = (
                        test_process.stderr.read().strip()
                    )

                audio_last_error = (
                    error_output
                    or f"재생 프로세스 종료 코드 {return_code}"
                )

                print(
                    f"[AUDIO CHECK][FAIL] "
                    f"{audio_last_error}"
                )

                print_alsa_playback_devices()
                return False

            print(
                "[AUDIO CHECK][OK] 오디오 파일 시험 재생이 "
                "정상 종료되었습니다."
            )
            return True

        except subprocess.TimeoutExpired:
            # 지정 시간 동안 정상 실행되었다면 장치 열기에 성공한 것입니다.
            test_process.terminate()

            try:
                test_process.wait(timeout=0.5)

            except subprocess.TimeoutExpired:
                test_process.kill()

            print(
                "[AUDIO CHECK][OK] ALSA 출력 장치를 정상적으로 "
                "열었습니다."
            )

            print(
                "[AUDIO CHECK] 방금 warning.wav 앞부분을 "
                f"{AUDIO_STARTUP_TEST_SECONDS:.1f}초 시험 재생했습니다."
            )
            return True

    except Exception as test_error:
        audio_last_error = str(test_error)

        print(
            f"[AUDIO CHECK][FAIL] 오디오 장치 확인 실패: "
            f"{audio_last_error}"
        )

        print_alsa_playback_devices()
        return False



# ============================================================
# 맥북 UDP RTP/JPEG 영상 송출 함수
# ============================================================

def resolve_macbook_ip():
    """
    MACBOOK_IP가 지정되어 있으면 그 값을 사용합니다.

    비어 있으면 현재 SSH 접속자의 IP를 맥북 주소로 사용합니다.
    맥북 터미널에서 SSH로 젯슨에 접속해 실행하는 경우
    별도 설정 없이 자동 인식됩니다.
    """

    configured_ip = MACBOOK_IP.strip()

    if configured_ip:
        return configured_ip

    for environment_name in (
        "SSH_CLIENT",
        "SSH_CONNECTION"
    ):
        connection_information = os.environ.get(
            environment_name,
            ""
        ).split()

        if connection_information:
            detected_ip = connection_information[0].strip()

            if detected_ip:
                print(
                    f"[INFO] {environment_name}에서 "
                    f"맥북 IP를 자동 인식했습니다: "
                    f"{detected_ip}"
                )

                return detected_ip

    raise RuntimeError(
        "맥북 IP를 자동 인식하지 못했습니다. "
        "파일 상단의 MACBOOK_IP에 맥북 IP를 입력하거나, "
        "'MACBOOK_IP=192.168.x.x python3 파일명.py' 형식으로 "
        "실행하십시오."
    )


class GstUdpJpegStreamer:
    """
    OpenCV의 GStreamer 지원 여부와 관계없이 gst-launch-1.0을
    별도 프로세스로 실행해 BGR 프레임을 RTP/JPEG로 송출합니다.

    맥북 수신 파이프라인:
    gst-launch-1.0 -v udpsrc port=5000 \
      caps="application/x-rtp,media=video,encoding-name=JPEG,payload=26" \
      ! rtpjpegdepay ! jpegdec ! videoconvert \
      ! autovideosink sync=false
    """

    def __init__(
        self,
        host,
        port,
        width,
        height,
        fps,
        jpeg_quality
    ):
        if shutil.which("gst-launch-1.0") is None:
            raise RuntimeError(
                "gst-launch-1.0을 찾지 못했습니다. "
                "젯슨에 GStreamer가 설치되어 있는지 확인하십시오."
            )

        self.host = host
        self.port = int(port)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(round(fps)))
        self.jpeg_quality = int(
            clamp(
                jpeg_quality,
                1,
                100
            )
        )

        frame_size = (
            self.width
            * self.height
            * 3
        )

        command = [
            "gst-launch-1.0",
            "-q",
            "fdsrc",
            "fd=0",
            "do-timestamp=true",
            f"blocksize={frame_size}",
            "!",
            "rawvideoparse",
            "format=bgr",
            f"width={self.width}",
            f"height={self.height}",
            f"framerate={self.fps}/1",
            "!",
            "queue",
            "max-size-buffers=1",
            "max-size-bytes=0",
            "max-size-time=0",
            "leaky=downstream",
            "!",
            "videoconvert",
            "!",
            "video/x-raw,format=I420",
            "!",
            "jpegenc",
            f"quality={self.jpeg_quality}",
            "!",
            "rtpjpegpay",
            "pt=26",
            "!",
            "udpsink",
            f"host={self.host}",
            f"port={self.port}",
            "sync=false",
            "async=false"
        ]

        print(
            f"[INFO] 맥북 영상 송출 시작: "
            f"{self.host}:{self.port}"
        )

        print(
            f"[INFO] 송출 형식: RTP/JPEG payload=26, "
            f"{self.width}x{self.height}, "
            f"{self.fps} FPS 설정"
        )

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL
        )

    def send(self, frame):
        if self.process is None:
            return

        if self.process.poll() is not None:
            raise RuntimeError(
                "GStreamer 영상 송출 프로세스가 종료되었습니다."
            )

        if (
            frame.shape[1] != self.width
            or frame.shape[0] != self.height
        ):
            frame = cv2.resize(
                frame,
                (
                    self.width,
                    self.height
                ),
                interpolation=cv2.INTER_LINEAR
            )

        if not frame.flags["C_CONTIGUOUS"]:
            frame = frame.copy()

        try:
            self.process.stdin.write(
                frame.tobytes()
            )

        except BrokenPipeError as stream_error:
            raise RuntimeError(
                "GStreamer 송출 파이프가 끊어졌습니다."
            ) from stream_error

    def stop(self):
        if self.process is None:
            return

        try:
            if self.process.stdin is not None:
                self.process.stdin.close()

        except Exception:
            pass

        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)

            except subprocess.TimeoutExpired:
                self.process.kill()

            except Exception:
                pass

        self.process = None

        print(
            "[INFO] 맥북 영상 송출을 종료했습니다."
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

# 팬 서보가 230도까지 움직일 수 있도록 전체 범위를 270도로 지정
kit.servo[TILT_SERVO_CHANNEL].actuation_range = (
    SERVO_ACTUATION_RANGE
)

kit.servo[PAN_SERVO_CHANNEL].actuation_range = (
    SERVO_ACTUATION_RANGE
)

kit.servo[TILT_SERVO_CHANNEL].set_pulse_width_range(
    MIN_PULSE,
    MAX_PULSE
)

kit.servo[PAN_SERVO_CHANNEL].set_pulse_width_range(
    MIN_PULSE,
    MAX_PULSE
)


# 미세 보간 출력기가 시작되기 전에는 직접 출력하고,
# 시작된 뒤에는 목표 각도만 전달합니다.
servo_output_smoother = None


def set_tilt_servo_angle_safe(angle):
    safe_angle = clamp_tilt_angle(
        angle
    )

    if servo_output_smoother is None:
        kit.servo[TILT_SERVO_CHANNEL].angle = (
            safe_angle
        )
    else:
        servo_output_smoother.set_tilt_target(
            safe_angle
        )

    return safe_angle


def set_pan_servo_angle_safe(angle):
    safe_angle = clamp_pan_angle(
        angle
    )

    if servo_output_smoother is None:
        kit.servo[PAN_SERVO_CHANNEL].angle = (
            safe_angle
        )
    else:
        servo_output_smoother.set_pan_target(
            safe_angle
        )

    return safe_angle


class ServoOutputSmoother:
    """
    YOLO 추론 루프와 독립적으로 팬/틸트 각도를 미세하게 보간합니다.

    메인 루프는 비교적 큰 간격으로 목표 각도를 갱신하고,
    이 클래스는 그 사이를 100 Hz 속도/가속도 제한으로 채웁니다.
    따라서 추론 FPS가 낮아도 서보가 한 번에 큰 각도로 뛰지 않습니다.
    """

    def __init__(self, initial_pan_angle, initial_tilt_angle):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.target_pan_angle = clamp_pan_angle(
            initial_pan_angle
        )
        self.target_tilt_angle = clamp_tilt_angle(
            initial_tilt_angle
        )

        self.output_pan_angle = self.target_pan_angle
        self.output_tilt_angle = self.target_tilt_angle

        self.output_pan_velocity = 0.0
        self.output_tilt_velocity = 0.0

        self.last_written_pan_angle = self.output_pan_angle
        self.last_written_tilt_angle = self.output_tilt_angle

        self.last_error = None

        self.thread = threading.Thread(
            target=self._run,
            name="servo-output-smoother",
            daemon=True
        )

    def start(self):
        self.thread.start()

    def set_pan_target(self, angle):
        with self.lock:
            self.target_pan_angle = clamp_pan_angle(
                angle
            )

    def set_tilt_target(self, angle):
        with self.lock:
            self.target_tilt_angle = clamp_tilt_angle(
                angle
            )

    def get_state(self):
        with self.lock:
            return {
                "pan_angle": self.output_pan_angle,
                "tilt_angle": self.output_tilt_angle,
                "pan_velocity": self.output_pan_velocity,
                "tilt_velocity": self.output_tilt_velocity,
                "target_pan_angle": self.target_pan_angle,
                "target_tilt_angle": self.target_tilt_angle,
                "error": self.last_error
            }

    def wait_until_settled(self, timeout=3.0):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            state = self.get_state()

            pan_error = abs(
                state["target_pan_angle"]
                - state["pan_angle"]
            )
            tilt_error = abs(
                state["target_tilt_angle"]
                - state["tilt_angle"]
            )

            if (
                pan_error <= SERVO_OUTPUT_SETTLE_ERROR
                and tilt_error <= SERVO_OUTPUT_SETTLE_ERROR
                and abs(state["pan_velocity"]) <= 0.5
                and abs(state["tilt_velocity"]) <= 0.5
            ):
                return True

            time.sleep(SERVO_OUTPUT_INTERVAL)

        return False

    def stop(self):
        self.stop_event.set()

        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    @staticmethod
    def _advance_axis(
        current_angle,
        current_velocity,
        target_angle,
        minimum_angle,
        maximum_angle,
        delta_time
    ):
        position_error = target_angle - current_angle

        target_velocity = clamp(
            position_error * SERVO_OUTPUT_POSITION_KP,
            -SERVO_OUTPUT_MAX_SPEED,
            SERVO_OUTPUT_MAX_SPEED
        )

        next_velocity = update_velocity_smoothly(
            current_velocity=current_velocity,
            target_velocity=target_velocity,
            max_acceleration=SERVO_OUTPUT_MAX_ACCELERATION,
            delta_time=delta_time
        )

        next_angle = (
            current_angle
            + next_velocity * delta_time
        )

        # 목표를 지나치면 목표 각도에 정확히 정지합니다.
        if (
            position_error > 0.0
            and next_angle >= target_angle
        ) or (
            position_error < 0.0
            and next_angle <= target_angle
        ):
            next_angle = target_angle
            next_velocity = 0.0

        next_angle = clamp(
            next_angle,
            minimum_angle,
            maximum_angle
        )

        if (
            next_angle <= minimum_angle
            and next_velocity < 0.0
        ) or (
            next_angle >= maximum_angle
            and next_velocity > 0.0
        ):
            next_velocity = 0.0

        if (
            abs(target_angle - next_angle)
            <= SERVO_OUTPUT_SETTLE_ERROR
            and abs(next_velocity) < 0.8
        ):
            next_angle = target_angle
            next_velocity = 0.0

        return next_angle, next_velocity

    def _run(self):
        previous_time = time.monotonic()
        next_wakeup_time = previous_time

        while not self.stop_event.is_set():
            now = time.monotonic()

            delta_time = clamp(
                now - previous_time,
                0.001,
                0.05
            )
            previous_time = now

            with self.lock:
                target_pan = self.target_pan_angle
                target_tilt = self.target_tilt_angle
                current_pan = self.output_pan_angle
                current_tilt = self.output_tilt_angle
                pan_velocity = self.output_pan_velocity
                tilt_velocity = self.output_tilt_velocity

            next_pan, next_pan_velocity = self._advance_axis(
                current_angle=current_pan,
                current_velocity=pan_velocity,
                target_angle=target_pan,
                minimum_angle=PAN_MIN_ANGLE,
                maximum_angle=PAN_MAX_ANGLE,
                delta_time=delta_time
            )

            next_tilt, next_tilt_velocity = self._advance_axis(
                current_angle=current_tilt,
                current_velocity=tilt_velocity,
                target_angle=target_tilt,
                minimum_angle=TILT_MIN_ANGLE,
                maximum_angle=TILT_MAX_ANGLE,
                delta_time=delta_time
            )

            try:
                if (
                    abs(
                        next_pan
                        - self.last_written_pan_angle
                    ) >= SERVO_OUTPUT_WRITE_EPSILON
                ):
                    kit.servo[PAN_SERVO_CHANNEL].angle = next_pan
                    self.last_written_pan_angle = next_pan

                if (
                    abs(
                        next_tilt
                        - self.last_written_tilt_angle
                    ) >= SERVO_OUTPUT_WRITE_EPSILON
                ):
                    kit.servo[TILT_SERVO_CHANNEL].angle = next_tilt
                    self.last_written_tilt_angle = next_tilt

                last_error = None

            except Exception as servo_output_error:
                last_error = str(servo_output_error)

            with self.lock:
                self.output_pan_angle = next_pan
                self.output_tilt_angle = next_tilt
                self.output_pan_velocity = next_pan_velocity
                self.output_tilt_velocity = next_tilt_velocity
                self.last_error = last_error

            next_wakeup_time += SERVO_OUTPUT_INTERVAL
            sleep_time = next_wakeup_time - time.monotonic()

            if sleep_time > 0.0:
                self.stop_event.wait(sleep_time)
            else:
                # 처리 지연이 누적되면 현재 시각부터 주기를 다시 맞춥니다.
                next_wakeup_time = time.monotonic()


# ============================================================
# YOLO 모델
# ============================================================

print("[INFO] YOLO 모델을 불러옵니다.")

model = YOLO(
    MODEL_PATH,
    task="detect"
)

print("[INFO] 탐지 대상: PERSON + CUP")
print("[INFO] 컵에서 멀어진 컵 소유자를 팬-틸트로 추적합니다.")
print(
    f"[INFO] 분리 판단 거리 비율: "
    f"{PERSON_SEPARATION_DISTANCE_RATIO:.2f}"
)

AUDIO_PATH = find_audio_file()

if AUDIO_PATH is not None:
    print(
        f"[INFO] 오디오 파일: {AUDIO_PATH}"
    )

    print(
        f"[INFO] 오디오 출력 장치: "
        f"{AUDIO_OUTPUT_DEVICE}"
    )
else:
    print(
        "[WARNING] 실행 폴더에서 오디오 파일을 찾지 못했습니다."
    )

if AUDIO_STARTUP_TEST:
    AUDIO_DEVICE_READY = check_audio_configuration(
        AUDIO_PATH
    )
else:
    AUDIO_DEVICE_READY = (
        AUDIO_PATH is not None
        and shutil.which("aplay") is not None
    )

    print(
        "[AUDIO CHECK] 시작 시험 재생이 비활성화되어 있습니다."
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
# 맥북 UDP 영상 송출 초기화
# ============================================================

stream_host = resolve_macbook_ip()

video_streamer = GstUdpJpegStreamer(
    host=stream_host,
    port=STREAM_PORT,
    width=actual_width,
    height=actual_height,
    fps=STREAM_FPS,
    jpeg_quality=STREAM_JPEG_QUALITY
)


# ============================================================
# 초기 상태
# ============================================================

current_tilt_angle = set_tilt_servo_angle_safe(
    TILT_INITIAL_ANGLE
)

current_pan_angle = set_pan_servo_angle_safe(
    PAN_INITIAL_ANGLE
)

time.sleep(1.0)

# 이후의 서보 출력은 YOLO 추론과 분리된 100 Hz 미세 보간 루프가 담당합니다.
servo_output_smoother = ServoOutputSmoother(
    initial_pan_angle=current_pan_angle,
    initial_tilt_angle=current_tilt_angle
)
servo_output_smoother.start()

print(
    f"[INFO] 서보 미세 보간 출력 시작: "
    f"{1.0 / SERVO_OUTPUT_INTERVAL:.0f} Hz, "
    f"최대 {SERVO_OUTPUT_MAX_SPEED:.1f} deg/s"
)

current_time = time.monotonic()

last_servo_update_time = current_time
last_target_detection_time = current_time

tilt_servo_velocity = 0.0
pan_servo_velocity = 0.0

filtered_target_x = None
filtered_target_y = None

previous_error_x = None
previous_error_y = None

filtered_derivative_x = 0.0
filtered_derivative_y = 0.0

returning_to_initial = False
angle_limit_active = False

previous_tilt_limit_state = None
previous_pan_limit_state = None

last_valid_face = None

last_valid_cup = None
last_cup_detection_time = 0.0
cup_target_held = False

# 상태 종류:
# WAITING_PAIR      : 컵과 사람이 가까워지기를 기다림
# MONITORING_HOLDER : 컵 소유자로 등록하고 거리 감시
# TRACKING_PERSON   : 사람이 컵에서 멀어진 뒤 얼굴 팬-틸트 추적
tracking_state = "WAITING_PAIR"

holder_confirm_count = 0
separation_confirm_count = 0
return_to_cup_confirm_count = 0

registered_person = None
registered_cup = None

last_registered_person_time = 0.0

current_distance_ratio = None
person_match_ratio = None

audio_process = None
last_audio_start_time = -AUDIO_REPEAT_INTERVAL

reset_after_return = False

# 실제 추적 대상이 마지막으로 검출된 시각을 기준으로
# 4초 소실 복귀를 강제로 수행하는 감시 상태입니다.
target_missing_since = None
lost_return_message_printed = False


# ============================================================
# 메인 루프
# ============================================================

try:
    while True:
        success, frame = camera.read()

        if not success:
            print(
                "[WARNING] 카메라 프레임을 읽지 못했습니다."
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
        # 컵 탐지 결과 유지
        # ====================================================

        cup_target_held = False

        if cups:
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
            cups = [last_valid_cup]
            cup_target_held = True

        else:
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
                        "[INFO] 컵을 들고 있던 사람을 등록했습니다."
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

                    reset_after_return = False
                    returning_to_initial = False

                    print(
                        "[INFO] 사람이 컵에서 멀어졌습니다."
                    )

                    print(
                        "[INFO] 등록된 사람의 얼굴 팬-틸트 추적을 시작합니다."
                    )

            else:
                # 거리 감시 상태에서도 등록된 사람이 사라진 뒤
                # 4초가 지나면 현재 위치에 멈추지 않고 초기 위치로 복귀합니다.
                if (
                    current_time
                    - last_registered_person_time
                    >= NO_ACTIVITY_RESET_TIME
                ):
                    print(
                        f"[INFO] 등록된 사람이 "
                        f"{NO_ACTIVITY_RESET_TIME:.1f}초 이상 "
                        "보이지 않아 초기 위치로 복귀합니다."
                    )

                    reset_after_return = True
                    returning_to_initial = True

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

                if (
                    return_to_cup_confirm_count
                    >= RETURN_TO_CUP_CONFIRM_FRAMES
                ):
                    tracking_state = "MONITORING_HOLDER"

                    return_to_cup_confirm_count = 0
                    separation_confirm_count = 0

                    last_valid_face = None

                    filtered_target_x = None
                    filtered_target_y = None

                    previous_error_x = None
                    previous_error_y = None

                    filtered_derivative_x = 0.0
                    filtered_derivative_y = 0.0

                    tilt_servo_velocity = 0.0
                    pan_servo_velocity = 0.0

                    returning_to_initial = False

                    print(
                        "[INFO] 추적 대상이 다시 컵 가까이 돌아왔습니다."
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
        # 추적 대상 4초 소실 감시
        # ====================================================

        real_target_detected = (
            tracking_state == "TRACKING_PERSON"
            and face_target is not None
            and not target_held
        )

        if real_target_detected:
            target_missing_since = None
            lost_return_message_printed = False

        elif tracking_state == "TRACKING_PERSON":
            if target_missing_since is None:
                # 마지막 실제 검출 시각부터 소실 시간을 계산합니다.
                target_missing_since = (
                    last_target_detection_time
                )

            target_missing_time = (
                current_time
                - target_missing_since
            )

            if (
                target_missing_time
                >= NO_ACTIVITY_RESET_TIME
            ):
                # 마지막 위치 유지값까지 제거하고 반드시 복귀 모드로 전환합니다.
                face_target = None
                last_valid_face = None
                target_held = False

                reset_after_return = True
                returning_to_initial = True

                if not lost_return_message_printed:
                    print(
                        f"[INFO] 추적 대상이 "
                        f"{NO_ACTIVITY_RESET_TIME:.1f}초 이상 "
                        "보이지 않아 초기 위치로 복귀합니다."
                    )

                    lost_return_message_printed = True

        elif returning_to_initial:
            # 복귀 중에는 기존 소실 시각을 유지합니다.
            pass

        else:
            target_missing_since = None
            lost_return_message_printed = False


        # ====================================================
        # 대상 탐지 중 오디오 반복 방송
        # ====================================================

        audio_should_play = (
            tracking_state == "TRACKING_PERSON"
            and face_target is not None
        )

        if audio_should_play:
            if (
                current_time - last_audio_start_time
                >= AUDIO_REPEAT_INTERVAL
            ):
                audio_process = stop_audio_playback(
                    audio_process
                )

                audio_process = start_audio_playback(
                    AUDIO_PATH
                )

                last_audio_start_time = current_time

        else:
            audio_process = stop_audio_playback(
                audio_process
            )

            last_audio_start_time = (
                current_time
                - AUDIO_REPEAT_INTERVAL
            )


        # ====================================================
        # 화면 중앙선과 데드존
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
            (screen_center_x, 0),
            (screen_center_x, frame_height),
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (
                0,
                screen_center_y - DEAD_ZONE_Y
            ),
            (
                frame_width,
                screen_center_y - DEAD_ZONE_Y
            ),
            (100, 100, 100),
            1
        )

        cv2.line(
            frame,
            (
                0,
                screen_center_y + DEAD_ZONE_Y
            ),
            (
                frame_width,
                screen_center_y + DEAD_ZONE_Y
            ),
            (100, 100, 100),
            1
        )

        cv2.line(
            frame,
            (
                screen_center_x - DEAD_ZONE_X,
                0
            ),
            (
                screen_center_x - DEAD_ZONE_X,
                frame_height
            ),
            (100, 100, 100),
            1
        )

        cv2.line(
            frame,
            (
                screen_center_x + DEAD_ZONE_X,
                0
            ),
            (
                screen_center_x + DEAD_ZONE_X,
                frame_height
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


        error_x = None
        error_y = None

        p_term_x = 0.0
        d_term_x = 0.0

        p_term_y = 0.0
        d_term_y = 0.0


        # ====================================================
        # 얼굴 팬-틸트 PD 추적
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

            if filtered_target_x is None:
                filtered_target_x = face_center_x

            elif not target_held:
                filtered_target_x = (
                    TARGET_FILTER_ALPHA
                    * face_center_x
                    + (
                        1.0
                        - TARGET_FILTER_ALPHA
                    )
                    * filtered_target_x
                )

            if filtered_target_y is None:
                filtered_target_y = raw_face_center_y

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

            error_x = (
                filtered_target_x
                - screen_center_x
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
                    int(filtered_target_x),
                    int(filtered_target_y)
                ),
                (255, 0, 255),
                2
            )


            # -----------------------------------------------
            # X/Y축 PD 제어
            # -----------------------------------------------

            if (
                current_time
                - last_servo_update_time
                >= SERVO_UPDATE_INTERVAL
            ):
                (
                    target_pan_velocity,
                    filtered_derivative_x,
                    p_term_x,
                    d_term_x
                ) = calculate_axis_target_velocity(
                    error=error_x,
                    previous_error=previous_error_x,
                    filtered_derivative=filtered_derivative_x,
                    dead_zone=DEAD_ZONE_X,
                    slow_zone=SLOW_ZONE_X,
                    direction=PAN_SERVO_DIRECTION,
                    delta_time=delta_time,
                    target_held=target_held
                )

                (
                    target_tilt_velocity,
                    filtered_derivative_y,
                    p_term_y,
                    d_term_y
                ) = calculate_axis_target_velocity(
                    error=error_y,
                    previous_error=previous_error_y,
                    filtered_derivative=filtered_derivative_y,
                    dead_zone=DEAD_ZONE_Y,
                    slow_zone=SLOW_ZONE_Y,
                    direction=TILT_SERVO_DIRECTION,
                    delta_time=delta_time,
                    target_held=target_held
                )


                # 중앙 데드존에서는 현재 속도를 감쇠
                if abs(error_x) <= DEAD_ZONE_X:
                    pan_servo_velocity *= (
                        CENTER_VELOCITY_DAMPING
                    )

                    if abs(pan_servo_velocity) < 0.08:
                        pan_servo_velocity = 0.0

                else:
                    pan_servo_velocity = (
                        update_velocity_smoothly(
                            current_velocity=(
                                pan_servo_velocity
                            ),
                            target_velocity=(
                                target_pan_velocity
                            ),
                            max_acceleration=(
                                MAX_SERVO_ACCELERATION
                            ),
                            delta_time=delta_time
                        )
                    )

                if abs(error_y) <= DEAD_ZONE_Y:
                    tilt_servo_velocity *= (
                        CENTER_VELOCITY_DAMPING
                    )

                    if abs(tilt_servo_velocity) < 0.08:
                        tilt_servo_velocity = 0.0

                else:
                    tilt_servo_velocity = (
                        update_velocity_smoothly(
                            current_velocity=(
                                tilt_servo_velocity
                            ),
                            target_velocity=(
                                target_tilt_velocity
                            ),
                            max_acceleration=(
                                MAX_SERVO_ACCELERATION
                            ),
                            delta_time=delta_time
                        )
                    )


                requested_pan_angle = (
                    current_pan_angle
                    + pan_servo_velocity
                    * delta_time
                )

                requested_tilt_angle = (
                    current_tilt_angle
                    + tilt_servo_velocity
                    * delta_time
                )


                # -------------------------------------------
                # 팬 서보 안전각 제한
                # -------------------------------------------

                if requested_pan_angle < PAN_MIN_ANGLE:
                    current_pan_angle = (
                        set_pan_servo_angle_safe(
                            PAN_MIN_ANGLE
                        )
                    )

                    angle_limit_active = True

                    if pan_servo_velocity < 0:
                        pan_servo_velocity = 0.0

                    if previous_pan_limit_state != "MIN":
                        print(
                            f"[SAFETY] 팬 최소각도 "
                            f"{PAN_MIN_ANGLE:.1f}도 제한"
                        )

                        previous_pan_limit_state = "MIN"

                elif requested_pan_angle > PAN_MAX_ANGLE:
                    current_pan_angle = (
                        set_pan_servo_angle_safe(
                            PAN_MAX_ANGLE
                        )
                    )

                    angle_limit_active = True

                    if pan_servo_velocity > 0:
                        pan_servo_velocity = 0.0

                    if previous_pan_limit_state != "MAX":
                        print(
                            f"[SAFETY] 팬 최대각도 "
                            f"{PAN_MAX_ANGLE:.1f}도 제한"
                        )

                        previous_pan_limit_state = "MAX"

                else:
                    current_pan_angle = (
                        set_pan_servo_angle_safe(
                            requested_pan_angle
                        )
                    )

                    previous_pan_limit_state = None


                # -------------------------------------------
                # 틸트 서보 안전각 제한
                # -------------------------------------------

                if requested_tilt_angle < TILT_MIN_ANGLE:
                    current_tilt_angle = (
                        set_tilt_servo_angle_safe(
                            TILT_MIN_ANGLE
                        )
                    )

                    angle_limit_active = True

                    if tilt_servo_velocity < 0:
                        tilt_servo_velocity = 0.0

                    if previous_tilt_limit_state != "MIN":
                        print(
                            f"[SAFETY] 틸트 최소각도 "
                            f"{TILT_MIN_ANGLE:.1f}도 제한"
                        )

                        previous_tilt_limit_state = "MIN"

                elif requested_tilt_angle > TILT_MAX_ANGLE:
                    current_tilt_angle = (
                        set_tilt_servo_angle_safe(
                            TILT_MAX_ANGLE
                        )
                    )

                    angle_limit_active = True

                    if tilt_servo_velocity > 0:
                        tilt_servo_velocity = 0.0

                    if previous_tilt_limit_state != "MAX":
                        print(
                            f"[SAFETY] 틸트 최대각도 "
                            f"{TILT_MAX_ANGLE:.1f}도 제한"
                        )

                        previous_tilt_limit_state = "MAX"

                else:
                    current_tilt_angle = (
                        set_tilt_servo_angle_safe(
                            requested_tilt_angle
                        )
                    )

                    previous_tilt_limit_state = None


                if not target_held:
                    previous_error_x = error_x
                    previous_error_y = error_y

                last_servo_update_time = (
                    current_time
                )


        # ====================================================
        # 얼굴 추적 대상 없음
        # ====================================================

        else:
            filtered_target_x = None
            filtered_target_y = None

            previous_error_x = None
            previous_error_y = None

            filtered_derivative_x = 0.0
            filtered_derivative_y = 0.0

            angle_limit_active = False

            previous_pan_limit_state = None
            previous_tilt_limit_state = None

            # 초기 위치 복귀 중이 아닐 때만 현재 속도를 0으로 감속합니다.
            if (
                not returning_to_initial
                and current_time
                - last_servo_update_time
                >= SERVO_UPDATE_INTERVAL
            ):
                pan_servo_velocity = (
                    update_velocity_smoothly(
                        current_velocity=(
                            pan_servo_velocity
                        ),
                        target_velocity=0.0,
                        max_acceleration=(
                            MAX_SERVO_ACCELERATION
                        ),
                        delta_time=delta_time
                    )
                )

                tilt_servo_velocity = (
                    update_velocity_smoothly(
                        current_velocity=(
                            tilt_servo_velocity
                        ),
                        target_velocity=0.0,
                        max_acceleration=(
                            MAX_SERVO_ACCELERATION
                        ),
                        delta_time=delta_time
                    )
                )

                if abs(pan_servo_velocity) < 0.05:
                    pan_servo_velocity = 0.0

                if abs(tilt_servo_velocity) < 0.05:
                    tilt_servo_velocity = 0.0

                last_servo_update_time = (
                    current_time
                )


            # 위의 '추적 대상 소실 감시'가 실제 검출 시각을 기준으로
            # 4초 후 강제 복귀를 처리합니다.


        # ====================================================
        # 팬과 틸트 초기 위치 복귀
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
                pan_return_error = (
                    PAN_INITIAL_ANGLE
                    - current_pan_angle
                )

                tilt_return_error = (
                    TILT_INITIAL_ANGLE
                    - current_tilt_angle
                )

                target_pan_return_velocity = clamp(
                    pan_return_error
                    * RETURN_KP,
                    -RETURN_MAX_SPEED,
                    RETURN_MAX_SPEED
                )

                target_tilt_return_velocity = clamp(
                    tilt_return_error
                    * RETURN_KP,
                    -RETURN_MAX_SPEED,
                    RETURN_MAX_SPEED
                )

                pan_servo_velocity = (
                    update_velocity_smoothly(
                        current_velocity=(
                            pan_servo_velocity
                        ),
                        target_velocity=(
                            target_pan_return_velocity
                        ),
                        max_acceleration=(
                            RETURN_ACCELERATION
                        ),
                        delta_time=delta_time
                    )
                )

                tilt_servo_velocity = (
                    update_velocity_smoothly(
                        current_velocity=(
                            tilt_servo_velocity
                        ),
                        target_velocity=(
                            target_tilt_return_velocity
                        ),
                        max_acceleration=(
                            RETURN_ACCELERATION
                        ),
                        delta_time=delta_time
                    )
                )

                next_pan_angle = (
                    current_pan_angle
                    + pan_servo_velocity
                    * delta_time
                )

                next_tilt_angle = (
                    current_tilt_angle
                    + tilt_servo_velocity
                    * delta_time
                )


                if (
                    current_pan_angle > PAN_INITIAL_ANGLE
                    and next_pan_angle < PAN_INITIAL_ANGLE
                ):
                    next_pan_angle = PAN_INITIAL_ANGLE

                elif (
                    current_pan_angle < PAN_INITIAL_ANGLE
                    and next_pan_angle > PAN_INITIAL_ANGLE
                ):
                    next_pan_angle = PAN_INITIAL_ANGLE


                if (
                    current_tilt_angle > TILT_INITIAL_ANGLE
                    and next_tilt_angle < TILT_INITIAL_ANGLE
                ):
                    next_tilt_angle = TILT_INITIAL_ANGLE

                elif (
                    current_tilt_angle < TILT_INITIAL_ANGLE
                    and next_tilt_angle > TILT_INITIAL_ANGLE
                ):
                    next_tilt_angle = TILT_INITIAL_ANGLE


                current_pan_angle = (
                    set_pan_servo_angle_safe(
                        next_pan_angle
                    )
                )

                current_tilt_angle = (
                    set_tilt_servo_angle_safe(
                        next_tilt_angle
                    )
                )

                last_servo_update_time = (
                    current_time
                )


            pan_return_complete = (
                abs(
                    current_pan_angle
                    - PAN_INITIAL_ANGLE
                ) <= 0.08
                and abs(
                    pan_servo_velocity
                ) <= 0.35
            )

            tilt_return_complete = (
                abs(
                    current_tilt_angle
                    - TILT_INITIAL_ANGLE
                ) <= 0.08
                and abs(
                    tilt_servo_velocity
                ) <= 0.35
            )

            if (
                pan_return_complete
                and tilt_return_complete
            ):
                current_pan_angle = (
                    set_pan_servo_angle_safe(
                        PAN_INITIAL_ANGLE
                    )
                )

                current_tilt_angle = (
                    set_tilt_servo_angle_safe(
                        TILT_INITIAL_ANGLE
                    )
                )

                pan_servo_velocity = 0.0
                tilt_servo_velocity = 0.0

                returning_to_initial = False

                if (
                    reset_after_return
                    or AUTO_RESET_TRACKING
                ):
                    tracking_state = "WAITING_PAIR"

                    holder_confirm_count = 0
                    separation_confirm_count = 0
                    return_to_cup_confirm_count = 0

                    registered_person = None
                    registered_cup = None

                    last_valid_face = None
                    last_valid_cup = None

                    current_distance_ratio = None
                    person_match_ratio = None

                    reset_after_return = False

                    print(
                        f"[INFO] {NO_ACTIVITY_RESET_TIME:.1f}초 동안 "
                        "대상 탐지가 없어 팬과 틸트를 "
                        "초기 위치로 복귀했습니다."
                    )


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
                "TARGET LOST - PAN/TILT RETURNING"
            )

        elif face_target is not None and target_held:
            status_text = (
                "PERSON TEMPORARILY HELD"
            )

        elif face_target is not None:
            if angle_limit_active:
                status_text = (
                    "PAN/TILT TRACKING - ANGLE LIMIT"
                )

            elif (
                error_x is not None
                and error_y is not None
                and abs(error_x) <= DEAD_ZONE_X
                and abs(error_y) <= DEAD_ZONE_Y
            ):
                status_text = (
                    "PERSON FACE CENTERED"
                )

            else:
                status_text = (
                    "PAN/TILT TRACKING CUP OWNER"
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

        if error_x is not None and error_y is not None:
            cv2.putText(
                frame,
                (
                    f"Face error X/Y: "
                    f"{error_x:.1f} / "
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
                    f"Pan/Tilt speed: "
                    f"{pan_servo_velocity:.2f} / "
                    f"{tilt_servo_velocity:.2f} deg/s"
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
                    f"Pan P/D: "
                    f"{p_term_x:.2f} / "
                    f"{d_term_x:.2f}"
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
                    f"Tilt P/D: "
                    f"{p_term_y:.2f} / "
                    f"{d_term_y:.2f}"
                ),
                (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (255, 200, 0),
                2
            )

        audio_process_running = (
            audio_process is not None
            and audio_process.poll() is None
        )

        if audio_last_error is not None:
            audio_status = "ERROR"

        elif not AUDIO_DEVICE_READY:
            audio_status = "DEVICE FAIL"

        elif audio_process_running:
            audio_status = "PLAYING"

        elif audio_should_play:
            audio_status = "READY/REPEAT"

        else:
            audio_status = "OFF"

        cv2.putText(
            frame,
            f"AUDIO: {audio_status}",
            (
                frame_width - 170,
                30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (
                (0, 0, 255)
                if audio_process_running
                else (
                    (0, 165, 255)
                    if audio_last_error is not None
                    or not AUDIO_DEVICE_READY
                    else (180, 180, 180)
                )
            ),
            2
        )

        servo_output_state = (
            servo_output_smoother.get_state()
            if servo_output_smoother is not None
            else None
        )

        if servo_output_state is not None:
            cv2.putText(
                frame,
                (
                    f"Output Pan/Tilt: "
                    f"{servo_output_state['pan_angle']:.2f} / "
                    f"{servo_output_state['tilt_angle']:.2f} deg"
                ),
                (
                    frame_width - 390,
                    frame_height - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 255, 200),
                2
            )

        cv2.putText(
            frame,
            (
                f"Pan CH{PAN_SERVO_CHANNEL}: "
                f"{current_pan_angle:.2f} deg"
            ),
            (
                10,
                frame_height - 90
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            (
                f"Tilt CH{TILT_SERVO_CHANNEL}: "
                f"{current_tilt_angle:.2f} deg"
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
                f"Pan range: "
                f"{PAN_MIN_ANGLE:.0f}-"
                f"{PAN_MAX_ANGLE:.0f} / "
                f"Tilt range: "
                f"{TILT_MIN_ANGLE:.0f}-"
                f"{TILT_MAX_ANGLE:.0f}"
            ),
            (
                10,
                frame_height - 40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
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


        # 처리 결과가 그려진 최종 프레임을 맥북으로 송출합니다.
        video_streamer.send(
            frame
        )

        # 필요할 때만 젯슨 로컬 화면도 함께 표시합니다.
        if ENABLE_LOCAL_PREVIEW:
            cv2.imshow(
                "Cup Owner Pan Tilt Tracking",
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

        else:
            # 원격 송출 전용 모드에서는 Ctrl+C로 종료합니다.
            key = 255


        # 로컬 미리보기 사용 시 r 키로 상태 초기화
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

            reset_after_return = False
            returning_to_initial = True

            target_missing_since = None
            lost_return_message_printed = False

            audio_process = stop_audio_playback(
                audio_process
            )

            last_audio_start_time = (
                current_time
                - AUDIO_REPEAT_INTERVAL
            )

            print(
                "[INFO] 추적 상태를 수동 초기화하고 "
                "팬/틸트를 초기 위치로 복귀합니다."
            )


        if (
            key == ord("q")
            or key == 27
        ):
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
    try:
        audio_process = stop_audio_playback(
            audio_process
        )

        print(
            "[INFO] 팬과 틸트 서보를 초기 위치로 복귀합니다."
        )

        pan_return_velocity = 0.0
        tilt_return_velocity = 0.0

        previous_time = time.monotonic()

        while (
            abs(
                current_pan_angle
                - PAN_INITIAL_ANGLE
            ) > 0.05
            or abs(
                current_tilt_angle
                - TILT_INITIAL_ANGLE
            ) > 0.05
        ):
            now = time.monotonic()

            delta_time = clamp(
                now - previous_time,
                0.001,
                0.1
            )

            previous_time = now

            pan_return_error = (
                PAN_INITIAL_ANGLE
                - current_pan_angle
            )

            tilt_return_error = (
                TILT_INITIAL_ANGLE
                - current_tilt_angle
            )

            target_pan_velocity = clamp(
                pan_return_error
                * RETURN_KP,
                -RETURN_MAX_SPEED,
                RETURN_MAX_SPEED
            )

            target_tilt_velocity = clamp(
                tilt_return_error
                * RETURN_KP,
                -RETURN_MAX_SPEED,
                RETURN_MAX_SPEED
            )

            pan_return_velocity = (
                update_velocity_smoothly(
                    current_velocity=(
                        pan_return_velocity
                    ),
                    target_velocity=(
                        target_pan_velocity
                    ),
                    max_acceleration=(
                        RETURN_ACCELERATION
                    ),
                    delta_time=delta_time
                )
            )

            tilt_return_velocity = (
                update_velocity_smoothly(
                    current_velocity=(
                        tilt_return_velocity
                    ),
                    target_velocity=(
                        target_tilt_velocity
                    ),
                    max_acceleration=(
                        RETURN_ACCELERATION
                    ),
                    delta_time=delta_time
                )
            )

            next_pan_angle = (
                current_pan_angle
                + pan_return_velocity
                * delta_time
            )

            next_tilt_angle = (
                current_tilt_angle
                + tilt_return_velocity
                * delta_time
            )


            if (
                current_pan_angle > PAN_INITIAL_ANGLE
                and next_pan_angle < PAN_INITIAL_ANGLE
            ):
                next_pan_angle = PAN_INITIAL_ANGLE

            elif (
                current_pan_angle < PAN_INITIAL_ANGLE
                and next_pan_angle > PAN_INITIAL_ANGLE
            ):
                next_pan_angle = PAN_INITIAL_ANGLE


            if (
                current_tilt_angle > TILT_INITIAL_ANGLE
                and next_tilt_angle < TILT_INITIAL_ANGLE
            ):
                next_tilt_angle = TILT_INITIAL_ANGLE

            elif (
                current_tilt_angle < TILT_INITIAL_ANGLE
                and next_tilt_angle > TILT_INITIAL_ANGLE
            ):
                next_tilt_angle = TILT_INITIAL_ANGLE


            current_pan_angle = (
                set_pan_servo_angle_safe(
                    next_pan_angle
                )
            )

            current_tilt_angle = (
                set_tilt_servo_angle_safe(
                    next_tilt_angle
                )
            )

            time.sleep(
                RETURN_INTERVAL
            )

        current_pan_angle = (
            set_pan_servo_angle_safe(
                PAN_INITIAL_ANGLE
            )
        )

        current_tilt_angle = (
            set_tilt_servo_angle_safe(
                TILT_INITIAL_ANGLE
            )
        )

        if servo_output_smoother is not None:
            settled = servo_output_smoother.wait_until_settled(
                timeout=3.0
            )

            if not settled:
                print(
                    "[WARNING] 서보 출력이 초기 위치에 완전히 "
                    "도달하기 전에 대기 시간이 끝났습니다."
                )

        print(
            "[INFO] 팬과 틸트 서보 초기 위치 복귀 완료"
        )

    except Exception as servo_return_error:
        print(
            f"[WARNING] 종료 중 서보 복귀 실패: "
            f"{servo_return_error}"
        )

    finally:
        try:
            if servo_output_smoother is not None:
                servo_output_smoother.stop()

        except Exception as smoother_stop_error:
            print(
                f"[WARNING] 서보 미세 보간 종료 중 오류: "
                f"{smoother_stop_error}"
            )

        try:
            video_streamer.stop()

        except Exception as stream_stop_error:
            print(
                f"[WARNING] 영상 송출 종료 중 오류: "
                f"{stream_stop_error}"
            )

        camera.release()
        cv2.destroyAllWindows()

        print(
            "[INFO] 카메라를 종료했습니다."
        )

        print(
            "[INFO] 프로그램을 종료했습니다."
        )
