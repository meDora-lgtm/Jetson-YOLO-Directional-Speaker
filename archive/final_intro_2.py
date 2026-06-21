import time
import math
import os
import glob
import shutil
import subprocess
import cv2

from ultralytics import YOLO
from adafruit_extended_bus import ExtendedI2C
from adafruit_servokit import ServoKit
from adafruit_bus_device import i2c_device


# ============================================================
# 기본 설정
# ============================================================

MODEL_PATH = "yolo26n.engine"
CAMERA_DEVICE = "/dev/video0"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

IMAGE_SIZE = 1024
CONFIDENCE = 0.25

# COCO person 클래스만 탐지합니다.
PERSON_CLASS_ID = 0


# ============================================================
# 맥북 UDP 영상 송출 설정
# ============================================================

# 비워 두면 SSH_CLIENT/SSH_CONNECTION에서 접속한 맥북 IP를
# 자동으로 가져옵니다.
MACBOOK_IP = ""

STREAM_PORT = 5000
STREAM_FPS = 30
STREAM_JPEG_QUALITY = 80

# False: 젯슨에서는 창을 띄우지 않고 맥북으로만 송출
# True : 젯슨에서도 cv2.imshow 창을 함께 표시
ENABLE_LOCAL_PREVIEW = False


# ============================================================
# 오디오 설정
# ============================================================

# 재생할 오디오 파일입니다. 각 사람에게 순서대로 두 번씩 재생합니다.
AUDIO_FILE = "/home/sos/Capstone_projects/hello_sos.mp3"

# MP3는 ffmpeg로 WAV 변환 후 aplay로 재생합니다.
AUDIO_DECODED_WAV_PATH = "/tmp/capstone_person_intro_playback.wav"
AUDIO_OUTPUT_DEVICE = "plughw:CARD=Audio,DEV=0"
AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.ogg", "*.m4a", "*.aac")

# 사람 한 명당 오디오 재생 횟수와 재생 사이 간격입니다.
AUDIO_REPEAT_COUNT = 2
AUDIO_REPEAT_GAP = 0.35

# 오디오 실행 자체가 실패한 경우에도 다음 재생/복귀로 넘어가기 위한 시간입니다.
AUDIO_FAILURE_RETURN_DELAY = 0.8


# ============================================================
# 사람 추적 설정
# ============================================================

# 사람 바운딩박스 상단에서 아래쪽으로 16% 지점을 얼굴 중심으로 사용합니다.
FACE_CENTER_RATIO_Y = 0.16
FACE_BOX_WIDTH_RATIO = 0.42
FACE_BOX_HEIGHT_RATIO = 0.24

# 사람 검출이 잠깐 끊겨도 마지막 위치를 유지하는 시간입니다.
PERSON_HOLD_TIME = 0.50

# 이전 프레임의 사람과 현재 프레임의 사람이 같은 사람인지 판단하는 기준입니다.
PERSON_MATCH_DISTANCE_RATIO = 0.85

# 한 그룹의 모든 사람에게 방송한 뒤, 같은 그룹을 다시 방송하지 않기 위해
# 화면에서 사람이 이 시간 이상 모두 사라져야 새 그룹을 등록합니다.
REARM_CLEAR_TIME = 1.0

# 처음 사람이 보인 직후 여러 프레임을 모아 가장 많은 인원이 검출된
# 화면을 그룹 기준으로 사용합니다.
GROUP_COLLECTION_TIME = 0.7

# 다음 순번 사람이 잠시 검출되지 않을 때 기다리는 최대 시간입니다.
# 이 시간이 지나도 보이지 않으면 해당 순번을 건너뛰고 다음 사람으로 진행합니다.
NEXT_PERSON_WAIT_TIME = 2.5

# 초기 위치에서 저장한 사람 박스와 현재 박스를 같은 사람으로 연결하는 기준입니다.
QUEUE_MATCH_SCORE_LIMIT = 0.38


# ============================================================
# PCA9685 및 팬-틸트 서보 설정
# ============================================================

PCA9685_ADDRESS = 0x41

# 0번 채널: 틸트(위/아래)
# 1번 채널: 팬(좌/우)
TILT_SERVO_CHANNEL = 0
PAN_SERVO_CHANNEL = 1

TILT_MIN_ANGLE = 125.0
TILT_MAX_ANGLE = 180.0
TILT_INITIAL_ANGLE = 125.5

PAN_MIN_ANGLE = 70.0
PAN_MAX_ANGLE = 230.0
PAN_INITIAL_ANGLE = 150.0

SERVO_ACTUATION_RANGE = 270
MIN_PULSE = 500
MAX_PULSE = 2500


# ============================================================
# 화면 중심 제어
# ============================================================

DEAD_ZONE_X = 32
DEAD_ZONE_Y = 26

CENTER_RELEASE_ZONE_X = 46
CENTER_RELEASE_ZONE_Y = 38

SLOW_ZONE_X = 130
SLOW_ZONE_Y = 110

CENTER_VELOCITY_DAMPING = 0.35


# ============================================================
# PD 제어 설정 - 기존 파일 값 유지
# ============================================================

PD_KP = 0.34
PD_KD = 0.003

DERIVATIVE_FILTER_ALPHA = 0.12
MAX_D_TERM = 3.0
MAX_RAW_DERIVATIVE = 1200.0

MAX_SERVO_SPEED = 95.0
TARGET_FILTER_ALPHA = 0.45
SERVO_UPDATE_INTERVAL = 0.015
HELD_TARGET_SPEED_RATIO = 0.25

SERVO_DIRECT_WRITE_EPSILON = 0.08


# ============================================================
# 초기 위치 복귀 설정 - 기존 파일 값 유지
# ============================================================

RETURN_MAX_SPEED = 28.0
RETURN_ACCELERATION = 120.0
RETURN_INTERVAL = 0.010
RETURN_KP = 3.5


# ============================================================
# 서보 방향 설정
# ============================================================

TILT_SERVO_DIRECTION = -1
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
        probe=False,
    )


i2c_device.I2CDevice.__init__ = no_probe_init


# ============================================================
# 공통 함수
# ============================================================


def clamp(value, minimum, maximum):
    return max(minimum, min(float(value), maximum))


def clamp_tilt_angle(angle):
    return clamp(angle, TILT_MIN_ANGLE, TILT_MAX_ANGLE)


def clamp_pan_angle(angle):
    return clamp(angle, PAN_MIN_ANGLE, PAN_MAX_ANGLE)


def update_velocity_smoothly(
    current_velocity,
    target_velocity,
    max_acceleration,
    delta_time,
):
    velocity_difference = target_velocity - current_velocity
    max_velocity_change = max_acceleration * delta_time

    velocity_change = clamp(
        velocity_difference,
        -max_velocity_change,
        max_velocity_change,
    )

    return current_velocity + velocity_change


def box_center(detection):
    x1, y1, x2, y2, _confidence = detection
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_width(detection):
    return max(1.0, detection[2] - detection[0])


def box_height(detection):
    return max(1.0, detection[3] - detection[1])


def detection_area(detection):
    return box_width(detection) * box_height(detection)


def center_distance(first_detection, second_detection):
    first_x, first_y = box_center(first_detection)
    second_x, second_y = box_center(second_detection)
    return math.hypot(first_x - second_x, first_y - second_y)


def normalized_person_match_distance(previous_person, current_person):
    reference_height = box_height(previous_person)
    return center_distance(previous_person, current_person) / reference_height


def get_person_detections(boxes):
    persons = []

    if boxes is None:
        return persons

    for box in boxes:
        class_id = int(box.cls[0].item())

        if class_id != PERSON_CLASS_ID:
            continue

        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

        persons.append(
            (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                confidence,
            )
        )

    return persons


def sort_persons_left_to_right(persons):
    """초기 화면에서 사람들을 왼쪽부터 오른쪽 순서로 정렬합니다."""
    return sorted(persons, key=lambda person: box_center(person)[0])


def queued_person_match_score(
    reference_person,
    current_person,
    frame_width,
    frame_height,
):
    """
    초기 위치에서 저장한 사람과 현재 검출된 사람이 얼마나 가까운지 계산합니다.

    카메라가 매 사람 방송 후 초기 위치로 돌아오기 때문에 화면 중심 위치와
    박스 크기를 함께 비교하면 같은 사람을 다시 찾는 데 충분히 안정적입니다.
    """

    reference_x, reference_y = box_center(reference_person)
    current_x, current_y = box_center(current_person)

    normalized_center_distance = math.hypot(
        (current_x - reference_x) / max(1.0, frame_width),
        (current_y - reference_y) / max(1.0, frame_height),
    )

    reference_area = max(1.0, detection_area(reference_person))
    current_area = max(1.0, detection_area(current_person))
    size_difference = abs(math.log(current_area / reference_area))

    return normalized_center_distance + 0.10 * min(size_difference, 3.0)


def match_person_queue(
    reference_people,
    current_people,
    frame_width,
    frame_height,
):
    """
    초기 그룹의 각 사람과 현재 검출 사람을 1:1로 연결합니다.

    모든 조합의 점수를 작은 순서대로 배정하므로, 이미 방송한 사람이
    다음 순번 사람으로 중복 선택되는 현상을 줄일 수 있습니다.
    """

    if not reference_people or not current_people:
        return {}, {}

    candidates = []

    for reference_index, reference_person in enumerate(reference_people):
        for current_index, current_person in enumerate(current_people):
            score = queued_person_match_score(
                reference_person,
                current_person,
                frame_width,
                frame_height,
            )
            candidates.append((score, reference_index, current_index))

    candidates.sort(key=lambda item: item[0])

    assigned_references = set()
    assigned_current = set()
    matched_people = {}
    matched_scores = {}

    for score, reference_index, current_index in candidates:
        if score > QUEUE_MATCH_SCORE_LIMIT:
            break

        if reference_index in assigned_references:
            continue

        if current_index in assigned_current:
            continue

        assigned_references.add(reference_index)
        assigned_current.add(current_index)
        matched_people[reference_index] = current_people[current_index]
        matched_scores[reference_index] = score

    return matched_people, matched_scores


def find_matching_person(previous_person, current_persons):
    if previous_person is None or not current_persons:
        return None, None

    best_person = None
    best_distance_ratio = None

    for person in current_persons:
        distance_ratio = normalized_person_match_distance(
            previous_person,
            person,
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
        return best_person, best_distance_ratio

    return None, best_distance_ratio


def get_face_target(person):
    x1, y1, x2, y2, confidence = person

    person_width = max(1, x2 - x1)
    person_height = max(1, y2 - y1)

    face_center_x = (x1 + x2) / 2.0
    face_center_y = y1 + person_height * FACE_CENTER_RATIO_Y

    face_box_width = person_width * FACE_BOX_WIDTH_RATIO
    face_box_height = person_height * FACE_BOX_HEIGHT_RATIO

    return {
        "center_x": face_center_x,
        "center_y": face_center_y,
        "box": (
            int(face_center_x - face_box_width / 2.0),
            int(face_center_y - face_box_height / 2.0),
            int(face_center_x + face_box_width / 2.0),
            int(face_center_y + face_box_height / 2.0),
        ),
        "confidence": confidence,
    }


def draw_person_detection(frame, detection, label, color, thickness=2):
    x1, y1, x2, y2, confidence = detection

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame,
        f"{label}: {confidence:.2f}",
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        color,
        2,
    )


def calculate_axis_target_velocity(
    error,
    previous_error,
    filtered_derivative,
    dead_zone,
    slow_zone,
    direction,
    delta_time,
    target_held,
):
    target_velocity = 0.0
    p_term = 0.0
    d_term = 0.0

    if abs(error) <= dead_zone:
        filtered_derivative *= 0.45
        return target_velocity, filtered_derivative, p_term, d_term

    if error > 0:
        effective_error = error - dead_zone
    else:
        effective_error = error + dead_zone

    if target_held or previous_error is None:
        raw_derivative = 0.0
    else:
        raw_derivative = (error - previous_error) / delta_time

    raw_derivative = clamp(
        raw_derivative,
        -MAX_RAW_DERIVATIVE,
        MAX_RAW_DERIVATIVE,
    )

    if not target_held:
        filtered_derivative = (
            DERIVATIVE_FILTER_ALPHA * raw_derivative
            + (1.0 - DERIVATIVE_FILTER_ALPHA) * filtered_derivative
        )

    p_term = PD_KP * effective_error
    d_term = clamp(
        PD_KD * filtered_derivative,
        -MAX_D_TERM,
        MAX_D_TERM,
    )

    if target_held:
        d_term = 0.0

    target_velocity = (p_term + d_term) * direction
    target_velocity = clamp(
        target_velocity,
        -MAX_SERVO_SPEED,
        MAX_SERVO_SPEED,
    )

    if abs(error) < slow_zone:
        slow_ratio = (abs(error) - dead_zone) / (slow_zone - dead_zone)
        slow_ratio = clamp(slow_ratio, 0.35, 1.0)
        target_velocity *= slow_ratio

    if target_held:
        target_velocity *= HELD_TARGET_SPEED_RATIO

    return target_velocity, filtered_derivative, p_term, d_term


# ============================================================
# 오디오 함수
# ============================================================


audio_last_error = None


def find_audio_file():
    script_directory = os.path.dirname(os.path.abspath(__file__))

    if AUDIO_FILE:
        candidate = AUDIO_FILE

        if not os.path.isabs(candidate):
            candidate = os.path.join(script_directory, candidate)

        if os.path.isfile(candidate):
            return candidate

        print(f"[WARNING] 지정한 오디오 파일을 찾지 못했습니다: {candidate}")
        return None

    for pattern in AUDIO_EXTENSIONS:
        matches = sorted(glob.glob(os.path.join(script_directory, pattern)))

        if matches:
            return matches[0]

    return None


def prepare_audio_for_alsa(audio_path):
    global audio_last_error

    if audio_path is None:
        return None

    extension = os.path.splitext(audio_path)[1].lower()

    if extension == ".wav":
        return audio_path

    if shutil.which("ffmpeg") is None:
        audio_last_error = (
            "MP3 변환에 필요한 ffmpeg를 찾지 못했습니다. "
            "sudo apt install ffmpeg 명령으로 설치하십시오."
        )
        print(f"[AUDIO][FAIL] {audio_last_error}")
        return None

    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        audio_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        AUDIO_DECODED_WAV_PATH,
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )

        if result.returncode != 0:
            audio_last_error = (
                result.stderr.strip()
                or f"ffmpeg 변환 종료 코드 {result.returncode}"
            )
            print(f"[AUDIO][FAIL] MP3 변환 실패: {audio_last_error}")
            return None

        if not os.path.isfile(AUDIO_DECODED_WAV_PATH):
            audio_last_error = "변환된 WAV 파일이 생성되지 않았습니다."
            print(f"[AUDIO][FAIL] {audio_last_error}")
            return None

        print(f"[AUDIO][OK] ALSA 재생용 WAV 생성: {AUDIO_DECODED_WAV_PATH}")
        return AUDIO_DECODED_WAV_PATH

    except Exception as convert_error:
        audio_last_error = str(convert_error)
        print(f"[AUDIO][FAIL] MP3 변환 중 오류: {audio_last_error}")
        return None


def build_audio_command(audio_path):
    if (
        audio_path is not None
        and os.path.splitext(audio_path)[1].lower() == ".wav"
        and shutil.which("aplay")
    ):
        return [
            "aplay",
            "-D",
            AUDIO_OUTPUT_DEVICE,
            "-q",
            audio_path,
        ]

    return None


def start_audio_playback(audio_path):
    global audio_last_error

    audio_last_error = None

    if audio_path is None:
        audio_last_error = "오디오 파일을 찾지 못했습니다."
        print(f"[WARNING] 오디오 재생 실패: {audio_last_error}")
        return None

    command = build_audio_command(audio_path)

    if command is None:
        audio_last_error = (
            "aplay 재생 명령을 만들지 못했습니다. "
            "aplay 설치와 WAV 변환 결과를 확인하십시오."
        )
        print(f"[WARNING] {audio_last_error}")
        return None

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        time.sleep(0.05)
        return_code = process.poll()

        if return_code is not None and return_code != 0:
            error_output = ""

            if process.stderr is not None:
                error_output = process.stderr.read().strip()

            audio_last_error = (
                error_output
                or f"재생 프로세스 종료 코드 {return_code}"
            )
            print(f"[WARNING] 오디오 재생 실패: {audio_last_error}")
            return None

        print("[INFO] 오디오 재생을 시작합니다.")
        return process

    except Exception as audio_error:
        audio_last_error = str(audio_error)
        print(f"[WARNING] 오디오 재생 실패: {audio_last_error}")
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


# ============================================================
# 맥북 UDP RTP/JPEG 영상 송출
# ============================================================


def resolve_macbook_ip():
    configured_ip = MACBOOK_IP.strip()

    if configured_ip:
        return configured_ip

    for environment_name in ("SSH_CLIENT", "SSH_CONNECTION"):
        connection_information = os.environ.get(
            environment_name,
            "",
        ).split()

        if connection_information:
            detected_ip = connection_information[0].strip()

            if detected_ip:
                print(
                    f"[INFO] {environment_name}에서 맥북 IP 자동 인식: "
                    f"{detected_ip}"
                )
                return detected_ip

    raise RuntimeError(
        "맥북 IP를 자동 인식하지 못했습니다. "
        "파일 상단의 MACBOOK_IP에 맥북 IP를 입력하십시오."
    )


class GstUdpJpegStreamer:
    def __init__(
        self,
        host,
        port,
        width,
        height,
        fps,
        jpeg_quality,
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
        self.jpeg_quality = int(clamp(jpeg_quality, 1, 100))

        frame_size = self.width * self.height * 3

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
            "async=false",
        ]

        print(f"[INFO] 맥북 영상 송출 시작: {self.host}:{self.port}")

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )

    def send(self, frame):
        if self.process is None:
            return

        if self.process.poll() is not None:
            raise RuntimeError("GStreamer 영상 송출 프로세스가 종료되었습니다.")

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(
                frame,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )

        if not frame.flags["C_CONTIGUOUS"]:
            frame = frame.copy()

        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as stream_error:
            raise RuntimeError("GStreamer 송출 파이프가 끊어졌습니다.") from stream_error

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
        print("[INFO] 맥북 영상 송출을 종료했습니다.")


# ============================================================
# 실행
# ============================================================


print("[INFO] PCA9685를 초기화합니다.")

i2c = ExtendedI2C(1)
kit = ServoKit(
    channels=16,
    i2c=i2c,
    address=PCA9685_ADDRESS,
)

kit.servo[TILT_SERVO_CHANNEL].actuation_range = SERVO_ACTUATION_RANGE
kit.servo[PAN_SERVO_CHANNEL].actuation_range = SERVO_ACTUATION_RANGE

kit.servo[TILT_SERVO_CHANNEL].set_pulse_width_range(MIN_PULSE, MAX_PULSE)
kit.servo[PAN_SERVO_CHANNEL].set_pulse_width_range(MIN_PULSE, MAX_PULSE)

last_written_tilt_angle = None
last_written_pan_angle = None


def set_tilt_servo_angle_safe(angle, force=False):
    global last_written_tilt_angle

    safe_angle = clamp_tilt_angle(angle)

    if (
        force
        or last_written_tilt_angle is None
        or abs(safe_angle - last_written_tilt_angle)
        >= SERVO_DIRECT_WRITE_EPSILON
    ):
        kit.servo[TILT_SERVO_CHANNEL].angle = safe_angle
        last_written_tilt_angle = safe_angle

    return safe_angle


def set_pan_servo_angle_safe(angle, force=False):
    global last_written_pan_angle

    safe_angle = clamp_pan_angle(angle)

    if (
        force
        or last_written_pan_angle is None
        or abs(safe_angle - last_written_pan_angle)
        >= SERVO_DIRECT_WRITE_EPSILON
    ):
        kit.servo[PAN_SERVO_CHANNEL].angle = safe_angle
        last_written_pan_angle = safe_angle

    return safe_angle


print("[INFO] YOLO 모델을 불러옵니다.")
model = YOLO(MODEL_PATH, task="detect")
print("[INFO] 탐지 대상: PERSON only")
print("[INFO] 화면의 사람들을 왼쪽부터 오른쪽 순서로 추적합니다.")
print(f"[INFO] 각 사람에게 오디오를 {AUDIO_REPEAT_COUNT}회 재생합니다.")
print("[INFO] 사람 한 명의 방송이 끝날 때마다 초기 위치로 복귀합니다.")

AUDIO_SOURCE_PATH = find_audio_file()
AUDIO_PATH = prepare_audio_for_alsa(AUDIO_SOURCE_PATH)

if AUDIO_SOURCE_PATH is not None:
    print(f"[INFO] 선택한 오디오 파일: {AUDIO_SOURCE_PATH}")
    print(f"[INFO] 오디오 출력 장치: {AUDIO_OUTPUT_DEVICE}")
else:
    print("[WARNING] 오디오 파일을 찾지 못했습니다.")

camera = None
video_streamer = None
audio_process = None

current_tilt_angle = TILT_INITIAL_ANGLE
current_pan_angle = PAN_INITIAL_ANGLE

try:
    camera = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {CAMERA_DEVICE}")

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    print(
        f"[INFO] 실제 카메라 설정: "
        f"{actual_width} x {actual_height}, {actual_fps:.1f} FPS"
    )

    stream_host = resolve_macbook_ip()
    video_streamer = GstUdpJpegStreamer(
        host=stream_host,
        port=STREAM_PORT,
        width=actual_width,
        height=actual_height,
        fps=STREAM_FPS,
        jpeg_quality=STREAM_JPEG_QUALITY,
    )

    # 프로그램 시작 시 기존 파일과 동일한 초기 각도로 설정합니다.
    current_tilt_angle = set_tilt_servo_angle_safe(
        TILT_INITIAL_ANGLE,
        force=True,
    )
    current_pan_angle = set_pan_servo_angle_safe(
        PAN_INITIAL_ANGLE,
        force=True,
    )

    time.sleep(1.0)

    current_time = time.monotonic()
    last_servo_update_time = current_time

    tilt_servo_velocity = 0.0
    pan_servo_velocity = 0.0

    filtered_target_x = None
    filtered_target_y = None

    previous_error_x = None
    previous_error_y = None

    filtered_derivative_x = 0.0
    filtered_derivative_y = 0.0

    pan_axis_locked = False
    tilt_axis_locked = False

    previous_tilt_limit_state = None
    previous_pan_limit_state = None
    angle_limit_active = False

    # 상태:
    # WAITING_GROUP    : 새 사람 그룹을 기다림
    # COLLECTING_GROUP : 여러 프레임에서 최대 인원 검출 결과를 수집
    # WAITING_NEXT     : 초기 위치에서 다음 순번 사람을 다시 찾음
    # TRACKING      : 현재 사람을 추적하며 오디오를 2회 재생
    # RETURNING     : 현재 사람 방송 종료 후 초기 위치 복귀
    # WAITING_CLEAR : 그룹 전체 완료 후 화면이 비워지기를 기다림
    tracking_state = "WAITING_GROUP"

    tracked_person = None
    last_valid_face = None
    last_person_detection_time = 0.0
    target_held = False

    # 그룹을 처음 감지한 초기 화면의 사람 박스를 왼쪽부터 저장합니다.
    person_queue = []
    current_person_index = -1
    group_total = 0
    group_collection_started = None
    group_candidates = []
    next_person_wait_started = None
    clear_since = None

    # 현재 사람의 오디오 반복 상태입니다.
    audio_play_count = 0
    audio_attempt_started_time = 0.0
    audio_last_end_time = None

    while True:
        success, frame = camera.read()

        if not success:
            print("[WARNING] 카메라 프레임을 읽지 못했습니다.")
            time.sleep(0.05)
            continue

        frame_height, frame_width = frame.shape[:2]
        screen_center_x = frame_width // 2
        screen_center_y = frame_height // 2

        current_time = time.monotonic()
        delta_time = clamp(
            current_time - last_servo_update_time,
            0.001,
            0.1,
        )

        # ----------------------------------------------------
        # 사람만 탐지
        # ----------------------------------------------------
        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        persons = get_person_detections(results[0].boxes)

        for person in persons:
            is_tracked_box = (
                tracked_person is not None
                and normalized_person_match_distance(
                    tracked_person,
                    person,
                ) <= PERSON_MATCH_DISTANCE_RATIO
            )

            draw_person_detection(
                frame,
                person,
                "TARGET PERSON" if is_tracked_box else "PERSON",
                (0, 0, 255) if is_tracked_box else (255, 120, 0),
                3 if is_tracked_box else 2,
            )

        # ----------------------------------------------------
        # 새 그룹 수집: 잠깐 동안 가장 많은 인원이 검출된 프레임을 저장
        # ----------------------------------------------------
        if tracking_state == "WAITING_GROUP" and persons:
            tracking_state = "COLLECTING_GROUP"
            group_collection_started = current_time
            group_candidates = list(persons)
            print("[INFO] 사람 그룹 인원을 수집합니다.")

        if tracking_state == "COLLECTING_GROUP":
            if len(persons) > len(group_candidates):
                group_candidates = list(persons)

            if (
                group_collection_started is not None
                and current_time - group_collection_started
                >= GROUP_COLLECTION_TIME
            ):
                if group_candidates:
                    person_queue = sort_persons_left_to_right(group_candidates)
                    group_total = len(person_queue)
                    current_person_index = 0

                    tracked_person = person_queue[current_person_index]
                    last_person_detection_time = current_time
                    last_valid_face = get_face_target(tracked_person)

                    audio_process = stop_audio_playback(audio_process)
                    audio_play_count = 1
                    audio_process = start_audio_playback(AUDIO_PATH)
                    audio_attempt_started_time = current_time
                    audio_last_end_time = None

                    tracking_state = "TRACKING"
                    clear_since = None

                    print(
                        f"[INFO] 사람 {group_total}명을 등록했습니다. "
                        "왼쪽부터 순서대로 방송합니다."
                    )
                    print(
                        f"[INFO] {current_person_index + 1}/{group_total}번 사람 추적 시작, "
                        f"오디오 {audio_play_count}/{AUDIO_REPEAT_COUNT}회"
                    )
                else:
                    tracking_state = "WAITING_GROUP"

                group_collection_started = None
                group_candidates = []

        # ----------------------------------------------------
        # 초기 위치에서 큐에 저장된 다음 순번 사람 찾기
        # ----------------------------------------------------
        if tracking_state == "WAITING_NEXT":
            if current_person_index >= group_total:
                tracking_state = "WAITING_CLEAR"
                clear_since = None

            else:
                matched_queue, _matched_queue_scores = match_person_queue(
                    person_queue,
                    persons,
                    frame_width,
                    frame_height,
                )

                queued_person = matched_queue.get(current_person_index)

                if queued_person is not None:
                    tracked_person = queued_person
                    last_person_detection_time = current_time
                    last_valid_face = get_face_target(tracked_person)

                    audio_process = stop_audio_playback(audio_process)
                    audio_play_count = 1
                    audio_process = start_audio_playback(AUDIO_PATH)
                    audio_attempt_started_time = current_time
                    audio_last_end_time = None

                    tracking_state = "TRACKING"
                    next_person_wait_started = None

                    print(
                        f"[INFO] {current_person_index + 1}/{group_total}번 사람 추적 시작, "
                        f"오디오 {audio_play_count}/{AUDIO_REPEAT_COUNT}회"
                    )

                else:
                    if next_person_wait_started is None:
                        next_person_wait_started = current_time

                    elif (
                        current_time - next_person_wait_started
                        >= NEXT_PERSON_WAIT_TIME
                    ):
                        print(
                            f"[WARNING] {current_person_index + 1}/{group_total}번 사람이 "
                            "보이지 않아 다음 순번으로 넘어갑니다."
                        )

                        current_person_index += 1
                        next_person_wait_started = current_time

                        if current_person_index >= group_total:
                            tracking_state = "WAITING_CLEAR"
                            clear_since = None
                            next_person_wait_started = None

        # ----------------------------------------------------
        # 그룹 전체 방송 완료 후 같은 사람들에게 다시 방송하지 않도록 대기
        # ----------------------------------------------------
        if tracking_state == "WAITING_CLEAR":
            if persons:
                clear_since = None
            else:
                if clear_since is None:
                    clear_since = current_time
                elif current_time - clear_since >= REARM_CLEAR_TIME:
                    person_queue = []
                    group_total = 0
                    current_person_index = -1
                    group_collection_started = None
                    group_candidates = []
                    tracking_state = "WAITING_GROUP"
                    clear_since = None
                    print("[INFO] 화면이 비워져 새 사람 그룹 탐지를 시작합니다.")

        # ----------------------------------------------------
        # 등록한 현재 사람만 계속 추적하며 오디오를 두 번 재생
        # ----------------------------------------------------
        face_target = None
        target_held = False

        if tracking_state == "TRACKING":
            matched_person, _match_ratio = find_matching_person(
                tracked_person,
                persons,
            )

            if matched_person is not None:
                tracked_person = matched_person
                face_target = get_face_target(matched_person)
                last_valid_face = face_target
                last_person_detection_time = current_time

            elif (
                last_valid_face is not None
                and current_time - last_person_detection_time
                <= PERSON_HOLD_TIME
            ):
                face_target = last_valid_face
                target_held = True

            else:
                face_target = None
                last_valid_face = None

            # 실행 중이던 오디오가 끝났는지 확인합니다.
            if (
                audio_process is not None
                and audio_process.poll() is not None
            ):
                audio_process = stop_audio_playback(audio_process)
                audio_last_end_time = current_time

            # 오디오 장치 오류로 프로세스가 시작되지 않은 경우에도
            # 일정 시간 뒤 해당 재생 시도를 완료한 것으로 처리합니다.
            if (
                audio_process is None
                and audio_last_end_time is None
                and current_time - audio_attempt_started_time
                >= AUDIO_FAILURE_RETURN_DELAY
            ):
                audio_last_end_time = current_time

            # 앞 재생이 끝났으면 짧은 간격 뒤 두 번째 재생을 시작합니다.
            if (
                audio_process is None
                and audio_last_end_time is not None
                and audio_play_count < AUDIO_REPEAT_COUNT
                and current_time - audio_last_end_time
                >= AUDIO_REPEAT_GAP
            ):
                audio_play_count += 1
                audio_process = start_audio_playback(AUDIO_PATH)
                audio_attempt_started_time = current_time
                audio_last_end_time = None

                print(
                    f"[INFO] {current_person_index + 1}/{group_total}번 사람, "
                    f"오디오 {audio_play_count}/{AUDIO_REPEAT_COUNT}회"
                )

            # 두 번째 재생까지 끝나면 초기 위치 복귀로 전환합니다.
            audio_cycle_finished = (
                audio_process is None
                and audio_last_end_time is not None
                and audio_play_count >= AUDIO_REPEAT_COUNT
            )

            if audio_cycle_finished:
                audio_process = stop_audio_playback(audio_process)
                tracking_state = "RETURNING"

                tracked_person = None
                last_valid_face = None
                face_target = None
                target_held = False

                filtered_target_x = None
                filtered_target_y = None
                previous_error_x = None
                previous_error_y = None
                filtered_derivative_x = 0.0
                filtered_derivative_y = 0.0
                pan_axis_locked = False
                tilt_axis_locked = False

                print(
                    f"[INFO] {current_person_index + 1}/{group_total}번 사람에게 "
                    f"오디오 {AUDIO_REPEAT_COUNT}회 방송 완료. 초기 위치로 복귀합니다."
                )

        # ----------------------------------------------------
        # 화면 중심 및 데드존 표시
        # ----------------------------------------------------
        cv2.line(
            frame,
            (0, screen_center_y),
            (frame_width, screen_center_y),
            (255, 255, 255),
            2,
        )
        cv2.line(
            frame,
            (screen_center_x, 0),
            (screen_center_x, frame_height),
            (255, 255, 255),
            2,
        )
        cv2.rectangle(
            frame,
            (
                screen_center_x - DEAD_ZONE_X,
                screen_center_y - DEAD_ZONE_Y,
            ),
            (
                screen_center_x + DEAD_ZONE_X,
                screen_center_y + DEAD_ZONE_Y,
            ),
            (100, 100, 100),
            1,
        )
        cv2.rectangle(
            frame,
            (
                screen_center_x - CENTER_RELEASE_ZONE_X,
                screen_center_y - CENTER_RELEASE_ZONE_Y,
            ),
            (
                screen_center_x + CENTER_RELEASE_ZONE_X,
                screen_center_y + CENTER_RELEASE_ZONE_Y,
            ),
            (120, 120, 0),
            1,
        )
        cv2.circle(
            frame,
            (screen_center_x, screen_center_y),
            7,
            (255, 255, 255),
            2,
        )

        error_x = None
        error_y = None
        p_term_x = 0.0
        d_term_x = 0.0
        p_term_y = 0.0
        d_term_y = 0.0

        # ----------------------------------------------------
        # 사람 얼굴 중심 팬-틸트 PD 추적
        # ----------------------------------------------------
        if tracking_state == "TRACKING" and face_target is not None:
            angle_limit_active = False

            face_center_x = face_target["center_x"]
            raw_face_center_y = face_target["center_y"]
            face_x1, face_y1, face_x2, face_y2 = face_target["box"]

            face_color = (0, 165, 255) if target_held else (0, 0, 255)

            cv2.rectangle(
                frame,
                (max(0, face_x1), max(0, face_y1)),
                (
                    min(frame_width - 1, face_x2),
                    min(frame_height - 1, face_y2),
                ),
                face_color,
                2,
            )
            cv2.circle(
                frame,
                (int(face_center_x), int(raw_face_center_y)),
                9,
                face_color,
                -1,
            )

            if filtered_target_x is None:
                filtered_target_x = face_center_x
            elif not target_held:
                filtered_target_x = (
                    TARGET_FILTER_ALPHA * face_center_x
                    + (1.0 - TARGET_FILTER_ALPHA) * filtered_target_x
                )

            if filtered_target_y is None:
                filtered_target_y = raw_face_center_y
            elif not target_held:
                filtered_target_y = (
                    TARGET_FILTER_ALPHA * raw_face_center_y
                    + (1.0 - TARGET_FILTER_ALPHA) * filtered_target_y
                )

            error_x = filtered_target_x - screen_center_x
            error_y = filtered_target_y - screen_center_y

            cv2.line(
                frame,
                (screen_center_x, screen_center_y),
                (int(filtered_target_x), int(filtered_target_y)),
                (255, 0, 255),
                2,
            )

            if current_time - last_servo_update_time >= SERVO_UPDATE_INTERVAL:
                if pan_axis_locked:
                    if abs(error_x) >= CENTER_RELEASE_ZONE_X:
                        pan_axis_locked = False
                elif abs(error_x) <= DEAD_ZONE_X:
                    pan_axis_locked = True

                if tilt_axis_locked:
                    if abs(error_y) >= CENTER_RELEASE_ZONE_Y:
                        tilt_axis_locked = False
                elif abs(error_y) <= DEAD_ZONE_Y:
                    tilt_axis_locked = True

                if pan_axis_locked:
                    target_pan_velocity = 0.0
                    filtered_derivative_x *= CENTER_VELOCITY_DAMPING
                else:
                    (
                        target_pan_velocity,
                        filtered_derivative_x,
                        p_term_x,
                        d_term_x,
                    ) = calculate_axis_target_velocity(
                        error=error_x,
                        previous_error=previous_error_x,
                        filtered_derivative=filtered_derivative_x,
                        dead_zone=DEAD_ZONE_X,
                        slow_zone=SLOW_ZONE_X,
                        direction=PAN_SERVO_DIRECTION,
                        delta_time=delta_time,
                        target_held=target_held,
                    )

                if tilt_axis_locked:
                    target_tilt_velocity = 0.0
                    filtered_derivative_y *= CENTER_VELOCITY_DAMPING
                else:
                    (
                        target_tilt_velocity,
                        filtered_derivative_y,
                        p_term_y,
                        d_term_y,
                    ) = calculate_axis_target_velocity(
                        error=error_y,
                        previous_error=previous_error_y,
                        filtered_derivative=filtered_derivative_y,
                        dead_zone=DEAD_ZONE_Y,
                        slow_zone=SLOW_ZONE_Y,
                        direction=TILT_SERVO_DIRECTION,
                        delta_time=delta_time,
                        target_held=target_held,
                    )

                pan_servo_velocity = (
                    0.0 if pan_axis_locked else target_pan_velocity
                )
                tilt_servo_velocity = (
                    0.0 if tilt_axis_locked else target_tilt_velocity
                )

                requested_pan_angle = (
                    current_pan_angle + pan_servo_velocity * delta_time
                )
                requested_tilt_angle = (
                    current_tilt_angle + tilt_servo_velocity * delta_time
                )

                if requested_pan_angle < PAN_MIN_ANGLE:
                    current_pan_angle = set_pan_servo_angle_safe(PAN_MIN_ANGLE)
                    angle_limit_active = True
                    pan_servo_velocity = max(0.0, pan_servo_velocity)

                    if previous_pan_limit_state != "MIN":
                        print(f"[SAFETY] 팬 최소각도 {PAN_MIN_ANGLE:.1f}도 제한")
                        previous_pan_limit_state = "MIN"

                elif requested_pan_angle > PAN_MAX_ANGLE:
                    current_pan_angle = set_pan_servo_angle_safe(PAN_MAX_ANGLE)
                    angle_limit_active = True
                    pan_servo_velocity = min(0.0, pan_servo_velocity)

                    if previous_pan_limit_state != "MAX":
                        print(f"[SAFETY] 팬 최대각도 {PAN_MAX_ANGLE:.1f}도 제한")
                        previous_pan_limit_state = "MAX"

                else:
                    current_pan_angle = set_pan_servo_angle_safe(
                        requested_pan_angle
                    )
                    previous_pan_limit_state = None

                if requested_tilt_angle < TILT_MIN_ANGLE:
                    current_tilt_angle = set_tilt_servo_angle_safe(
                        TILT_MIN_ANGLE
                    )
                    angle_limit_active = True
                    tilt_servo_velocity = max(0.0, tilt_servo_velocity)

                    if previous_tilt_limit_state != "MIN":
                        print(
                            f"[SAFETY] 틸트 최소각도 "
                            f"{TILT_MIN_ANGLE:.1f}도 제한"
                        )
                        previous_tilt_limit_state = "MIN"

                elif requested_tilt_angle > TILT_MAX_ANGLE:
                    current_tilt_angle = set_tilt_servo_angle_safe(
                        TILT_MAX_ANGLE
                    )
                    angle_limit_active = True
                    tilt_servo_velocity = min(0.0, tilt_servo_velocity)

                    if previous_tilt_limit_state != "MAX":
                        print(
                            f"[SAFETY] 틸트 최대각도 "
                            f"{TILT_MAX_ANGLE:.1f}도 제한"
                        )
                        previous_tilt_limit_state = "MAX"

                else:
                    current_tilt_angle = set_tilt_servo_angle_safe(
                        requested_tilt_angle
                    )
                    previous_tilt_limit_state = None

                if not target_held:
                    previous_error_x = error_x
                    previous_error_y = error_y

                last_servo_update_time = current_time

        elif tracking_state != "RETURNING":
            filtered_target_x = None
            filtered_target_y = None
            previous_error_x = None
            previous_error_y = None
            filtered_derivative_x = 0.0
            filtered_derivative_y = 0.0
            pan_axis_locked = False
            tilt_axis_locked = False
            pan_servo_velocity = 0.0
            tilt_servo_velocity = 0.0
            angle_limit_active = False

        # ----------------------------------------------------
        # 오디오 종료 후 부드럽게 초기 위치 복귀
        # ----------------------------------------------------
        if tracking_state == "RETURNING":
            if current_time - last_servo_update_time >= RETURN_INTERVAL:
                pan_return_error = PAN_INITIAL_ANGLE - current_pan_angle
                tilt_return_error = TILT_INITIAL_ANGLE - current_tilt_angle

                target_pan_return_velocity = clamp(
                    pan_return_error * RETURN_KP,
                    -RETURN_MAX_SPEED,
                    RETURN_MAX_SPEED,
                )
                target_tilt_return_velocity = clamp(
                    tilt_return_error * RETURN_KP,
                    -RETURN_MAX_SPEED,
                    RETURN_MAX_SPEED,
                )

                pan_servo_velocity = update_velocity_smoothly(
                    current_velocity=pan_servo_velocity,
                    target_velocity=target_pan_return_velocity,
                    max_acceleration=RETURN_ACCELERATION,
                    delta_time=delta_time,
                )
                tilt_servo_velocity = update_velocity_smoothly(
                    current_velocity=tilt_servo_velocity,
                    target_velocity=target_tilt_return_velocity,
                    max_acceleration=RETURN_ACCELERATION,
                    delta_time=delta_time,
                )

                next_pan_angle = (
                    current_pan_angle + pan_servo_velocity * delta_time
                )
                next_tilt_angle = (
                    current_tilt_angle + tilt_servo_velocity * delta_time
                )

                if (
                    current_pan_angle > PAN_INITIAL_ANGLE
                    and next_pan_angle < PAN_INITIAL_ANGLE
                ) or (
                    current_pan_angle < PAN_INITIAL_ANGLE
                    and next_pan_angle > PAN_INITIAL_ANGLE
                ):
                    next_pan_angle = PAN_INITIAL_ANGLE

                if (
                    current_tilt_angle > TILT_INITIAL_ANGLE
                    and next_tilt_angle < TILT_INITIAL_ANGLE
                ) or (
                    current_tilt_angle < TILT_INITIAL_ANGLE
                    and next_tilt_angle > TILT_INITIAL_ANGLE
                ):
                    next_tilt_angle = TILT_INITIAL_ANGLE

                current_pan_angle = set_pan_servo_angle_safe(next_pan_angle)
                current_tilt_angle = set_tilt_servo_angle_safe(next_tilt_angle)
                last_servo_update_time = current_time

            pan_return_complete = (
                abs(current_pan_angle - PAN_INITIAL_ANGLE) <= 0.08
                and abs(pan_servo_velocity) <= 0.35
            )
            tilt_return_complete = (
                abs(current_tilt_angle - TILT_INITIAL_ANGLE) <= 0.08
                and abs(tilt_servo_velocity) <= 0.35
            )

            if pan_return_complete and tilt_return_complete:
                current_pan_angle = set_pan_servo_angle_safe(
                    PAN_INITIAL_ANGLE,
                    force=True,
                )
                current_tilt_angle = set_tilt_servo_angle_safe(
                    TILT_INITIAL_ANGLE,
                    force=True,
                )

                pan_servo_velocity = 0.0
                tilt_servo_velocity = 0.0

                current_person_index += 1
                audio_play_count = 0
                audio_last_end_time = None

                if current_person_index < group_total:
                    tracking_state = "WAITING_NEXT"
                    next_person_wait_started = current_time
                    print(
                        "[INFO] 초기 위치 복귀 완료. "
                        f"다음 {current_person_index + 1}/{group_total}번 사람을 찾습니다."
                    )
                else:
                    tracking_state = "WAITING_CLEAR"
                    next_person_wait_started = None
                    clear_since = None
                    print(
                        f"[INFO] 등록된 사람 {group_total}명 모두에게 "
                        f"오디오 {AUDIO_REPEAT_COUNT}회씩 방송했습니다."
                    )

        # ----------------------------------------------------
        # 화면 상태 표시
        # ----------------------------------------------------
        if tracking_state == "WAITING_GROUP":
            status_text = "WAITING FOR PERSON GROUP"
        elif tracking_state == "COLLECTING_GROUP":
            status_text = "COLLECTING PERSON GROUP"
        elif tracking_state == "WAITING_NEXT":
            status_text = "SELECTING NEXT PERSON"
        elif tracking_state == "WAITING_CLEAR":
            status_text = "GROUP DONE - WAITING SCENE CLEAR"
        elif tracking_state == "RETURNING":
            status_text = "PERSON AUDIO 2X DONE - RETURNING"
        elif target_held:
            status_text = "TRACKING PERSON - TARGET HELD"
        elif angle_limit_active:
            status_text = "TRACKING PERSON - ANGLE LIMIT"
        elif face_target is None:
            status_text = "TRACKING PERSON - TEMPORARILY LOST"
        else:
            status_text = "TRACKING PERSON + AUDIO 2X"

        audio_running = (
            audio_process is not None
            and audio_process.poll() is None
        )

        if audio_last_error is not None:
            audio_status = "ERROR"
        elif audio_running:
            audio_status = (
                f"PLAYING {audio_play_count}/{AUDIO_REPEAT_COUNT}"
            )
        elif tracking_state == "TRACKING":
            audio_status = (
                f"GAP {audio_play_count}/{AUDIO_REPEAT_COUNT}"
            )
        elif tracking_state == "RETURNING":
            audio_status = "FINISHED"
        else:
            audio_status = "OFF"

        cv2.putText(
            frame,
            f"Status: {status_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Persons: {len(persons)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (0, 255, 255),
            2,
        )
        if group_total > 0:
            display_index = min(current_person_index + 1, group_total)
            cv2.putText(
                frame,
                f"Group target: {display_index}/{group_total}",
                (frame_width - 230, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (0, 255, 255),
                2,
            )
        cv2.putText(
            frame,
            f"AUDIO: {audio_status}",
            (frame_width - 230, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 0, 255) if audio_running else (180, 180, 180),
            2,
        )

        if error_x is not None and error_y is not None:
            cv2.putText(
                frame,
                f"Face error X/Y: {error_x:.1f} / {error_y:.1f}px",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                (
                    f"Pan/Tilt speed: {pan_servo_velocity:.2f} / "
                    f"{tilt_servo_velocity:.2f} deg/s"
                ),
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 0, 255),
                2,
            )

        cv2.putText(
            frame,
            f"Pan CH{PAN_SERVO_CHANNEL}: {current_pan_angle:.2f} deg",
            (10, frame_height - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Tilt CH{TILT_SERVO_CHANNEL}: {current_tilt_angle:.2f} deg",
            (10, frame_height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Detection: PERSON only / Audio: {AUDIO_REPEAT_COUNT} playbacks each",
            (10, frame_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (200, 255, 200),
            2,
        )

        video_streamer.send(frame)

        if ENABLE_LOCAL_PREVIEW:
            cv2.imshow("Sequential Multi-Person Tracking - Audio Twice", frame)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = 255

        # r: 현재 사람 방송을 중지하고 초기 위치로 복귀한 뒤 다음 사람으로 진행
        if key == ord("r"):
            audio_process = stop_audio_playback(audio_process)

            if tracking_state == "TRACKING":
                tracking_state = "RETURNING"
                tracked_person = None
                last_valid_face = None
                audio_last_end_time = None
                print("[INFO] 현재 사람 방송을 건너뛰고 초기 위치로 복귀합니다.")
            elif tracking_state in (
                "COLLECTING_GROUP",
                "WAITING_NEXT",
                "WAITING_CLEAR",
            ):
                person_queue = []
                group_total = 0
                current_person_index = -1
                group_collection_started = None
                group_candidates = []
                tracking_state = "WAITING_GROUP"
                clear_since = None
                print("[INFO] 사람 큐를 초기화했습니다.")

        if key == ord("q") or key == 27:
            break

except KeyboardInterrupt:
    print("\n[INFO] 사용자가 프로그램을 중단했습니다.")

except Exception as main_error:
    print(f"\n[ERROR] 프로그램 실행 중 오류 발생: {main_error}")
    raise

finally:
    try:
        audio_process = stop_audio_playback(audio_process)

        print("[INFO] 팬과 틸트 서보를 초기 위치로 복귀합니다.")

        pan_return_velocity = 0.0
        tilt_return_velocity = 0.0
        previous_time = time.monotonic()

        while (
            abs(current_pan_angle - PAN_INITIAL_ANGLE) > 0.05
            or abs(current_tilt_angle - TILT_INITIAL_ANGLE) > 0.05
        ):
            now = time.monotonic()
            delta_time = clamp(now - previous_time, 0.001, 0.1)
            previous_time = now

            target_pan_velocity = clamp(
                (PAN_INITIAL_ANGLE - current_pan_angle) * RETURN_KP,
                -RETURN_MAX_SPEED,
                RETURN_MAX_SPEED,
            )
            target_tilt_velocity = clamp(
                (TILT_INITIAL_ANGLE - current_tilt_angle) * RETURN_KP,
                -RETURN_MAX_SPEED,
                RETURN_MAX_SPEED,
            )

            pan_return_velocity = update_velocity_smoothly(
                pan_return_velocity,
                target_pan_velocity,
                RETURN_ACCELERATION,
                delta_time,
            )
            tilt_return_velocity = update_velocity_smoothly(
                tilt_return_velocity,
                target_tilt_velocity,
                RETURN_ACCELERATION,
                delta_time,
            )

            next_pan_angle = (
                current_pan_angle + pan_return_velocity * delta_time
            )
            next_tilt_angle = (
                current_tilt_angle + tilt_return_velocity * delta_time
            )

            if (
                current_pan_angle > PAN_INITIAL_ANGLE
                and next_pan_angle < PAN_INITIAL_ANGLE
            ) or (
                current_pan_angle < PAN_INITIAL_ANGLE
                and next_pan_angle > PAN_INITIAL_ANGLE
            ):
                next_pan_angle = PAN_INITIAL_ANGLE

            if (
                current_tilt_angle > TILT_INITIAL_ANGLE
                and next_tilt_angle < TILT_INITIAL_ANGLE
            ) or (
                current_tilt_angle < TILT_INITIAL_ANGLE
                and next_tilt_angle > TILT_INITIAL_ANGLE
            ):
                next_tilt_angle = TILT_INITIAL_ANGLE

            current_pan_angle = set_pan_servo_angle_safe(next_pan_angle)
            current_tilt_angle = set_tilt_servo_angle_safe(next_tilt_angle)
            time.sleep(RETURN_INTERVAL)

        current_pan_angle = set_pan_servo_angle_safe(
            PAN_INITIAL_ANGLE,
            force=True,
        )
        current_tilt_angle = set_tilt_servo_angle_safe(
            TILT_INITIAL_ANGLE,
            force=True,
        )

        print("[INFO] 팬과 틸트 서보 초기 위치 복귀 완료")

    except Exception as servo_return_error:
        print(f"[WARNING] 종료 중 서보 복귀 실패: {servo_return_error}")

    finally:
        if video_streamer is not None:
            try:
                video_streamer.stop()
            except Exception as stream_stop_error:
                print(
                    f"[WARNING] 영상 송출 종료 중 오류: "
                    f"{stream_stop_error}"
                )

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()
        print("[INFO] 카메라를 종료했습니다.")
        print("[INFO] 프로그램을 종료했습니다.")
