# Capstone Project: Jetson Nano 기반 객체 추적 및 경보 시스템

이 프로젝트는 NVIDIA Jetson 보드와 ServoKit 팬/틸트(Pan/Tilt) 모터, YOLOv8(yolo26n) 모델을 활용하여 실시간으로 사람 및 객체(컵)를 탐지하고 추적 및 오디오 경보를 수행하는 시스템입니다.

---

## 📂 폴더 구조 (Folder Structure)

```text
Capstone_projects/
├── assets/                     # 프로젝트 리소스 폴더
│   ├── audio/                  # 경보 및 안내 방송 오디오 (.wav, .mp3) 
│   │   ├── hello_sos.mp3       # "sos팀입니다 안녕하세요" 오디오 파일
│   │   ├── hello_sos.wav       
│   │   ├── warning.mp3         # "쓰레기를 제대로 버리세요" 오디오 파일
│   │   └── warning.wav
│   └── models/                 # YOLO 모델 가중치 파일
│       ├── yolo26n.pt          # PyTorch 가중치
│       ├── yolo26n.onnx        # ONNX 포맷 -> 어떤 하드웨어 도구든 이 모델의 구조 이해할 수 있게 변형
│       ├── yolo26n.engine      # TensorRT Engine (젯슨 가속용)
│       └── yolo26n_960.engine
│
├── dataset/                    # 학습 및 태깅 데이터셋 폴더
│   ├── images/                 # 쓰레기/객체 이미지 파일들 (.jpg)
│   └── annotations/            # Pascal VOC 형식의 XML 라벨링 파일들 (.xml)
│
├── archive/                    # 이전 버전 파일 백업 보관소. 업그레이드 단계별 정리 파일들
│   ├── final_1.py ~ final_5.py
│   ├── final_intro.py ~ final_intro_2.py
│   └── cup_tracking_fast_interruptible_return_*.py
│
├── main.py                     # 실시간 사람 및 컵 분리 추적 프로그램
├── main_intro.py               # 인트로 및 사람 그룹 안내 방송 프로그램
├── yolo_basic_model.py         # YOLO 모델 기반의 단순 GPU 사람 검출 및 테스트 프로그램
├── export_yolo26_engine.py     # .pt 가중치를 TensorRT (.engine) 파일로 컴파일하는 스크립트
├── .gitignore                  # Git 커밋 제외 설정 파일
└── README.md                   # 프로젝트 개요 및 설명서
```

---

## 🛠️ 주요 기능 설명 (Key Scripts)

1. **`main.py`**
   - YOLO 모델을 로드하여 사람과 컵을 동시에 검출합니다.
   - 컵과 사람이 분리되었을 때 경고 오디오(`assets/audio/warning.mp3`)를 자동으로 변환 및 재생하고, 팬-틸트를 사용하여 컵 소유자를 지속적으로 정밀 추적합니다.

2. **`main_intro.py`**
   - 카메라 뷰에 사람이 등장하면 인트로 오디오 안내 방송(`assets/audio/hello_sos.mp3`)을 송출합니다.
   - 단체(그룹)로 사람이 감지되었을 때 순차적으로 타겟팅하여 얼굴/사람 중심 방향으로 카메라가 바라볼 수 있도록 추적합니다.

3. **`export_yolo26_engine.py`**
   - 젯슨 GPU 하드웨어 가속을 극대화하기 위해 `yolo26n.pt`를 TensorRT `.engine` 파일로 빌드 및 내보내는 도구입니다.

4. **`yolo_basic_model.py`**
   - 기본적인 실시간 카메라 테스트용도로, 화면 중심과 검출 객체 간의 픽셀 에러를 계산하여 화면에 표시합니다.

---

## 🚀 시작 가이드 (Getting Started)

### 1. 필수 의존성 설치
젯슨 및 터미널 환경에 맞춰 다음 패키지들을 설치해야 합니다:
```bash
pip install ultralytics adafruit-circuitpython-servokit opencv-python torch torchvision
```

### 2. 가속 모델 변환 (TensorRT Engine Build)
가속엔진 빌드가 필요하다면 아래 명령을 실행합니다:
```bash
python3 export_yolo26_engine.py
```
*(성공적으로 컴파일 완료 시 `assets/models/yolo26n.engine` 파일이 생성됩니다.)*

### 3. 메인 프로그램 구동
```bash
# 사람 및 컵 분리 경보 시스템 실행
python3 main.py

# 사람 탐지 및 그룹 소개 인트로 방송 실행
python3 main_intro.py
```

---

## ⚠️ 주의 사항 (Important Notes)

- **TensorRT Engine 호환성**: `assets/models/*.engine` 파일은 빌드한 기기의 **GPU 아키텍처 및 TensorRT 버전**에 종속됩니다. 다른 환경에서 이 리포지토리를 복제(Clone)하여 실행할 경우, `export_yolo26_engine.py` 스크립트를 사용하여 다시 엔진 파일을 생성해 주어야 합니다. 처음에 YOLO26n.py 파일을 다운로드 받은다음, 이를 .engine 형식으로 변형해서 연산 작용을 수월하게 만들어 줍니다. 그러므로, 파일 형식을 바꿔주는 스텝이 중요합니다.
