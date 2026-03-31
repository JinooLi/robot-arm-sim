# Robot Arm Sim

7-DOF 로봇 팔(Franka Panda)의 안전 제어 시뮬레이션 환경입니다. PyBullet 물리 엔진과 Pinocchio 동역학 라이브러리를 기반으로, CLBF(Control Lyapunov Barrier Function) 제어기를 통해 장애물을 회피하면서 목표 지점으로 end-effector를 이동시킵니다.

## 요구 사항

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (패키지 매니저)

## 실행

```bash
# 기본 설정으로 실행
uv run main.py

# 커스텀 설정 파일 지정
uv run main.py --config configs/default.yaml
```

실행하면 PyBullet GUI 창이 열리고, 시뮬레이션 종료 후 `results/` 디렉토리에 데이터와 분석 리포트가 저장됩니다.

## 시각화 GUI 

```bash
# 제일 최근 데이터 시각화
uv run viewer.py 

# 원하는 데이터 시각화
uv run viewer.py ./results/<원하는 데이터 디렉토리 경로>
```

실행하면 matplotlib GUI 창이 열리며 데이터를 시각화합니다.

## 프로젝트 구조

```
robot-arm-sim/
├── configs/
│   └── default.yaml                 # 시뮬레이션 설정 파일
├── docs/
│   └── configuration_guide.md       # 장애물 구성 및 제어기 구현 가이드
├── example/
│   └── control.py                   # CLBF 제어기 구현 예시 (참고용)
├── src/robot_arm_sim/
│   ├── interfaces/                  # 추상 인터페이스
│   │   ├── controller.py            #   ControllerInterface, ControlOutput
│   │   └── simulator.py             #   SimulatorInterface, RobotState
│   ├── controller/                  # 제어기 구현
│   │   ├── pd_controller.py         #   PD 제어기
│   │   └── clbf_controller.py       #   CLBF 제어기
│   ├── simulator/
│   │   └── pybullet_sim.py          # PyBullet 시뮬레이터
│   ├── dynamics/
│   │   └── pinocchio_model.py       # Pinocchio 동역학 모델
│   ├── data/
│   │   └── engine.py                # 시뮬레이션 데이터 기록
│   ├── visualization/
│   │   └── analytics.py             # 결과 시각화 및 리포트
|   |   └── viewer.py                # GUI로 시각화
│   ├── config/
│   │   └── manager.py               # YAML 설정 관리
│   └── main.py                      # 진입점
├── results/                         # 시뮬레이션 결과 출력
├── pyproject.toml
└── README.md
```

## 제어기

### PD Controller

태스크 공간 PD 제어기로, 중력 보상을 포함합니다. 장애물 회피 기능은 없으며, 기본 동작 확인 및 비교 기준으로 사용합니다.

```yaml
controller:
  type: "pd"
```

### CLBF Controller

Control Lyapunov Barrier Function 기반 제어기입니다. Lyapunov 함수로 목표 수렴을 보장하고, Barrier 함수로 장애물 회피를 수행합니다. 모든 관절 링크와 end-effector에 대해 softmax 가중치로 안전 제어를 합성합니다.

```yaml
controller:
  type: "clbf"
  control_mode: "velocity"
  clbf:
    k: 10.0                       # Barrier 함수 이득
    s: 1.0                        # Barrier 함수 스케일
    barrier_radius_margin: 0.05   # 장애물 안전 마진 (m)
    alpha: 1.0                    # Softmax 온도 파라미터
```

## 설정

`configs/default.yaml`에서 시뮬레이션 환경, 로봇, 제어기, 목표 위치, 장애물 등을 설정합니다.

장애물 구성 및 커스텀 제어기 구현에 대한 자세한 내용은 [Configuration Guide](docs/configuration_guide.md)를 참고하세요.

## 결과 출력

시뮬레이션이 완료되면 `results/` 디렉토리에 다음이 저장됩니다:

- 시뮬레이션 데이터 (CSV/Parquet)
- 사용된 설정 파일 사본
- 분석 리포트 (관절 상태, 태스크 공간 궤적, Barrier/Lyapunov 값 그래프)

## 라이선스

MIT
