# Configuration Guide

이 문서는 시뮬레이션 환경의 장애물 구성과 커스텀 제어기 구현 방법을 설명합니다.

---

## 1. 장애물(Obstacle) 구성

장애물은 `configs/default.yaml`의 `obstacles` 섹션에서 정의합니다. 각 장애물은 구(Sphere) 형태이며, 시뮬레이션 환경(PyBullet GUI)에 시각적으로 렌더링됩니다.

### 파라미터

| 파라미터   | 타입           | 설명                                                  |
|-----------|---------------|------------------------------------------------------|
| `position` | `[x, y, z]`  | 장애물 중심의 월드 좌표 (단위: m)                       |
| `radius`   | `float`       | 장애물 반지름 (단위: m)                                |
| `color`    | `[r, g, b, a]`| RGBA 색상 (각 값 0.0~1.0, a는 투명도)                  |

### 예시

```yaml
obstacles:
  # 빨간 구: 작업 공간 중앙
  - position: [0.5, 0.0, 0.4]
    radius: 0.1
    color: [1, 0, 0, 0.5]

  # 주황 구: 좌측 하단
  - position: [0.3, 0.3, 0.3]
    radius: 0.08
    color: [1, 0.5, 0, 0.5]

  # 장애물 추가: 리스트에 항목을 추가하면 됩니다
  - position: [0.4, -0.2, 0.5]
    radius: 0.12
    color: [0, 0, 1, 0.3]
```

### 주의사항

- `position`은 로봇 베이스 프레임 기준 월드 좌표입니다. Panda 로봇의 작업 반경은 대략 0.855m이므로, 이 범위 내에 배치해야 의미가 있습니다.
- `radius`는 물리 충돌 및 시각 렌더링 모두에 적용됩니다.
- 제어기에서 안전 마진을 적용하려면 `controller.clbf.barrier_radius_margin` 값을 조정하세요. 이 값이 장애물 반지름에 더해져 Barrier Function의 안전 영역을 결정합니다.
- 장애물 개수에 제한은 없으나, 많을수록 제어기 연산량이 증가합니다.

---

## 2. 제어기(Controller) 구현

제어기는 `ControllerInterface` 추상 클래스를 상속하여 구현합니다.

### 인터페이스 구조

```python
# src/robot_arm_sim/interfaces/controller.py

class ControlMode(Enum):
    VELOCITY = auto()  # 관절 속도 명령
    TORQUE = auto()    # 관절 토크 명령

@dataclass
class ControlOutput:
    command: np.ndarray      # 제어 명령 (7-DOF)
    mode: ControlMode        # 제어 모드
    barrier_value: float     # Barrier Function 값 (안전 지표)
    lyapunov_value: float    # Lyapunov Function 값 (수렴 지표)
    info: dict[str, Any]     # 추가 디버깅/로깅 정보

class ControllerInterface(ABC):
    def setup(self, config: dict[str, Any]) -> None: ...
    def compute(self, state: RobotState, target: np.ndarray, obstacles: list[dict]) -> ControlOutput: ...
    def reset(self) -> None: ...
    def control_mode(self) -> ControlMode: ...  # property
```

### 커스텀 제어기 구현 예시 (CLBF)

`src/robot_arm_sim/controller/` 아래에 새 파일을 생성합니다:

```python
# src/robot_arm_sim/controller/clbf_controller.py

from typing import Any
import numpy as np
from ..dynamics.pinocchio_model import PinocchioModel
from ..interfaces.controller import ControllerInterface, ControlMode, ControlOutput
from ..interfaces.simulator import RobotState


class CLBFController(ControllerInterface):
    def __init__(self, dynamics: PinocchioModel) -> None:
        self._dynamics = dynamics
        self._mode = ControlMode.TORQUE
        # CLBF 파라미터
        self._barrier_gain = 10.0
        self._lyapunov_gain = 1.0
        self._barrier_margin = 0.05
        self._alpha = 1.0

    def setup(self, config: dict[str, Any]) -> None:
        ctrl_cfg = config["controller"]
        clbf_cfg = ctrl_cfg.get("clbf", {})
        self._barrier_gain = clbf_cfg.get("barrier_gain", 10.0)
        self._lyapunov_gain = clbf_cfg.get("lyapunov_gain", 1.0)
        self._barrier_margin = clbf_cfg.get("barrier_radius_margin", 0.05)
        self._alpha = clbf_cfg.get("alpha", 1.0)

        mode_str = ctrl_cfg.get("control_mode", "torque")
        self._mode = ControlMode.TORQUE if mode_str == "torque" else ControlMode.VELOCITY

    def compute(
        self,
        state: RobotState,
        target: np.ndarray,
        obstacles: list[dict[str, Any]],
    ) -> ControlOutput:
        q = state.joint_positions
        dq = state.joint_velocities

        ee_pos = self._dynamics.forward_kinematics(q)
        J = self._dynamics.jacobian(q)[:3, :7]
        M = self._dynamics.mass_matrix(q)
        g = self._dynamics.gravity_vector(q)

        # --- 여기에 CLBF 로직을 구현하세요 ---
        # 1. Lyapunov 함수: V(x) = 0.5 * ||ee_pos - target||^2
        # 2. Barrier 함수: 각 장애물에 대해 h(x) = ||ee_pos - obs||^2 - (r + margin)^2
        # 3. QP 또는 해석적 방법으로 안전 제약 하에서 토크 계산
        #
        # barrier_value = min(h_i) for all obstacles
        # lyapunov_value = V(x)

        tau = np.zeros(7)           # 계산된 토크
        barrier_value = 0.0         # Barrier 함수 최솟값
        lyapunov_value = 0.0        # Lyapunov 함수 값

        return ControlOutput(
            command=tau,
            mode=self._mode,
            barrier_value=barrier_value,
            lyapunov_value=lyapunov_value,
            info={"ee_position": ee_pos},
        )

    def reset(self) -> None:
        pass

    @property
    def control_mode(self) -> ControlMode:
        return self._mode
```

### 제어기 등록 및 사용

구현한 제어기를 `main.py`에서 사용하려면, `src/robot_arm_sim/main.py`의 컴포넌트 빌드 부분을 수정합니다:

```python
# src/robot_arm_sim/main.py 에서 controller 생성 부분 변경

from .controller.clbf_controller import CLBFController

# 기존: controller = PDController(dynamics)
# 변경:
controller = CLBFController(dynamics)
```

또는 config의 `controller.type` 값에 따라 분기할 수 있습니다:

```python
ctrl_type = config["controller"]["type"]
if ctrl_type == "clbf":
    controller = CLBFController(dynamics)
elif ctrl_type == "pd":
    controller = PDController(dynamics)
else:
    raise ValueError(f"Unknown controller type: {ctrl_type}")
```

### PinocchioModel에서 사용 가능한 동역학 정보

제어기 구현 시 `PinocchioModel`이 제공하는 메서드:

| 메서드                        | 반환값                | 설명                                |
|------------------------------|----------------------|-------------------------------------|
| `forward_kinematics(q)`     | `np.ndarray (3,)`   | End-effector 위치 [x, y, z]         |
| `jacobian(q)`               | `np.ndarray (6, 7)` | 기하학적 자코비안 (위치 + 회전)         |
| `mass_matrix(q)`            | `np.ndarray (7, 7)` | 관성 행렬 M(q)                       |
| `coriolis(q, dq)`           | `np.ndarray (7, 7)` | 코리올리 행렬 C(q, dq)               |
| `gravity_vector(q)`         | `np.ndarray (7,)`   | 중력 벡터 g(q)                       |
| `nonlinear_effects(q, dq)`  | `np.ndarray (7,)`   | 비선형 효과 C(q,dq)*dq + g(q)        |

모든 메서드는 7-DOF 입력을 받고 7-DOF 결과를 반환합니다 (Panda의 2개 핑거 관절은 내부적으로 처리).

### YAML 설정 파라미터

```yaml
controller:
  type: "clbf"              # 제어기 타입 식별자
  control_mode: "torque"    # "torque" 또는 "velocity"
  gains:
    kp: [100, 100, 100, 100, 50, 50, 50]   # 비례 이득 (7-DOF)
    kd: [20, 20, 20, 20, 10, 10, 10]       # 미분 이득 (7-DOF)
  clbf:
    barrier_gain: 10.0            # Barrier 함수 이득 (클수록 장애물 회피 강화)
    lyapunov_gain: 1.0            # Lyapunov 함수 이득 (클수록 목표 수렴 강화)
    barrier_radius_margin: 0.05   # 장애물 안전 마진 (m)
    alpha: 1.0                    # Class-K 함수 파라미터
```

---

## 3. 커스텀 제어기와 Config 확장 시 주의사항

새로운 제어기를 추가하면 해당 제어기 고유의 파라미터가 필요합니다. 이때 YAML config 구조를 확장해야 하며, 다음 사항을 반드시 지켜야 합니다.

### 3.1 제어기 전용 섹션을 분리하라

`controller` 아래에 제어기 타입 이름과 동일한 하위 키를 만들어 파라미터를 격리합니다. 다른 제어기의 파라미터와 섞이지 않도록 네임스페이스를 분리하세요.

```yaml
controller:
  type: "mpc"
  control_mode: "torque"
  gains:
    kp: [80, 80, 80, 80, 40, 40, 40]
    kd: [15, 15, 15, 15, 8, 8, 8]
  # 기존 CLBF 파라미터는 그대로 둬도 무방 (사용하지 않으면 무시됨)
  clbf:
    barrier_gain: 10.0
    ...
  # MPC 전용 파라미터
  mpc:
    horizon: 20
    dt_prediction: 0.01
    cost_weights:
      position: 100.0
      velocity: 1.0
      effort: 0.01
```

### 3.2 `setup()`에서 자기 섹션만 읽어라

제어기의 `setup()` 메서드에서는 자신의 전용 섹션만 참조해야 합니다. 다른 제어기 섹션의 키에 의존하면 config 파일을 전환할 때 `KeyError`가 발생합니다.

```python
def setup(self, config: dict[str, Any]) -> None:
    ctrl_cfg = config["controller"]
    # 공통 파라미터
    mode_str = ctrl_cfg.get("control_mode", "torque")
    self._mode = ControlMode.TORQUE if mode_str == "torque" else ControlMode.VELOCITY

    # 제어기 전용 파라미터: .get()으로 기본값 지정
    mpc_cfg = ctrl_cfg.get("mpc", {})
    self._horizon = mpc_cfg.get("horizon", 20)
    self._dt_pred = mpc_cfg.get("dt_prediction", 0.01)
```

### 3.3 공통 키와 전용 키를 구분하라

모든 제어기가 공유하는 키(`type`, `control_mode`, `gains`)는 `controller` 바로 아래에 둡니다. 제어기별 고유 파라미터는 반드시 전용 하위 키 아래에 둡니다.

| 위치 | 키 예시 | 용도 |
|------|--------|------|
| `controller.type` | `"pd"`, `"clbf"`, `"mpc"` | 제어기 선택 (공통) |
| `controller.control_mode` | `"torque"`, `"velocity"` | 제어 모드 (공통) |
| `controller.gains` | `kp`, `kd` | PD 이득 (공통으로 쓸 경우) |
| `controller.<type>.*` | `barrier_gain`, `horizon` | 해당 제어기 전용 |

### 3.4 기본값이 없는 키는 명시적으로 검증하라

`setup()` 진입 시, 필수 파라미터가 없으면 즉시 에러를 발생시켜 실행 중 디버깅이 어려운 상황을 방지합니다.

```python
def setup(self, config: dict[str, Any]) -> None:
    mpc_cfg = config["controller"].get("mpc")
    if mpc_cfg is None:
        raise ValueError(
            "MPC controller requires 'controller.mpc' section in config. "
            "See docs/configuration_guide.md for the required format."
        )
    if "horizon" not in mpc_cfg:
        raise ValueError("'controller.mpc.horizon' is required.")
```

### 3.5 결과 폴더에 config가 복사된다는 점을 기억하라

`ConfigManager.save_to()`가 시뮬레이션 종료 시 config YAML을 결과 폴더에 복사합니다. 따라서:

- 실험 재현을 위해, 실행에 영향을 주는 **모든 파라미터는 YAML에 기록**되어야 합니다. 코드에 하드코딩된 매직 넘버는 재현성을 깨뜨립니다.
- 결과 폴더의 config를 보고 어떤 제어기가 어떤 파라미터로 실행되었는지 완전히 파악할 수 있어야 합니다.

### 3.6 제어기별 config 파일을 분리하는 것을 권장

실험이 많아지면 `configs/` 아래에 제어기별 설정 파일을 분리하면 관리가 수월합니다.

```
configs/
├── default.yaml       # 기본 (PD)
├── clbf.yaml          # CLBF 실험용
├── mpc_fast.yaml      # MPC 빠른 예측
└── mpc_precise.yaml   # MPC 정밀 예측
```

실행 시 `--config` 플래그로 선택:

```bash
uv run python main.py --config configs/mpc_fast.yaml
```
