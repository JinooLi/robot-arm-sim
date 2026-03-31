import numpy as np
from numpy.typing import NDArray
from typing import Callable
from interface import Controller, State, ControlType, CLBFGenerator


class MyCLBFGenerator(CLBFGenerator):
    def __init__(
        self,
        unsafe_region_center: np.ndarray,
        unsafe_region_radius: float,
        unsafe_region_margin: float,
        k: float,
        s: float,
        Lyapunov_center: np.ndarray,
    ):
        self.unsafe_region_center = unsafe_region_center
        self.unsafe_region_radius = unsafe_region_radius
        self.unsafe_region_margin = unsafe_region_margin
        self.k = k
        self.s = s
        self.Lyapunov_center = Lyapunov_center
        print(f"Max B: {self._B(unsafe_region_center)}")

    def _sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def _dsigmoid(self, s):
        sig = self._sigmoid(s)
        return sig * (1 - sig)

    def _test(self, s):
        return np.clip(max(0.0, np.exp(s - 1) * (s - 1) + np.exp(-1)), 0, 1)

    def _dtest(self, s):
        return np.exp(s - 1) * s if s >= 0 else 0.0

    def _Circ(self, x: np.ndarray):
        return -0.5 * (
            (x - self.unsafe_region_center) @ (x - self.unsafe_region_center)
            - (self.unsafe_region_radius + self.unsafe_region_margin) ** 2
        )

    def _dCirc_dx(self, x: np.ndarray):
        return -(x - self.unsafe_region_center)

    def _B(self, x: np.ndarray):
        return self._sigmoid(self._Circ(x) * self.k * self.s) * self.k

    def _dB_dx(self, x: np.ndarray):
        return self.s * self._B(x) * (self.k - self._B(x)) * self._dCirc_dx(x)

    def _V(self, x: np.ndarray):
        return 0.5 * (x - self.Lyapunov_center) @ (x - self.Lyapunov_center)

    def _dV_dx(self, x: np.ndarray):
        return x - self.Lyapunov_center


class MyController(Controller):
    def __init__(
        self,
        clbf_generator: MyCLBFGenerator,
    ):
        super().__init__()
        self.clbf_generator = clbf_generator
        self.pre_pos = np.zeros(7)
        self.ee_target_pos = clbf_generator.Lyapunov_center
        self.pre_torques = np.zeros(7)
        self.set_control_type(ControlType.VELOCITY)
        self.pre_dq = np.zeros(7)

    def control(self, state: State, t) -> np.ndarray:
        """제어입력을 만든다.

        Args:
            state: 현재 state
            t: 현재 시뮬레이션 시간

        Returns:
            np.ndarray: 제어 입력
        """

        velo = self.velocity_control(state, t)

        return velo

    def random_input_generator(self):
        dp = np.random.uniform(-0.05, 0.05, size=self.robot_info.ctrl_joint_number)
        pos = self.pre_pos + dp
        if self.control_type == ControlType.POSITION:
            pos = np.clip(
                pos,
                self.robot_info.joint_angle_min[: self.robot_info.ctrl_joint_number],
                self.robot_info.joint_angle_max[: self.robot_info.ctrl_joint_number],
            )
        elif self.control_type == ControlType.VELOCITY:
            vel = dp * self.robot_info.control_frequency
            vel = np.clip(
                vel,
                -np.array(
                    self.robot_info.velocity_limits[: self.robot_info.ctrl_joint_number]
                ),
                np.array(
                    self.robot_info.velocity_limits[: self.robot_info.ctrl_joint_number]
                ),
            )
            pos = vel
        elif self.control_type == ControlType.TORQUE:
            pos = np.random.uniform(
                -np.array(
                    self.robot_info.torque_limits[: self.robot_info.ctrl_joint_number]
                ),
                np.array(
                    self.robot_info.torque_limits[: self.robot_info.ctrl_joint_number]
                ),
            )
        self.pre_pos = pos.copy()
        return pos

    def velocity_control(self, state: State, t) -> np.ndarray:
        cjn = self.robot_info.ctrl_joint_number
        end_effector_control = self.end_effector_control(state)
        dq = self.soft_safety_control(state, end_effector_control, alpha=2.0)

        # 속도 제한 보정
        velo_limit = 0.3 * self.robot_info.velocity_limits

        for i in range(cjn):
            if abs(dq[i]) > velo_limit[i]:
                dq *= velo_limit[i] / abs(dq[i])

        return dq

    def pinv_with_zero_removal(self, A, index, gain, tol=1e-7):
        """유사 역행렬

        Args:
            A (Ndarray): 행렬
            index (int): 중요한 관절의 index
            gain (flaot): 1보다 큰 가중치
            tol (float, optional): 이 수보다 작은 원소는 0으로 간주한다. Defaults to 1e-5.

        Returns:
            Ndarray: 유사 역행렬
        """
        # 1. 0이 아닌 유효한 행과 열의 인덱스 찾기
        # 각 행과 열의 절댓값 합이 허용 오차(tol)보다 큰지 확인
        non_zero_rows = np.where(np.sum(np.abs(A), axis=1) > tol)[0]
        non_zero_cols = np.where(np.sum(np.abs(A), axis=0) > tol)[0]

        # 행렬 전체가 0인 경우, 크기만 전치된 영행렬 반환
        if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
            return np.zeros((A.shape[1], A.shape[0]))

        # 2. 유효한 부분 행렬 추출 (치환)
        # np.ix_를 사용하여 특정 행과 열의 교차점에 있는 요소만 뽑아냄
        A_sub = A[np.ix_(non_zero_rows, non_zero_cols)]

        d = []
        for i in range(len(non_zero_cols)):
            if i == index - 1:
                d.append(gain)
            else:
                d.append(1.0)
        W = np.diag(d)
        W = W / np.linalg.det(W) ** (1 / len(d))

        # 3. 크기가 줄어든 부분 행렬의 유사 역행렬 계산
        A_sub_pinv = W @ A_sub.T @ np.linalg.inv(A_sub @ W @ A_sub.T)

        # 4. 원래 크기(A가 m x n 이면 유사 역행렬은 n x m)의 영행렬 생성 및 복원
        A_pinv = np.zeros((A.shape[1], A.shape[0]))

        # A의 열 인덱스가 A_pinv의 행 인덱스가 됨 (전치 관계)
        A_pinv[np.ix_(non_zero_cols, non_zero_rows)] = A_sub_pinv

        return A_pinv

    def end_effector_control(self, state: State):
        cjn = self.robot_info.ctrl_joint_number

        # end-effector 제어
        J = self.J_linear(state.positions[:cjn], 11)  # end-effector의 Jacobian

        J_pinv = np.linalg.pinv(J)

        end_effector_control = -J_pinv @ self.clbf_generator._dV_dx(state.ee_position)
        return end_effector_control

    def safety_control(self, state: State, original_control):
        cjn = self.robot_info.ctrl_joint_number

        # 안전 제어
        maximum_barrier_val = 0
        saved_pos = np.zeros(3)
        important_index = 0
        safety_control = np.zeros(cjn)

        for i in range(cjn):
            pos = self.get_pos_of_joint(i)
            barrier_val = self.clbf_generator._B(pos)

            if maximum_barrier_val < barrier_val:
                maximum_barrier_val = barrier_val
                important_index = i
                saved_pos = pos

            J_link: NDArray = self.J_linear(state.positions[:cjn], i)

        # end-effector도 포함함.
        if maximum_barrier_val < self.clbf_generator._B(state.ee_position):
            maximum_barrier_val = self.clbf_generator._B(state.ee_position)
            important_index = 11
            saved_pos = state.ee_position

        J_link: NDArray = self.J_linear(state.positions[:cjn], important_index)
        J_pinv_link = J_link.T @ np.linalg.inv(
            J_link @ J_link.T + 1e-6 * np.eye(3)
        )  # np.linalg.pinv(J_link) <- 이거 쓰면 기존 행렬이 거의 singular일 때 터진다.
        # J_pinv_link = self.pinv_with_zero_removal(J_link, important_index, gain=10.0)

        safety_control = -J_pinv_link @ self.clbf_generator._dB_dx(saved_pos)

        threshold = self.clbf_generator.k / 2

        # 안전 제어와 end-effector 제어를 합성
        dq = 15 * (
            15 * original_control * (1 - maximum_barrier_val / threshold)
            + safety_control * maximum_barrier_val / threshold
            # + weight @ Nullspace_control
        )

        return dq

    def soft_safety_control(self, state: State, original_control, alpha=2.0):
        cjn = self.robot_info.ctrl_joint_number
        ee_index = 11

        # softmax 계산
        w_i_denominator = []
        w_i_numerator = 0
        for i in range(cjn):
            pos = self.get_pos_of_joint(i)
            barrier_val = self.clbf_generator._B(pos)
            w_i_denominator.append(np.exp(alpha * barrier_val))
            w_i_numerator += np.exp(alpha * barrier_val)
        # end-effector도 포함함.
        barrier_val = self.clbf_generator._B(state.ee_position)
        w_i_denominator.append(np.exp(alpha * barrier_val))
        w_i_numerator += np.exp(alpha * barrier_val)
        # 각 상수와 soft max값 계산
        w_i_list = [w_i / w_i_numerator for w_i in w_i_denominator]
        soft_max = np.log(w_i_numerator) / alpha

        # 안전 제어 계산
        safety_control = np.zeros(cjn)
        for i in range(cjn):
            pos = self.get_pos_of_joint(i)
            J_i: NDArray = self.J_linear(state.positions[:cjn], i)
            safety_control += -w_i_list[i] * J_i.T @ self.clbf_generator._dB_dx(pos)
        # end-effector도 포함함.
        J_ee: NDArray = self.J_linear(state.positions[:cjn], ee_index)
        safety_control += (
            -w_i_list[cjn] * J_ee.T @ self.clbf_generator._dB_dx(state.ee_position)
        )

        threshold = self.clbf_generator.k / 2 - np.log(cjn + 1) / alpha

        dq = 10 * original_control * (1 - soft_max / threshold) + safety_control * (
            soft_max / threshold
        )

        return dq


if __name__ == "__main__":
    from simulation import RobotSim

    clbf_Gen = MyCLBFGenerator(
        unsafe_region_center=np.array([0.0, 0.0, 0.6]),
        unsafe_region_radius=0.3,
        unsafe_region_margin=0.05,
        k=10,
        s=1,
        Lyapunov_center=np.array([0.5, 0.5, 0.5]),
    )

    controller = MyController(clbf_generator=clbf_Gen)

    sim = RobotSim(
        controller=controller,
        gravity=-9.81,
        time_frequency=1000.0,
        control_frequency=100.0,
        simulation_duration=10.0,
    )

    print(sim.get_robot_info())
    sim.simulate()
    sim.save_simulation_data(name="log_traj")
    print("시뮬레이션 데이터 저장 완료.")
    sim.visualize(file_name="log_traj", fps=50)
    print("시뮬레이션 재생 완료.")
