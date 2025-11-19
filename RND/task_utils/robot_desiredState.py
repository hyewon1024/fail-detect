import torch
from collections.abc import Sequence

class GripperState:
    """ 로봇 제어를 위한 그리퍼 state 정의 """
    OPEN = 1.0
    CLOSE = -1.0

class PickAndPlaceSmState:
    """ 로봇 제어를 위한  상황 state 정의 """
    REST = 0
    PREDICT = 1
    READY = 2
    PREGRASP = 3
    GRASP = 4
    CLOSE = 5
    LIFT = 6
    MOVE_TO_BIN = 7
    LOWER = 8
    RELEASE = 9
    BACK = 10
    BACK_TO_READY = 11

class PickAndPlaceSmWaitTime:
    """ 각 pick-and-place 상황 state 별 대기 시간(초) 정의 """
    REST = 0.2
    PREDICT = 0.0
    READY = 0.0
    PREGRASP = 0.2
    GRASP = 0.5
    CLOSE = 0.5
    LIFT = 0.2
    MOVE_TO_BIN = 0.2
    LOWER = 0.2
    RELEASE = 0.2
    BACK = 0.2
    BACK_TO_READY = 0.2


class PickAndPlaceSm:
    """
    로봇이 물체를 집어 옮기는(Pick-and-Place) 작업을 상태머신(State Machine)으로 구현.
    각 단계별로 End-Effector 위치와 그리퍼 상태를 지정해줌.

    0. REST: 로봇이 초기자세 상태에 있습니다.
    1. PREDICT: 파지 예측을 수행합니다.
    2. READY: 로봇이 초기자세 상태에 위치하고, 그리퍼를 CLOSE 상태로 둡니다.
    3. PREGRASP: 타겟 물체 앞쪽의 pre-grasp 자세로 이동합니다.
    4. GRASP: 엔드이펙터를 타겟 물체에 grasp 자세로 접근합니다.
    5. CLOSE: 그리퍼를 닫아 물체를 집습니다.
    6. LIFT: 물체를 들어올립니다.
    7. MOVE_TO_BIN: 물체를 목표 xy 위치(바구니)로 이동시키고, 높이도 특정 높이까지 유지합니다.
    8. LOWER: 물체를 낮은 z 위치까지 내립니다.
    9. RELEASE: 그리퍼를 열어 물체를 놓습니다.
    10. BACK: 엔드이펙터를 바구니 위의 특정 높이로 다시 이동시킵니다.
    11. BACK_TO_READY: 엔드이펙터를 원래 초기 위치로 이동시킵니다.
    """
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # state machine 파라미터 값(1)
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        # state machine 파라미터 값(2)
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # 목표 로봇 끝단(end-effector) 자세 및 그리퍼 상태
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs, 1), 0.0, device=self.device)

        # 물체 이미지를 취득하기 위한 준비 자세
        self.ready_pose = torch.tensor([[ 3.0280e-01, -5.6916e-02,  3.2400e-01, -1.4891e-10,  1.0000e+00, 8.4725e-11, -8.7813e-10]], device=self.device)  # (x, y, z, qw, qx, qy, qz)
        self.ready_pose = self.ready_pose.repeat(num_envs, 1)

        # 물체를 상자에 두기 위해 상자 위에 위치하는 자세
        self.bin_pose = torch.tensor([[ 0.2, 0.6, 0.55, 0, 0.7071, 0.7071, 0]], device=self.device)   # (x, y, z, qw, qx, qy, qz)
        self.bin_pose = self.bin_pose.repeat(num_envs, 1)

        # 물체를 안정적으로 상자에 두기 위한 낮은 자세
        self.bin_lower_pose = torch.tensor([[ 0.2, 0.6, 0.35, 0, 0.7071, 0.7071, 0]], device=self.device)   # (x, y, z, qw, qx, qy, qz)
        self.bin_lower_pose = self.bin_lower_pose.repeat(num_envs, 1)

        # Contact-GraspNet 추론 값을 담기위한 변수 선언
        self.grasp_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.pregrasp_pose = torch.zeros((self.num_envs, 7), device=self.device)

        # Gripper가 원하는 위치에 도달하지 못하는 경우, statemachine이 멈추는 것을 방지하기 위한 변수 선언
        self.stack_ee_pose = []

    # env idx 를 통한 reset 상태 실행
    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = PickAndPlaceSmState.REST
        self.sm_wait_time[env_ids] = 0.0

    ##################################### State Machine #####################################
    # 로봇의 end-effector 및 그리퍼의 목표 상태 계산
    def compute(self, ee_pose: torch.Tensor, grasp_pose: torch.Tensor, pregrasp_pose: torch.Tensor, robot_data):
        ee_pos = ee_pose[:, :3]

        # 각 environment에 반복적으로 적용
        for i in range(self.num_envs):
            state = self.sm_state[i]
            # 각 상태에 따른 로직 구현
            if state == PickAndPlaceSmState.REST:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 특정 시간 동안 대기
                if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.REST:
                    # 다음 state 로 전환 및 state 시간 초기화
                    self.sm_state[i] = PickAndPlaceSmState.PREDICT
                    self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.PREDICT:
                # 다음 state 로 전환 및 state 시간 초기화
                self.sm_state[i] = PickAndPlaceSmState.READY
                self.sm_wait_time[i] = 0.0
                
            elif state == PickAndPlaceSmState.READY:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.READY:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.PREGRASP
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.PREGRASP:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = pregrasp_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 현재 state에서의 end-effector position을 저장
                self.stack_ee_pose.append(ee_pos[i])
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.PREGRASP:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.GRASP
                        self.sm_wait_time[i] = 0.0
                # end-effector의 위치가 일정 step 이상 바뀌지 않을때, 다음 state 로 전환 및 state 시간 초기화
                else:
                    if len(self.stack_ee_pose) > 50:
                        if torch.linalg.norm(ee_pos[i] - self.stack_ee_pose[-30]) < self.position_threshold:
                            self.sm_state[i] = PickAndPlaceSmState.CLOSE
                            self.sm_wait_time[i] = 0.0
                            self.stack_ee_pose = []

            elif state == PickAndPlaceSmState.GRASP:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = grasp_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 현재 state에서의 end-effector position을 저장
                self.stack_ee_pose.append(ee_pos[i])
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.GRASP:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.CLOSE
                        self.sm_wait_time[i] = 0.0
                        self.stack_ee_pose = []
                # end-effector의 위치가 일정 step 이상 바뀌지 않을때, 다음 state 로 전환 및 state 시간 초기화
                else:
                    if len(self.stack_ee_pose) > 50:
                        if torch.linalg.norm(ee_pos[i] - self.stack_ee_pose[-30]) < self.position_threshold:
                            self.sm_state[i] = PickAndPlaceSmState.CLOSE
                            self.sm_wait_time[i] = 0.0
                            self.stack_ee_pose = []

            elif state == PickAndPlaceSmState.CLOSE:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = ee_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                # 특정 시간 동안 대기
                if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.CLOSE:
                    # 다음 state 로 전환 및 state 시간 초기화
                    self.sm_state[i] = PickAndPlaceSmState.LIFT
                    self.sm_wait_time[i] = 0.0
                    # 일정 높이로 들어 올릴 위치 설정
                    self.lift_pose = grasp_pose[i]
                    self.lift_pose[2] = self.lift_pose[2] + 0.4

            elif state == PickAndPlaceSmState.LIFT:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.lift_pose 
                self.des_gripper_state[i] = GripperState.CLOSE
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.LIFT:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.MOVE_TO_BIN
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.MOVE_TO_BIN:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.bin_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                # 현재 state에서의 end-effector position을 저장
                self.stack_ee_pose.append(ee_pos[i])
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.MOVE_TO_BIN:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.LOWER
                        self.sm_wait_time[i] = 0.0
                        self.stack_ee_pose = []
                # end-effector의 위치가 일정 step 이상 바뀌지 않을때, 다음 state 로 전환 및 state 시간 초기화
                else:
                    if len(self.stack_ee_pose) > 50:
                        if torch.linalg.norm(ee_pos[i] - self.stack_ee_pose[-30]) < self.position_threshold:
                            self.sm_state[i] = PickAndPlaceSmState.CLOSE
                            self.sm_wait_time[i] = 0.0
                            self.stack_ee_pose = []

            elif state == PickAndPlaceSmState.LOWER:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.bin_lower_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.LOWER:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.RELEASE
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.RELEASE:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.bin_lower_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.RELEASE:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.BACK
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.BACK:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.bin_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.BACK:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.BACK_TO_READY
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.BACK_TO_READY:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.BACK_TO_READY:
                        # 남은 물체를 잡기 위해, PREDICT state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.PREDICT
                        self.sm_wait_time[i] = 0.0
                        
            # state machine 단위시간 경과
            self.sm_wait_time[i] += self.dt

            actions = torch.cat([self.des_ee_pose, self.des_gripper_state], dim=-1)

        return actions