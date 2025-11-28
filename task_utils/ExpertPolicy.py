import torch
from collections.abc import Sequence

class GripperState:
    OPEN = 1.0
    CLOSE = -1.0

class PickAndPlaceSmState:
    REACH = 0      # Move to object
    PREGRASP = 1   # Approach from above
    GRASP = 2      # Move down to grasp
    CLOSE = 3      # Close gripper
    LIFT = 4       # Lift object
    MOVE_TO_BIN = 5  # Move to bin
    RELEASE = 6    # Open gripper and drop
    BACK = 7       # Move back up from bin
    BACK_TO_READY = 8  # Return to ready position

class PickAndPlaceSmWaitTime:
    REACH = 0.2
    PREGRASP = 0.2
    GRASP = 0.5
    CLOSE = 0.5
    LIFT = 0.2
    MOVE_TO_BIN = 0.2
    RELEASE = 0.2
    BACK = 0.2
    BACK_TO_READY = 0.2

class ExpertPolicy:
    """
    Observation-based state machine for pick-and-place.
    Determines phase from observations and computes appropriate actions.
    """
    
    def __init__(self, dt: float, num_envs: int, env, device: torch.device | str = "cpu", position_threshold=0.02):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.env = env
        self.position_threshold = position_threshold  # Increased to 0.02 (was 0.01)
        
        # State machine variables
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)
        
        # Desired end-effector pose and gripper state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs, 1), 0.0, device=self.device)
        
        # Observation indices (total 25 dimensions)
        self.idx_joint_pos = slice(0, 9)
        self.idx_joint_vel = slice(9, 18)
        self.idx_object_pos = slice(18, 21)
        self.idx_bin_pos = slice(21, 24)
        self.idx_gripper = 24
        
        # Constants
        self.pregrasp_offset = 0.085
        self.lift_height = 0.4
        self.bin_height_offset = 0.55
        self.bin_lower_offset = 0.35
        
        # Grasp failure detection - RELAXED THRESHOLDS
        self.grasp_check_distance = 0.10  # Increased from 0.05 to 0.10
        self.object_dropped_height = 0.03 #15  # Slightly lower threshold
        
        # Ready pose (initial position)
        self.ready_pose = torch.tensor([[3.0280e-01, -5.6916e-02, 3.2400e-01, -1.4891e-10, 1.0000e+00, 8.4725e-11, -8.7813e-10]], device=device)
        self.ready_pose = self.ready_pose.repeat(num_envs, 1)
        
        # Orientations
        self.grasp_orientation = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
        self.bin_orientation = torch.tensor([[0.0, 0.7071, 0.7071, 0.0]], device=device)
        
        # Store lift target per environment
        self.lift_target = torch.zeros((self.num_envs, 3), device=self.device)
    
    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = PickAndPlaceSmState.REACH
        self.sm_wait_time[env_ids] = 0.0
    
    def _get_ee_pose(self):
        """Get current end-effector pose from environment."""
        ee_frame_sensor = self.env.scene["ee_frame"]
        tcp_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - self.env.scene.env_origins
        tcp_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
        return torch.cat([tcp_position, tcp_orientation], dim=-1)
    
    def _is_object_in_bin(self, obj_pos_robot_frame, bin_pos_robot_frame):
        """Check if object is inside bin."""
        xy_distance = torch.norm(obj_pos_robot_frame[:2] - bin_pos_robot_frame[:2])
        return xy_distance < 0.15 and obj_pos_robot_frame[2] < 0.1
    
    def _is_object_grasped(self, ee_pos, obj_pos_w, gripper_width):
        """
        Check if object is successfully grasped.
        More lenient criteria to avoid false negatives.
        """
        gripper_closed = gripper_width < 0.02  # Slightly more lenient (was 0.015)
        ee_obj_distance = torch.linalg.norm(ee_pos - obj_pos_w)
        object_close_to_ee = ee_obj_distance < self.grasp_check_distance
        object_lifted = obj_pos_w[2] > self.object_dropped_height
        
        return gripper_closed and object_close_to_ee and object_lifted
    
    def compute(self, obs):
        """Compute actions based on observations."""
        ee_pose = self._get_ee_pose()
        ee_pos = ee_pose[:, :3]
        
        for i in range(self.num_envs):
            obs_single = obs[i]
            state = self.sm_state[i].item()
            
            # Extract observation components
            obj_pos_robot_frame = obs_single[self.idx_object_pos]
            bin_pos_robot_frame = obs_single[self.idx_bin_pos]
            gripper_width = obs_single[self.idx_gripper]
            
            # Check if object is in bin
            object_in_bin = self._is_object_in_bin(obj_pos_robot_frame, bin_pos_robot_frame)
            
            # Convert to world frame
            robot = self.env.scene["robot"]
            robot_pos_w = robot.data.root_state_w[i, :3]
            robot_quat_w = robot.data.root_state_w[i, 3:7]
            
            from isaaclab.utils.math import combine_frame_transforms
            obj_pos_w, _ = combine_frame_transforms(
                robot_pos_w.unsqueeze(0),
                robot_quat_w.unsqueeze(0),
                obj_pos_robot_frame.unsqueeze(0)
            )
            bin_pos_w, _ = combine_frame_transforms(
                robot_pos_w.unsqueeze(0),
                robot_quat_w.unsqueeze(0),
                bin_pos_robot_frame.unsqueeze(0)
            )
            
            obj_pos_w = obj_pos_w.squeeze(0)
            bin_pos_w = bin_pos_w.squeeze(0)
            
            # Check if object is grasped
            object_grasped = self._is_object_grasped(ee_pos[i], obj_pos_w, gripper_width)
            
            # ONLY check grasp failure during LIFT and MOVE_TO_BIN
            # Don't check immediately after CLOSE - give it time
            if state in [PickAndPlaceSmState.MOVE_TO_BIN]: #PickAndPlaceSmState.LIFT
                if not object_grasped:
                    # Check if object actually dropped (not just loose in gripper)
                    if obj_pos_w[2] < self.object_dropped_height:
                        # print(f"[Env {i}] Object dropped during state {state}. Restarting.")
                        self.sm_state[i] = PickAndPlaceSmState.REACH
                        self.sm_wait_time[i] = 0.0
                        continue
            
            # State machine logic
            if state == PickAndPlaceSmState.REACH:
                if object_in_bin:
                    self.sm_state[i] = PickAndPlaceSmState.BACK_TO_READY
                    self.sm_wait_time[i] = 0.0
                    continue
                
                target_pos = obj_pos_w.clone()
                target_pos[2] += self.pregrasp_offset
                self.des_ee_pose[i, :3] = target_pos
                self.des_ee_pose[i, 3:7] = self.grasp_orientation
                self.des_gripper_state[i] = GripperState.OPEN
                
                if torch.linalg.norm(ee_pos[i] - target_pos) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.REACH:
                        self.sm_state[i] = PickAndPlaceSmState.PREGRASP
                        self.sm_wait_time[i] = 0.0
            
            elif state == PickAndPlaceSmState.PREGRASP:
                target_pos = obj_pos_w.clone()
                target_pos[2] += self.pregrasp_offset
                self.des_ee_pose[i, :3] = target_pos
                self.des_ee_pose[i, 3:7] = self.grasp_orientation
                self.des_gripper_state[i] = GripperState.OPEN
                
                if torch.linalg.norm(ee_pos[i] - target_pos) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.PREGRASP:
                        self.sm_state[i] = PickAndPlaceSmState.GRASP
                        self.sm_wait_time[i] = 0.0
            
            elif state == PickAndPlaceSmState.GRASP:
                self.des_ee_pose[i, :3] = obj_pos_w
                self.des_ee_pose[i, 3:7] = self.grasp_orientation
                self.des_gripper_state[i] = GripperState.OPEN
                
                if torch.linalg.norm(ee_pos[i] - obj_pos_w) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.GRASP:
                        self.sm_state[i] = PickAndPlaceSmState.CLOSE
                        self.sm_wait_time[i] = 0.0
            
            elif state == PickAndPlaceSmState.CLOSE:
                self.des_ee_pose[i] = ee_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                
                if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.CLOSE:
                    # Always proceed to LIFT, check grasp during LIFT instead
                    self.sm_state[i] = PickAndPlaceSmState.LIFT
                    self.sm_wait_time[i] = 0.0
                    # Store lift target
                    self.lift_target[i] = obj_pos_w.clone()
                    self.lift_target[i, 2] += self.lift_height
            
            elif state == PickAndPlaceSmState.LIFT:
                self.des_ee_pose[i, :3] = self.lift_target[i]
                self.des_ee_pose[i, 3:7] = self.grasp_orientation
                self.des_gripper_state[i] = GripperState.CLOSE
                
                if torch.linalg.norm(ee_pos[i] - self.lift_target[i]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.LIFT:
                        self.sm_state[i] = PickAndPlaceSmState.MOVE_TO_BIN
                        self.sm_wait_time[i] = 0.0
            
            elif state == PickAndPlaceSmState.MOVE_TO_BIN:
                target_pos = bin_pos_w.clone()
                target_pos[2] = self.bin_height_offset
                self.des_ee_pose[i, :3] = target_pos
                self.des_ee_pose[i, 3:7] = self.bin_orientation
                self.des_gripper_state[i] = GripperState.CLOSE
                
                if torch.linalg.norm(ee_pos[i] - target_pos) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.MOVE_TO_BIN:
                        self.sm_state[i] = PickAndPlaceSmState.RELEASE
                        self.sm_wait_time[i] = 0.0
            
            elif state == PickAndPlaceSmState.RELEASE:
                target_pos = bin_pos_w.clone()
                target_pos[2] = self.bin_lower_offset
                self.des_ee_pose[i, :3] = target_pos
                self.des_ee_pose[i, 3:7] = self.bin_orientation
                # self.des_gripper_state[i] = GripperState.OPEN  # FIXED: Was commented out
                
                if torch.linalg.norm(ee_pos[i] - target_pos) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.RELEASE:
                        self.sm_state[i] = PickAndPlaceSmState.BACK
                        self.sm_wait_time[i] = 0.0
            
            elif state == PickAndPlaceSmState.BACK:
                target_pos = bin_pos_w.clone()
                target_pos[2] = self.bin_lower_offset  # FIXED: Was bin_height_offset in my previous version
                self.des_ee_pose[i, :3] = target_pos
                self.des_ee_pose[i, 3:7] = self.bin_orientation
                self.des_gripper_state[i] = GripperState.OPEN
                
                if torch.linalg.norm(ee_pos[i] - target_pos) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.BACK:
                        self.sm_state[i] = PickAndPlaceSmState.BACK_TO_READY
                        self.sm_wait_time[i] = 0.0
            
            elif state == PickAndPlaceSmState.BACK_TO_READY:
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                
                if torch.linalg.norm(ee_pos[i] - self.ready_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.BACK_TO_READY:
                        self.sm_state[i] = PickAndPlaceSmState.REACH
                        self.sm_wait_time[i] = 0.0
            
            self.sm_wait_time[i] += self.dt
        
            self.des_ee_pose[i, :3] = ee_pos + torch.clip(self.des_ee_pose[i, :3] - ee_pos, -0.1, 0.1)
        actions = torch.cat([self.des_ee_pose, self.des_gripper_state], dim=-1)
        return actions