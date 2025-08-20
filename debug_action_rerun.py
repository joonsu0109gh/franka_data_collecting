import time
import logging
import traceback
import numpy as np
from typing import Any, Dict
from scipy.spatial.transform import Rotation as R
from franky import Robot, JointMotion, ReferenceType, Affine, CartesianMotion, Gripper
import threading
from pathlib import Path
import argparse


CONTROL_LOOP_RATE_HZ = 5
GRIPPER_OPEN_WIDTH = 0.08
GRIPPER_CLOSED_WIDTH = 0.02

class RobotController:
    def __init__(self, hostname: str):
        self.robot = Robot(hostname)
        self.gripper = Gripper(hostname)
        self.robot.relative_dynamics_factor = 0.075
        self.robot.recover_from_errors()
        self.previous_gripper_state = None
        self._gripper_thread = None  # 스레드 폭주 방지용
        
    def move_home(self):
        home_pose = [-0.0026, -0.7855, 0.0011, -2.3576, 0.0038, 1.5738, 0.7780]
        self.robot.move(JointMotion(home_pose, ReferenceType.Absolute), asynchronous=True)
        self.robot.join_motion(timeout=10.0)
        self.gripper.move_async(width=GRIPPER_OPEN_WIDTH, speed=0.1)

    def _spawn_gripper_task(self, target):
        # 이전 스레드가 아직 살아 있으면 새로 안 만든다(폭주 방지)
        if self._gripper_thread is not None and self._gripper_thread.is_alive():
            return
        t = threading.Thread(target=target, daemon=True)
        t.start()
        self._gripper_thread = t

    def apply_action(self, action: Dict[str, Any]):
        # 키 누락 대비
        try:
            dpos = np.array([action['dpos_x'], action['dpos_y'], action['dpos_z']], dtype=float)
            euler = [action['drot_x'], action['drot_y'], action['drot_z']]
        except KeyError as e:
            raise KeyError(f"Missing action key: {e}")

        # as_quat()는 [x,y,z,w] 반환. franky Affine가 동일 순서를 기대하는지 확인 필요
        drot = R.from_euler("xyz", euler).as_quat()

        current_pose = self.robot.state.O_T_EE
        ee_pos = current_pose.translation
        ee_ori = current_pose.quaternion

        target_pos = ee_pos + dpos
        target_ori = R.from_quat(ee_ori) * R.from_quat(drot)
        target_ori = target_ori.as_quat()

        motion = CartesianMotion(Affine(target_pos, target_ori), ReferenceType.Absolute)
        self.robot.move(motion, asynchronous=True)

        self.robot.join_motion(timeout=10.0)

        grip_cmd = action.get("grip_command")
        if grip_cmd != self.previous_gripper_state:
            if grip_cmd == "close":
                self._spawn_gripper_task(self.gripper_close)
            elif grip_cmd == "open":
                self._spawn_gripper_task(self.gripper_open)
        self.previous_gripper_state = grip_cmd

    def gripper_open(self):
        self.gripper.move_async(width=GRIPPER_OPEN_WIDTH, speed=0.1)

    def gripper_close(self):
        # franky 버전에 따라 grasp_async 인자/이름 확인 필요
        self.gripper.grasp_async(
            width=GRIPPER_CLOSED_WIDTH, speed=0.1, force=20,
            epsilon_inner=0.1, epsilon_outer=0.1
        )

def main():
    parser = argparse.ArgumentParser(description="Generate video with 3D plot from collected data.")
    parser.add_argument('--episode-path', type=str, default="", help="Task data path (e.g., /home/rvi/panda_datacollect/my_robot_data/0004_mydata/train/episode_0.npy)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    hostname = "172.16.0.2"

    # 에피소드 로드 & 검증
    episode_path = args.episode_path

    if not Path(episode_path).exists():
        raise FileNotFoundError(episode_path)
    
    episode_data = np.load(episode_path, allow_pickle=True)
    n_steps = len(episode_data)

    try:
        robot = RobotController(hostname)
        robot.move_home()
        print("✅ Robot and gripper initialized.")

        control_period = 1.0 / CONTROL_LOOP_RATE_HZ
        next_tick = time.time()
        idx = 0
        
        while idx < n_steps:

            rec = episode_data[idx]
            action = rec['action']
            action_dict = {
                "dpos_x": float(action[0]),
                "dpos_y": float(action[1]),
                "dpos_z": float(action[2]),
                "drot_x": float(action[3]),
                "drot_y": float(action[4]),
                "drot_z": float(action[5]),
                "grip_command": "close" if float(action[6]) == 0 else "open",
            }
            
            print(f"Step {idx + 1}/{n_steps}: {action_dict}")
            try:
                robot.apply_action(action_dict)
            except:
                logging.error(f"Error applying action at step {idx}: {traceback.format_exc()}")
                pass
            idx += 1  # ★★ 반드시 증가 ★★

            # 주기 유지
            next_tick += control_period
            sleep = next_tick - time.time()
            if sleep > 0:
                # 너무 짧으면 바쁜대기, 그 외는 sleep
                if sleep < 0.0005:
                    while time.time() < next_tick:
                        pass
                else:
                    time.sleep(sleep)
            else:
                next_tick = time.time()

        print("✅ Episode playback finished.")

    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
        logging.error(traceback.format_exc())
    finally:
        try:
            if robot is not None:
                robot.gripper.move_async(width=GRIPPER_OPEN_WIDTH, speed=0.05)
        except Exception:
            pass
        print("✅ Cleanup completed. Exiting.")

if __name__ == "__main__":
    main()
