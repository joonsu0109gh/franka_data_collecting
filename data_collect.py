import time
import os
import h5py
import numpy as np
from pathlib import Path
from multiprocessing import Process, Queue, Lock
import pyspacemouse
from scipy.spatial.transform import Rotation as R

# frankapy는 설치되어 있어야 합니다.
from src.frankapy.src.realrobot import PandaRealRobot

# --- 1. SpaceMouse 입력을 읽는 함수 ---
# 별도 프로세스에서 실행되어 메인 프로그램의 지연을 막습니다.
def read_spacemouse(queue, lock):
    """Continuously reads the SpaceMouse state and puts it into a queue."""
    pyspacemouse.open()
    while True:
        state = pyspacemouse.read()
        with lock:
            if queue.full():
                queue.get()  # 오래된 데이터는 버림
            queue.put(state)

# --- 2. 로깅 경로를 생성하는 함수 ---
def get_hdf5_log_path(log_root: str):
    """Creates a unique directory and returns the path for the HDF5 file."""
    # 현재 시간을 기반으로 기본 폴더 생성
    base_dir = Path(log_root) / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(base_dir, exist_ok=True)
    
    # 내부에 'episode_0001'과 같은 폴더 생성
    existing_episodes = [d for d in os.listdir(base_dir) if d.startswith('episode_')]
    next_episode_num = len(existing_episodes)
    episode_dir = base_dir / f"episode_{next_episode_num:04d}"
    os.makedirs(episode_dir, exist_ok=True)
    
    return episode_dir / "traj.hdf5"

# --- 3. 메인 데이터 수집 로직 ---
if __name__ == "__main__":
    # --- 설정 변수 ---
    # 마우스 움직임을 로봇 움직임으로 변환하는 스케일
    TRANSLATION_SCALE = 0.005 
    ROTATION_SCALE = 0.5
    # 데이터가 저장될 최상위 폴더
    DATA_ROOT_DIR = "./my_robot_data" 

    # --- 로봇 및 SpaceMouse 초기화 ---
    print("Connecting to the robot...")
    robot = PandaRealRobot()
    
    print("Opening SpaceMouse...")
    # SpaceMouse 입력을 받을 큐(Queue)와 락(Lock) 설정
    mouse_queue = Queue(maxsize=1)
    mouse_lock = Lock()
    mouse_process = Process(target=read_spacemouse, args=(mouse_queue, mouse_lock))
    mouse_process.start()

    # --- 데이터 저장을 위한 리스트 초기화 ---
    # 이 리스트들에 매 스텝의 데이터가 쌓입니다.
    trajectory_data = {
        "end_effector_pose": [],
        "joint_positions": [],
        "gripper_width": [],
        "timestamps": [],
    }
    
    # HDF5 파일 경로 생성
    hdf5_path = get_hdf5_log_path(DATA_ROOT_DIR)
    print(f"Data will be saved to: {hdf5_path}")

    # --- 그리퍼 및 로봇 초기 상태 설정 ---
    gripper_open = True
    robot.open_gripper()
    current_ee_pose = robot.get_ee_pose()

    print("Ready to collect data. Press Ctrl+C to stop and save.")
    
    try:
        while True:
            start_time_step = time.time()

            # 큐에서 SpaceMouse 상태 가져오기
            state = None
            if not mouse_queue.empty():
                with mouse_lock:
                    state = mouse_queue.get()
            
            if state is None:
                time.sleep(0.01)
                continue

            # --- SpaceMouse 입력을 로봇 목표 포즈(Action)로 변환 ---
            # 1. 이동(Translation) 계산
            delta_translation = np.array([state.x, state.y, state.z]) * TRANSLATION_SCALE
            current_ee_pose[:3, 3] += delta_translation

            # 2. 회전(Rotation) 계산 (오른쪽 버튼 누를 때만)
            if state.buttons[1]:
                delta_rotation = R.from_euler(
                    "yxz",
                    [state.roll * ROTATION_SCALE, -state.pitch * ROTATION_SCALE, -state.yaw * ROTATION_SCALE],
                    degrees=True
                )
                current_rotation = R.from_matrix(current_ee_pose[:3, :3])
                new_rotation = delta_rotation * current_rotation
                current_ee_pose[:3, :3] = new_rotation.as_matrix()

            # 3. 그리퍼 상태 변경 (왼쪽 버튼 누를 때)
            if state.buttons[0]:
                gripper_open = not gripper_open
                if gripper_open:
                    robot.open_gripper()
                else:
                    robot.close_gripper()
                time.sleep(0.5) # 그리퍼 동작 대기

            # --- 로봇 제어 및 데이터 기록 ---
            # 1. 로봇에 움직임 명령 전달
            robot.set_ee_pose(current_ee_pose, duration=0.1) # 짧은 시간 내에 도달하도록 명령

            # 2. 현재 로봇 상태(Observation) 가져오기 및 저장
            obs = robot.get_obs()
            trajectory_data["end_effector_pose"].append(obs["panda_hand_pose"])
            trajectory_data["joint_positions"].append(obs["panda_joint_positions"])
            trajectory_data["gripper_width"].append(obs["panda_gripper_width"])
            trajectory_data["timestamps"].append(time.time())
            
            # 루프 실행 주기 조절 (약 10Hz)
            time.sleep(max(0, 0.1 - (time.time() - start_time_step)))

    except KeyboardInterrupt:
        print("\nStopping data collection...")

    finally:
        # --- 프로그램 종료 시 데이터 저장 ---
        print(f"Saving data for {len(trajectory_data['timestamps'])} steps...")
        with h5py.File(hdf5_path, 'w') as f:
            for key, data_list in trajectory_data.items():
                f.create_dataset(key, data=np.array(data_list))
        
        print("Data saved successfully!")
        
        # 프로세스 및 로봇 연결 종료
        mouse_process.terminate()
        mouse_process.join()
        robot.end()
        print("Robot connection closed.")