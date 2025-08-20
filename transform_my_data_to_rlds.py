import os
import numpy as np
import cv2
import argparse

def get_binary_gripper_state(gripper_width, threshold=0.07):
    """
    Convert gripper width to binary state.
    open = 1, closed = 0.
    """
    return np.array([1 if gripper_width > threshold else 0])



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate video with 3D plot from collected data.")
    parser.add_argument('--data-path', type=str, default="", help="Task data path")
    args = parser.parse_args()
    
    if not args.data_path:
        raise ValueError("Please provide the path to the data using --data-path argument.")

    # Define the root directory path and get the list of episodes
    root_directory_path = args.data_path
    get_directory_list = lambda path: [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    episode_list = sorted(get_directory_list(root_directory_path))

    target_data_directory = root_directory_path + "_mydata/train"
    os.makedirs(target_data_directory, exist_ok=True)

    for j in range(len(episode_list)):
        step_directory_list = sorted(get_directory_list(os.path.join(episode_list[j], "steps")))

        episode = []

        for i in range(len(step_directory_list)):
            primary_image_path = os.path.join(step_directory_list[i], "image_primary.jpg")
            wrist_image_path = os.path.join(step_directory_list[i], "image_wrist.jpg")
            state_data_path = os.path.join(step_directory_list[i], "other.npy")

            primary_image = cv2.imread(primary_image_path)
            wrist_image = cv2.imread(wrist_image_path)

            # save image as numpy array unit8
            primary_image = np.asarray(primary_image, dtype=np.uint8)
            wrist_image = np.asarray(wrist_image, dtype=np.uint8)

            print(f"Processing step {i} for episode {j}: {state_data_path}")

            data = np.load(state_data_path, allow_pickle=True)
            obs = data.item()["observation"]

            joint_positions = obs["joint_positions"]  # shape: (7,)
            gripper_state = get_binary_gripper_state(obs["gripper_state"])      # shape: (1,)

            state = np.concatenate([joint_positions, gripper_state], axis=0)

            ee_pose = obs["ee_pose_6d"]     # shape: (7,)

            # get next step action
            if i < len(step_directory_list) - 1:
                next_step_directory = step_directory_list[i + 1]
                next_data_path = os.path.join(next_step_directory, "other.npy")
                next_data = np.load(next_data_path, allow_pickle=True)
                next_obs = next_data.item()["observation"]
                next_ee_positions = next_obs["ee_pose_6d"]
                next_gripper_state = get_binary_gripper_state(next_obs["gripper_state"])

                delta_ee_pose = next_ee_positions - ee_pose  # shape: (7,)
                action = np.concatenate([delta_ee_pose, next_gripper_state], axis=0)  # shape: (8,)

            else:
                action = np.zeros(6)
                action = np.concatenate([action, gripper_state], axis=0)

            episode_structure_data = {
                'image': np.asarray(primary_image, dtype=np.uint8),
                'wrist_image': np.asarray(wrist_image, dtype=np.uint8),
                'state': state.astype(np.float32),
                'action': action.astype(np.float32),
                'language_instruction': data.item()["task"]['language_instruction']
            }

            episode.append(episode_structure_data)

        save_path = os.path.join(target_data_directory, f"episode_{j}.npy")
        arr = np.array(episode, dtype=object)
        np.save(save_path, arr, allow_pickle=True)
