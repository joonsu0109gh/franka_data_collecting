import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.spatial.transform import Rotation as R
import argparse

def get_binary_gripper_state(gripper_width, threshold=0.07):
    """
    Convert gripper width to binary state.
    open = 1, closed = 0.
    """
    return 1 if gripper_width > threshold else 0

def depth_to_colormap(depth, min_valid=10, max_valid=3000):
    depth = depth.astype(np.float32).copy()
    depth = np.clip(depth, min_valid, max_valid)
    norm = (depth - min_valid) / (max_valid - min_valid)
    norm = np.clip(norm, 0, 1)
    norm = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    return color


def make_frame(primary_img, wrist_img, depth_primary_img, depth_wrist_img,
               current_index, positions, gripper_states, ee_rpy_list):
    fig = plt.figure(figsize=(16, 6))
    canvas = FigureCanvas(fig)

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axis('off')
    ax1.set_title('Primary View')
    ax1.imshow(cv2.cvtColor(primary_img, cv2.COLOR_BGR2RGB))

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.axis('off')
    ax2.set_title('Wrist View')
    ax2.imshow(cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB))

    ax3 = fig.add_subplot(2, 3, 4)
    ax3.axis('off')
    ax3.set_title('Primary Depth')
    ax3.imshow(cv2.cvtColor(depth_primary_img, cv2.COLOR_BGR2RGB))

    ax4 = fig.add_subplot(2, 3, 5)
    ax4.axis('off')
    ax4.set_title('Wrist Depth')
    ax4.imshow(cv2.cvtColor(depth_wrist_img, cv2.COLOR_BGR2RGB))

    ax5 = fig.add_subplot(1, 3, 3, projection='3d')
    ax5.view_init(elev=30, azim=20)
    ax5.set_box_aspect([1, 1, 1])
    ax5.set_title('End-Effector Trajectory')

    margin = 0.1
    x_all = np.append(positions[:, 0], 0.0)
    y_all = np.append(positions[:, 1], 0.0)
    z_all = np.append(positions[:, 2], 0.0)
    ax5.set_xlim(np.min(x_all) - margin, np.max(x_all) + margin)
    ax5.set_ylim(np.min(y_all) - margin, np.max(y_all) + margin)
    ax5.set_zlim(np.min(z_all) - margin, np.max(z_all) + margin)

    ax5.plot(positions[:current_index+1, 0],
             positions[:current_index+1, 1],
             positions[:current_index+1, 2],
             color='blue', label='Trajectory')

    x, y, z = positions[current_index]
    gripper_state = get_binary_gripper_state(gripper_states[current_index])

    rpy = ee_rpy_list[current_index]
    rot_mat = R.from_euler('xyz', rpy).as_matrix()
    ee_color = 'green' if gripper_state == 0 else 'red'
    ax5.scatter(x, y, z, color=ee_color, s=50, label='Current Pose')

    axis_len = 0.05
    ax5.quiver(x, y, z, *(rot_mat[:, 0] * axis_len), color='r')
    ax5.quiver(x, y, z, *(rot_mat[:, 1] * axis_len), color='g')
    ax5.quiver(x, y, z, *(rot_mat[:, 2] * axis_len), color='b')
    ax5.text(x, y, z, f"EE ({'Close' if gripper_state == 0 else 'Open'})", fontsize=8, color='k')

    base = np.array([0.0, 0.0, 0.0])
    base_len = 0.1
    ax5.quiver(*base, base_len, 0, 0, color='r', alpha=0.8)
    ax5.quiver(*base, 0, base_len, 0, color='g', alpha=0.8)
    ax5.quiver(*base, 0, 0, base_len, color='b', alpha=0.8)
    ax5.text(*base, 'Base', fontsize=8, color='k')

    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax5.set_zlabel("Z")
    ax5.legend(loc='upper left')

    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]

    plt.close(fig)
    return buf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video with 3D plot from collected data.")
    parser.add_argument('--episode-path', type=str, default="000000", help="Episode number in ./my_robot_data.")
    args = parser.parse_args()
    episode_path = args.episode_path
    root_directory_path = os.path.join(episode_path, "steps")
    get_directory_list = lambda path: [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    step_folders = sorted(get_directory_list(root_directory_path))
    primary_image_list = [os.path.join(folder, "image_primary.jpg") for folder in step_folders]
    wrist_image_list = [os.path.join(folder, "image_wrist.jpg") for folder in step_folders]
    primary_depth_list = [os.path.join(folder, "depth_primary.npy") for folder in step_folders]
    wrist_depth_list = [os.path.join(folder, "depth_wrist.npy") for folder in step_folders]
    state_data_list = [os.path.join(folder, "other.npy") for folder in step_folders]

    positions = []
    gripper_states = []
    ee_rpy_list = []

    try:
        for state_path in state_data_list:
            data = np.load(state_path, allow_pickle=True)
            obs = data.item()["observation"]
            positions.append(obs["ee_pose_6d"][:3])
            gripper_states.append(obs["gripper_state"])
            ee_rpy_list.append(obs["ee_pose_6d"][3:])
            
    except Exception as e:
        print(f"!!! Error processing state data: {e}")
        
    positions = np.array(positions)
    gripper_states = np.array(gripper_states)
    ee_rpy_list = np.array(ee_rpy_list)

    output_path = os.path.join(os.path.dirname(root_directory_path), "output_with_3d_plot.mp4")
    fps = 30

    first_img_primary = cv2.imread(primary_image_list[0])
    first_img_wrist = cv2.imread(wrist_image_list[0])
    first_depth_primary = np.load(primary_depth_list[0])
    first_depth_wrist = np.load(wrist_depth_list[0])
    first_depth_primary_img = depth_to_colormap(first_depth_primary)
    first_depth_wrist_img = depth_to_colormap(first_depth_wrist)

    fig_frame = make_frame(first_img_primary, first_img_wrist, first_depth_primary_img, first_depth_wrist_img,
                           0, positions, gripper_states, ee_rpy_list)
    h, w, _ = fig_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for idx in tqdm(range(len(primary_image_list))):
        img_primary = cv2.imread(primary_image_list[idx])
        img_wrist = cv2.imread(wrist_image_list[idx])
        if img_primary is None or img_wrist is None:
            continue

        depth_primary_raw = np.load(primary_depth_list[idx])
        depth_wrist_raw = np.load(wrist_depth_list[idx])
        depth_primary_img = depth_to_colormap(depth_primary_raw)
        depth_wrist_img = depth_to_colormap(depth_wrist_raw)

        frame = make_frame(img_primary, img_wrist, depth_primary_img, depth_wrist_img,
                           idx, positions, gripper_states, ee_rpy_list)
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"✅ Video saved at: {output_path}")
