import cv2
import time
import numpy as np
from multiprocessing import shared_memory

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("⚠️ pyrealsense2 not available, using dummy cameras")


def camera_process_realsense(shm_name_primary, shm_name_wrist, shape, dtype, stop_event,
                             primary_serial=None, wrist_serial=None):
    """
    Continuously captures frames from RealSense cameras and writes them to shared memory.
    This runs in a separate process and doesn't block the main control loop.
    """
    try:
        # Attach to the existing shared memory blocks
        shm_primary = shared_memory.SharedMemory(name=shm_name_primary)
        shm_wrist = shared_memory.SharedMemory(name=shm_name_wrist)

        # Create NumPy arrays backed by the shared memory buffers
        shared_frame_primary = np.ndarray(
            shape, dtype=dtype, buffer=shm_primary.buf)
        shared_frame_wrist = np.ndarray(
            shape, dtype=dtype, buffer=shm_wrist.buf)

        # Initialize RealSense pipeline
        pipeline_primary = rs.pipeline()
        pipeline_wrist = rs.pipeline()

        config_primary = rs.config()
        config_wrist = rs.config()

        if primary_serial:
            config_primary.enable_device(primary_serial)
        if wrist_serial:
            config_wrist.enable_device(wrist_serial)

        config_primary.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config_wrist.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start cameras
        pipeline_primary.start(config_primary)
        pipeline_wrist.start(config_wrist)

        print("📸 Camera processes started with shared memory")

        frame_count = 0
        last_fps_time = time.time()

        while not stop_event.is_set():
            try:
                # Capture frames (this is fast since we're in a dedicated process)
                frames_primary = pipeline_primary.wait_for_frames()
                frames_wrist = pipeline_wrist.wait_for_frames()

                color_frame_primary = frames_primary.get_color_frame()
                color_frame_wrist = frames_wrist.get_color_frame()

                if color_frame_primary and color_frame_wrist:
                    # Convert to numpy arrays
                    img_primary = np.asanyarray(color_frame_primary.get_data())
                    img_wrist = np.asanyarray(color_frame_wrist.get_data())

                    # Write directly into shared memory (atomic operation)
                    shared_frame_primary[:] = img_primary[:]
                    shared_frame_wrist[:] = img_wrist[:]

                    frame_count += 1

                    # Print FPS occasionally
                    if frame_count % 30 == 0:
                        current_time = time.time()
                        fps = 30 / (current_time - last_fps_time)
                        print(f"📸 Camera FPS: {fps:.1f}")
                        last_fps_time = current_time

                # Small sleep to prevent 100% CPU usage
                time.sleep(0.001)

            except Exception as e:
                print(f"⚠️ Camera capture error: {e}")
                time.sleep(0.01)

    except Exception as e:
        print(f"❌ Camera process error: {e}")
    finally:
        try:
            pipeline_primary.stop()
            pipeline_wrist.stop()
            shm_primary.close()
            shm_wrist.close()
            print("📸 Camera processes stopped")
        except Exception:
            pass


def dummy_camera_process(shm_name_primary, shm_name_wrist, shape, dtype, stop_event):
    """
    Dummy camera process for testing when no cameras are available.
    """
    try:
        # Attach to the existing shared memory blocks
        shm_primary = shared_memory.SharedMemory(name=shm_name_primary)
        shm_wrist = shared_memory.SharedMemory(name=shm_name_wrist)

        # Create NumPy arrays backed by the shared memory buffers
        shared_frame_primary = np.ndarray(
            shape, dtype=dtype, buffer=shm_primary.buf)
        shared_frame_wrist = np.ndarray(
            shape, dtype=dtype, buffer=shm_wrist.buf)

        print("📸 Dummy camera processes started (no real cameras)")

        frame_count = 0
        while not stop_event.is_set():
            # Generate dummy colored frames
            dummy_primary = np.full(
                shape, (0, 100, 200), dtype=dtype)  # Orange-ish
            dummy_wrist = np.full(shape, (200, 100, 0),
                                  dtype=dtype)    # Blue-ish

            # Add frame counter for visual feedback
            cv2.putText(dummy_primary, f"Primary: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(dummy_wrist, f"Wrist: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write to shared memory
            shared_frame_primary[:] = dummy_primary[:]
            shared_frame_wrist[:] = dummy_wrist[:]

            frame_count += 1
            time.sleep(1/30)  # 30 FPS dummy rate

    except Exception as e:
        print(f"❌ Dummy camera process error: {e}")
    finally:
        try:
            shm_primary.close()
            shm_wrist.close()
            print("📸 Dummy camera processes stopped")
        except Exception:
            pass
