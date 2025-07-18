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
    Self-contained camera process with robust error handling and automatic reset.
    This process handles its own hardware reset to avoid race conditions.
    """
    if not REALSENSE_AVAILABLE:
        print("❌ RealSense not available, cannot run camera process")
        return

    print(
        f"📸 Camera process starting for serials: {primary_serial}, {wrist_serial}")

    # Initialize variables
    shm_primary = None
    shm_wrist = None
    pipeline_primary = None
    pipeline_wrist = None

    # Get default serial numbers if not provided
    if primary_serial is None or wrist_serial is None:
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            available_serials = [dev.get_info(
                rs.camera_info.serial_number) for dev in devices]
            print(f"📸 Found cameras: {available_serials}")

            if len(available_serials) >= 2:
                primary_serial = primary_serial or available_serials[0]
                wrist_serial = wrist_serial or available_serials[1]
                print(
                    f"📸 Using cameras: Primary={primary_serial}, Wrist={wrist_serial}")
            else:
                print(
                    f"❌ Need at least 2 cameras, found {len(available_serials)}")
                return
        except Exception as e:
            print(f"❌ Failed to enumerate cameras: {e}")
            return

    # Retry loop for robust connection
    max_retries = 5
    retry_count = 0

    while not stop_event.is_set() and retry_count < max_retries:
        try:
            print(
                f"📸 Attempt {retry_count + 1}/{max_retries} to start cameras...")

            # Attach to shared memory
            shm_primary = shared_memory.SharedMemory(name=shm_name_primary)
            shm_wrist = shared_memory.SharedMemory(name=shm_name_wrist)

            # Create NumPy array views
            shared_frame_primary = np.ndarray(
                shape, dtype=dtype, buffer=shm_primary.buf)
            shared_frame_wrist = np.ndarray(
                shape, dtype=dtype, buffer=shm_wrist.buf)

            # Initialize RealSense pipelines
            pipeline_primary = rs.pipeline()
            pipeline_wrist = rs.pipeline()

            config_primary = rs.config()
            config_wrist = rs.config()

            config_primary.enable_device(primary_serial)
            config_wrist.enable_device(wrist_serial)

            config_primary.enable_stream(
                rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config_wrist.enable_stream(
                rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Try to start pipelines - this is where "device busy" errors occur
            print(f"📸 Starting pipeline for camera {primary_serial}...")
            pipeline_primary.start(config_primary)

            print(f"📸 Starting pipeline for camera {wrist_serial}...")
            pipeline_wrist.start(config_wrist)

            print("✅ Both camera pipelines started successfully!")

            # Main capture loop
            frame_count = 0
            last_fps_time = time.time()

            while not stop_event.is_set():
                try:
                    # Capture frames
                    frames_primary = pipeline_primary.wait_for_frames()
                    frames_wrist = pipeline_wrist.wait_for_frames()

                    color_frame_primary = frames_primary.get_color_frame()
                    color_frame_wrist = frames_wrist.get_color_frame()

                    if color_frame_primary and color_frame_wrist:
                        # Convert to numpy arrays
                        img_primary = np.asanyarray(
                            color_frame_primary.get_data())
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

            # If we get here, stop_event was set - clean exit
            print("📸 Camera process stopping normally")
            break

        except RuntimeError as e:
            error_msg = str(e)
            if "Device or resource busy" in error_msg or "No device connected" in error_msg:
                print(
                    f"⚠️ Camera busy/disconnected error on attempt {retry_count + 1}: {e}")

                # Clean up any partially initialized resources
                try:
                    if pipeline_primary:
                        pipeline_primary.stop()
                    if pipeline_wrist:
                        pipeline_wrist.stop()
                except Exception:
                    pass
                pipeline_primary = None
                pipeline_wrist = None

                # 👇 --- SELF-CONTAINED RESET LOGIC ---
                print("🔄 Attempting hardware reset...")
                try:
                    ctx = rs.context()
                    devices = ctx.query_devices()
                    reset_count = 0

                    for dev in devices:
                        dev_serial = dev.get_info(rs.camera_info.serial_number)
                        if dev_serial == primary_serial or dev_serial == wrist_serial:
                            print(f"📸 Resetting camera {dev_serial}...")
                            dev.hardware_reset()
                            reset_count += 1

                    if reset_count > 0:
                        print(
                            f"⏳ Reset {reset_count} camera(s). Waiting 5 seconds...")
                        time.sleep(5)
                    else:
                        print("⚠️ No cameras found to reset")

                except Exception as reset_error:
                    print(f"⚠️ Reset failed: {reset_error}")
                # --- END OF RESET LOGIC ---

                retry_count += 1
                if retry_count < max_retries:
                    print(
                        f"🔄 Retrying in 2 seconds... ({retry_count}/{max_retries})")
                    time.sleep(2)
                else:
                    print(
                        f"❌ Failed to start cameras after {max_retries} attempts")
                    break

            else:
                print(f"❌ Unhandled camera error: {e}")
                break

        except Exception as e:
            print(f"❌ Critical camera process error: {e}")
            break

    # Cleanup
    print("🛑 Camera process cleanup...")
    try:
        if pipeline_primary:
            pipeline_primary.stop()
        if pipeline_wrist:
            pipeline_wrist.stop()
        if shm_primary:
            shm_primary.close()
        if shm_wrist:
            shm_wrist.close()
        print("✅ Camera cleanup complete")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")


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
