import time
import numpy as np
from multiprocessing import shared_memory, Event, Process
import pygame
import config

COMMAND_SHAPE = (21,)  # 6 axes + 15 buttons
COMMAND_DTYPE = np.float32

class XboxState:
    def __init__(self, cmd):
        self.x = cmd[0]
        self.y = cmd[1]
        self.z = cmd[2]
        self.roll = cmd[3]
        self.pitch = cmd[4]
        self.yaw = cmd[5]
        self.buttons = [bool(cmd[6 + i]) for i in range(len(cmd) - 6)]

class XboxManager:
    def __init__(self, controller_loop_rate_hz=None):

        self.control_inverse_mode = config.CONTROL_INVERSE_MODE  # Default to inverse mode for face-to-face control

        command_size = int(np.prod(COMMAND_SHAPE) * np.dtype(COMMAND_DTYPE).itemsize)
        self.input_shm = shared_memory.SharedMemory(create=True, size=command_size)
        self.latest_command_view = np.ndarray(COMMAND_SHAPE, dtype=COMMAND_DTYPE, buffer=self.input_shm.buf)
        self.latest_command_view.fill(0.0)

        self.controller_loop_rate_hz = config.XBOX_CONTROLLER_LOOP_RATE_HZ
        self.stop_event = Event()
        self.process = None
        print("🎮 Xbox controller process started.")

        # Initialize pygame and joystick
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()


    def xbox_process(self, shm_name: str, stop_event):
        existing_shm = None
        try:
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            shared_command = np.ndarray(COMMAND_SHAPE, dtype=COMMAND_DTYPE, buffer=existing_shm.buf)

            while not stop_event.is_set():
                pygame.event.pump()  # Required to update joystick state

                # Axes mapping (you can modify based on your controller layout)
                if not self.control_inverse_mode:
                    shared_command[0] = -self.joystick.get_axis(1)  # left stick y (dy)
                    shared_command[1] = -self.joystick.get_axis(0)  # left stick x (dx)
                    shared_command[2] = ((self.joystick.get_axis(2)+1)/2 - (self.joystick.get_axis(5)+1)/2)  # LT - RT -> dz

                    shared_command[3] = self.joystick.get_axis(3)   # right stick x (rx)
                    shared_command[4] = self.joystick.get_axis(4)   # right stick y (ry)
                    shared_command[5] = -self.joystick.get_button(4) + self.joystick.get_button(5)  # (no rz by default, set to 0)
                else:
                    shared_command[0] = self.joystick.get_axis(1)  # left stick y (dy)
                    shared_command[1] = self.joystick.get_axis(0)  # left stick x (dx)
                    shared_command[2] = ((self.joystick.get_axis(2)+1)/2 - (self.joystick.get_axis(5)+1)/2)  # LT - RT -> dz
                    shared_command[3] = -self.joystick.get_axis(3)  # right stick x (rx)
                    shared_command[4] = -self.joystick.get_axis(4)  # right stick y (ry)
                    shared_command[5] = self.joystick.get_button(4) - self.joystick.get_button(5)  # (no rz by default, set to 0)

                # Buttons
                num_buttons = self.joystick.get_numbuttons()

                for i in range(num_buttons):
                    shared_command[6 + i] = float(self.joystick.get_button(i))

                time.sleep(1.0 / self.controller_loop_rate_hz)
                # print(f"button a pressed: {shared_command[6]}")  # Example for button A
                # print(f"button b pressed: {shared_command[7]}")  # Example for button B
        except Exception as e:
            raise RuntimeError(f"Xbox Process Error: {e}")
        finally:
            if existing_shm:
                existing_shm.close()
            pygame.quit()

    def start(self):
        if self.process is None:
            self.process = Process(
                target=self.xbox_process,
                args=(self.input_shm.name, self.stop_event),
                daemon=True
            )
            self.process.start()

    def get_state(self):
        try:
            command = self.latest_command_view.copy()
            return XboxState(command)
        except Exception:
            return None

    def is_alive(self):
        return self.process is not None and self.process.is_alive()

    def stop(self):
        if self.process and self.process.is_alive():
            self.stop_event.set()
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()

        try:
            self.input_shm.close()
            self.input_shm.unlink()
        except Exception:
            pass
