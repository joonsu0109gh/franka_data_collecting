import pyspacemouse
import time

success = pyspacemouse.open(dof_callback=pyspacemouse.print_state, button_callback=pyspacemouse.print_buttons)
if success:
    while True:
        state = pyspacemouse.read()
        print(state)
        time.sleep(0.01)  
