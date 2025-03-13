from controller import Robot, Motor, Camera, LED
import math
import sys

# Constants
NUMBER_OF_LEDS = 8
NUMBER_OF_JOINTS = 12
NUMBER_OF_CAMERAS = 5

# Initialize robot
robot = Robot()
time_step = int(robot.getBasicTimeStep())

# Device names
motor_names = [
    "front left shoulder abduction motor", "front left shoulder rotation motor", "front left elbow motor",
    "front right shoulder abduction motor", "front right shoulder rotation motor", "front right elbow motor",
    "rear left shoulder abduction motor", "rear left shoulder rotation motor", "rear left elbow motor",
    "rear right shoulder abduction motor", "rear right shoulder rotation motor", "rear right elbow motor"
]

camera_names = [
    "left head camera", "right head camera", "left flank camera",
    "right flank camera", "rear camera"
]

led_names = [
    "left top led", "left middle up led", "left middle down led", "left bottom led",
    "right top led", "right middle up led", "right middle down led", "right bottom led"
]

# Initialize devices
motors = [robot.getDevice(name) for name in motor_names]
cameras = [robot.getDevice(name) for name in camera_names]
leds = [robot.getDevice(name) for name in led_names]

# Obtener el dispositivo GPS
gps = robot.getDevice("gps")
gps.enable(time_step)

# Enable cameras
for i in range(2):  # Only enabling the two front cameras
    cameras[i].enable(2 * time_step)

# Turn on LEDs
for led in leds:
    led.set(1)

# Initialize motors
for motor in motors:
    motor.setPosition(0.0)

def step():
    if robot.step(time_step) == -1:
        sys.exit(0)

def movement_decomposition(target, duration):
    n_steps = int(duration * 1000 / time_step)
    current_position = [motor.getTargetPosition() for motor in motors]
    step_diff = [(t - c) / n_steps for t, c in zip(target, current_position)]

    for _ in range(n_steps):
        for i, motor in enumerate(motors):
            current_position[i] += step_diff[i]
            motor.setPosition(current_position[i])
        step()

def lie_down(duration):
    target = [-0.40, -0.99, 1.59, 0.40, -0.99, 1.59, -0.40, -0.99, 1.59, 0.40, -0.99, 1.59]
    movement_decomposition(target, duration)

def stand_up(duration):
    target = [-0.1, 0.0, 0.0, 0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.1, 0.0, 0.0]
    movement_decomposition(target, duration)

def sit_down(duration):
    target = [-0.20, -0.40, -0.19, 0.20, -0.40, -0.19, -0.40, -0.90, 1.18, 0.40, -0.90, 1.18]
    movement_decomposition(target, duration)

def give_paw():
    target_stable = [-0.20, -0.30, 0.05, 0.20, -0.40, -0.19, -0.40, -0.90, 1.18, 0.49, -0.90, 0.80]
    movement_decomposition(target_stable, 4)
    
    initial_time = robot.getTime()
    while robot.getTime() - initial_time < 8:
        motors[4].setPosition(0.2 * math.sin(2 * robot.getTime()) + 0.6)  # Upperarm movement
        motors[5].setPosition(0.4 * math.sin(2 * robot.getTime()))  # Forearm movement
        step()
    
    target_sit = [-0.20, -0.40, -0.19, 0.20, -0.40, -0.19, -0.40, -0.90, 1.18, 0.40, -0.90, 1.18]
    movement_decomposition(target_sit, 4)

# Main loop
while robot.step(time_step) != -1:
    position = gps.getValues()  # Devuelve una lista [x, y, z]
    print(f"Position: x={position[0]}, y={position[1]}, z={position[2]}")
    lie_down(4.0)
    stand_up(4.0)
    sit_down(4.0)
    give_paw()
    stand_up(4.0)
    lie_down(3.0)
    stand_up(3.0)
    lie_down(2.0)
    stand_up(2.0)
    lie_down(1.0)
    stand_up(1.0)
    lie_down(0.75)
    stand_up(0.75)
    lie_down(0.5)
    stand_up(0.5)