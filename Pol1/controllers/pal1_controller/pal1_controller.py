"""pal1_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import time

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

MAX_SPEED = 6.28

arm1 = robot.getDevice('arm1')
arm1.setPosition(0.2)
arm1.setVelocity(0.5 * MAX_SPEED)


arm2 = robot.getDevice('arm2')
arm2.setPosition(0.2)
arm2.setVelocity(0.5 * MAX_SPEED)

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)
count = 0
dir = True
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    print(count)
    
    
    if count == 100:
        dir = True
    if count == 0:
        dir = False
    
    if dir == False:
        count += 1
    else:
        count -= 1
        
        
    pos = count / 100
    print(arm1.getTargetPosition())
    arm1.setPosition(pos)
    arm2.setPosition(pos)
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
