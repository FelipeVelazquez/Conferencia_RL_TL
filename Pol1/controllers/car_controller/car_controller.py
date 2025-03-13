"""car_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

MAX_SPEED = 6.28

flw = robot.getDevice('FLWheel')
frw = robot.getDevice('FRWheel')
brw = robot.getDevice('BRWheel')
blw = robot.getDevice('BLWheel')

flw.setPosition(float('inf'))
flw.setVelocity(0.0)

frw.setPosition(float('inf'))
frw.setVelocity(0.0)

brw.setPosition(float('inf'))
brw.setVelocity(0.0)

blw.setPosition(float('inf'))
blw.setVelocity(0.0)

dsf = robot.getDevice('FSensor')
dsf.enable(timestep)

lsf = robot.getDevice('LSensor')
lsf.enable(timestep)

rsf = robot.getDevice('RSensor')
rsf.enable(timestep)

# Define states
FORWARD = 0
BACKWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

# Define initial state
state = FORWARD
turn_count = 0
min_dis_value = 800

# Main loop:
while robot.step(timestep) != -1:
    dsf_v = dsf.getValue()
    lsf_v = lsf.getValue()
    rsf_v = rsf.getValue()
    
    print(state)
    
    # State transitions
    if state == FORWARD:
        if dsf_v < min_dis_value:
            state = BACKWARD
            print('Obstacle detected in front, moving backward')
        elif lsf_v < min_dis_value:
            state = TURN_RIGHT
        elif rsf_v < min_dis_value:
            state = TURN_LEFT
    elif state == BACKWARD:
        if turn_count < 15:  # Adjust as needed
            state = TURN_RIGHT
            print('Turning right to avoid obstacle')
        else:
            if lsf_v < min_dis_value:
                state = TURN_RIGHT
                print('Obstacle detected on left, turning right')
            elif rsf_v < min_dis_value:
                state = TURN_LEFT
                print('Obstacle detected on right, turning left')
    elif state == TURN_LEFT or state == TURN_RIGHT:
        if turn_count >= 30:  # Adjust as needed
            state = FORWARD
            turn_count = 0
            print('Resuming forward motion')
    
    # State actions
    if state == FORWARD:
        flw.setVelocity(0.5 * MAX_SPEED)
        frw.setVelocity(0.5 * MAX_SPEED)
        brw.setVelocity(0.5 * MAX_SPEED)
        blw.setVelocity(0.5 * MAX_SPEED)
    elif state == BACKWARD:
        flw.setVelocity(-0.5 * MAX_SPEED)
        frw.setVelocity(-0.5 * MAX_SPEED)
        brw.setVelocity(-0.5 * MAX_SPEED)
        blw.setVelocity(-0.5 * MAX_SPEED)
    elif state == TURN_LEFT:
        flw.setVelocity(-0.5 * MAX_SPEED)
        frw.setVelocity(0.5 * MAX_SPEED)
        brw.setVelocity(0.5 * MAX_SPEED)
        blw.setVelocity(-0.5 * MAX_SPEED)
        turn_count += 1
    elif state == TURN_RIGHT:
        flw.setVelocity(0.5 * MAX_SPEED)
        frw.setVelocity(-0.5 * MAX_SPEED)
        brw.setVelocity(-0.5 * MAX_SPEED)
        blw.setVelocity(0.5 * MAX_SPEED)
        turn_count += 1

    #print(dsf_v, lsf_v, rsf_v)
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
