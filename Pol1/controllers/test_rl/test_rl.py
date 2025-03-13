from controller import Robot, Motor
import numpy as np
from stable_baselines3 import DQN
import os

# Inicializar el robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Configuración de motores
MAX_SPEED = 6.28
flw = robot.getDevice("FLWheel")
frw = robot.getDevice("FRWheel")
brw = robot.getDevice("BRWheel")
blw = robot.getDevice("BLWheel")

flw.setPosition(float("inf"))
frw.setPosition(float("inf"))
brw.setPosition(float("inf"))
blw.setPosition(float("inf"))

flw.setVelocity(0.0)
frw.setVelocity(0.0)
brw.setVelocity(0.0)
blw.setVelocity(0.0)

# Configuración de sensores
dsf = robot.getDevice("FSensor")
dsf.enable(timestep)
bsf = robot.getDevice("BSensor")
bsf.enable(timestep)
lsf = robot.getDevice("LSensor")
lsf.enable(timestep)
rsf = robot.getDevice("RSensor")
rsf.enable(timestep)

gps = robot.getGPS("gps")
gps.enable(timestep)

goal_x, goal_y = [1.44022, -1.69506]

# Cargar el modelo preentrenado
base_path = "D:/Simulaciones/Pol1/controllers/rl_controller"
model_path = os.path.join(base_path, "dqn_robocar_transfer")
model = DQN.load(model_path)

def obtener_observacion():
    """Obtiene los valores de los sensores y devuelve la observación como un array."""
    lsf_val = lsf.getValue() 
    dsf_val = dsf.getValue() 
    rsf_val = rsf.getValue()
    bsf_val = bsf.getValue()
    return np.array([lsf_val, dsf_val, rsf_val, bsf_val], dtype=np.float32)

def ejecutar_accion(accion):
    """Ejecuta la acción seleccionada en el robot."""
    if accion == 3:  # Girar Izquierda
        flw.setVelocity(-0.5 * MAX_SPEED)
        frw.setVelocity(0.5 * MAX_SPEED)
        brw.setVelocity(0.5 * MAX_SPEED)
        blw.setVelocity(-0.5 * MAX_SPEED)
    elif accion == 0:  # Girar Derecha
        flw.setVelocity(0.5 * MAX_SPEED)
        frw.setVelocity(-0.5 * MAX_SPEED)
        brw.setVelocity(-0.5 * MAX_SPEED)
        blw.setVelocity(0.5 * MAX_SPEED)
    elif accion == 1:  # Avanzar
        flw.setVelocity(0.5 * MAX_SPEED)
        frw.setVelocity(0.5 * MAX_SPEED)
        brw.setVelocity(0.5 * MAX_SPEED)
        blw.setVelocity(0.5 * MAX_SPEED)
    elif accion == 2:  # Retroceder
        flw.setVelocity(-0.5 * MAX_SPEED)
        frw.setVelocity(-0.5 * MAX_SPEED)
        brw.setVelocity(-0.5 * MAX_SPEED)
        blw.setVelocity(-0.5 * MAX_SPEED)

# Bucle principal de control
print("Iniciando control del robot...")
while robot.step(timestep) != -1:
    obs = obtener_observacion()
    print(obs)
    accion, _ = model.predict(obs, deterministic=True)
    ejecutar_accion(accion)

print("Simulación finalizada.")
