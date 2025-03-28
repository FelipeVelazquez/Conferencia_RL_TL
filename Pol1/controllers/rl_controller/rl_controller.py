from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
import random
import math

# Configuraciones de Webots
from controller import Robot, Motor

robot = Robot()
timestep = int(robot.getBasicTimeStep())

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

front_sensor = robot.getDevice("FSensor")
front_sensor.enable(timestep)

back_sensor = robot.getDevice("BSensor")
back_sensor.enable(timestep)

left_sensor = robot.getDevice("LSensor")
left_sensor.enable(timestep)

right_sensor = robot.getDevice("RSensor")
right_sensor.enable(timestep)

gps = robot.getGPS("gps")
gps.enable(timestep)

goal_points = [
    [-1.18978, 1.32494],
    [1.44022, -1.69506],
    [-0.51978, -1.36506],
    [-2.06978, 0.08494],
    [1.69022, 1.92494],
]

rand_target = random.choice(goal_points)

class RobocarEnv(Env):
    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=1000, shape=(4,), dtype=np.float32)
        self.estado = None
        self.goal_x, self.goal_y = rand_target
        self.pre_x, self.pre_y, _ = gps.getValues()
        self.tiempo_prueba = 60
        self.reward = 0
        self.estado = self._get_sensors()
        self.left_sensor_val = self._sanitize_value(left_sensor.getValue())
        self.front_sensor_val = self._sanitize_value(front_sensor.getValue())
        self.right_sensor_val = self._sanitize_value(right_sensor.getValue())
        self.back_sensor_val = self._sanitize_value(back_sensor.getValue())

    def _sanitize_value(self, val):
        """Evita valores NaN o None en sensores"""
        return 1000 if val is None or math.isnan(val) else val

    def _get_sensors(self):
        """Obtiene valores de sensores y los normaliza"""
        self.left_sensor_val = self._sanitize_value(left_sensor.getValue())
        self.front_sensor_val = self._sanitize_value(front_sensor.getValue())
        self.right_sensor_val = self._sanitize_value(right_sensor.getValue())
        self.back_sensor_val = self._sanitize_value(back_sensor.getValue())
        return np.array([self.left_sensor_val, self.front_sensor_val, self.right_sensor_val, self.back_sensor_val], dtype=np.float32)

    def _execute_action(self, accion):
        """Ejecuta el movimiento del robot basado en la acción"""
        if accion == 3:  # Girar Izquierda
            if self.right_sensor_val <= 1000:
                self.reward += 1
            else:
                self.reward -= 4
            flw.setVelocity(-0.5 * MAX_SPEED)
            frw.setVelocity(0.5 * MAX_SPEED)
            brw.setVelocity(0.5 * MAX_SPEED)
            blw.setVelocity(-0.5 * MAX_SPEED)
        elif accion == 0:  # Girar Derecha
            if self.left_sensor_val <= 1000:
                self.reward += 1
            else:
                self.reward -= 4
            flw.setVelocity(0.5 * MAX_SPEED)
            frw.setVelocity(-0.5 * MAX_SPEED)
            brw.setVelocity(-0.5 * MAX_SPEED)
            blw.setVelocity(0.5 * MAX_SPEED)
        elif accion == 1:  # Avanzar
            if self.front_sensor_val >= 600:
                self.reward += 1
            else:
                self.reward -= 4
            flw.setVelocity(0.5 * MAX_SPEED)
            frw.setVelocity(0.5 * MAX_SPEED)
            brw.setVelocity(0.5 * MAX_SPEED)
            blw.setVelocity(0.5 * MAX_SPEED)
        elif accion == 2:  # Retroceder
            if self.back_sensor_val >= 600:
                self.reward += 1
            else:
                self.reward -= 4
            flw.setVelocity(-0.5 * MAX_SPEED)
            frw.setVelocity(-0.5 * MAX_SPEED)
            brw.setVelocity(-0.5 * MAX_SPEED)
            blw.setVelocity(-0.5 * MAX_SPEED)

    def step(self, accion):
        if robot.step(timestep) == -1:
            return self.estado, 0, True, {}

        self.tiempo_prueba -= 1
        done = False

        gps_val = gps.getValues()
        if gps_val is None or len(gps_val) < 2:
            print("Advertencia: gps.getValues() retornó un valor inválido")
            gps_val = [0, 0, 0]

        x_robot, y_robot, _ = gps_val
        dist_pre_x = self.goal_x - self.pre_x
        dist_act_x = self.goal_x - x_robot
        self.reward += 1 if dist_act_x < dist_pre_x else -1

        dist_pre_y = self.goal_y - self.pre_y
        dist_act_y = self.goal_y - y_robot
        self.reward += 1 if dist_act_y < dist_pre_y else -1

        self.pre_x = x_robot
        self.pre_y = y_robot
        self.estado = self._get_sensors()

        # Ejecutar la acción en los motores
        self._execute_action(accion)

        if np.isnan(self.reward) or np.isinf(self.reward):
            print(f"Advertencia: La recompensa es inválida ({self.reward}). Reiniciando a 0.")
            self.reward = 0

        if self.tiempo_prueba <= 0:
            done = True

        return self.estado, self.reward, done, {}

    def reset(self):
        flw.setVelocity(0)
        frw.setVelocity(0)
        brw.setVelocity(0)
        blw.setVelocity(0)
        self.tiempo_prueba = 60
        self.reward = 0
        self.goal_x, self.goal_y = rand_target
        self.estado = self._get_sensors()

        assert not np.any(np.isnan(self.estado)), "Error: Hay valores NaN en el estado en reset!"
        assert not np.isnan(self.reward), "Error: La recompensa es NaN en reset!"

        return self.estado

    def render(self, mode="human"):
        pass


# Crear y vectorizar el entorno
env = DummyVecEnv([lambda: RobocarEnv()])

# Definir ruta del modelo base
base_path = "D:/Simulaciones/Pol1/controllers/rl_controller"
model_path = os.path.join(base_path, "dqn_robocar_transfer")

# Cargar el modelo preentrenado
model_transfer = DQN.load("dqn_robocar_transfer", env=env)

# Ajuste del learning rate
model_transfer.learning_rate = 1e-5  
model_transfer.learn(total_timesteps=50000, reset_num_timesteps=False)

# Guardar el modelo ajustado
model_transfer.save("dqn_robocar_transfer")

# Evaluación
print("Evaluando el modelo...")
model = DQN.load("dqn_robocar_transfer")
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    print(f"Observacion: {obs}, Reward: {rewards}")
    if dones:
        break
