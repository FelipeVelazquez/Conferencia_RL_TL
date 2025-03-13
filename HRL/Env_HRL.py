import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import paho.mqtt.client as mqtt
import json
import time
from stable_baselines3 import PPO  # Para cargar BalanceNet

MQTT_BROKER = "localhost"
MQTT_PORT = 1883

class MqttRobotEnv_Balance(gym.Env):
    def __init__(self, robot_id):
        super(MqttRobotEnv_Balance, self).__init__()
        self.robot_id = robot_id

        # 🔹 Observación: Solo datos del IMU (roll, pitch, yaw)
        self.observation_space = Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32)

        # 🔹 Acciones: Control de los motores responsables del equilibrio (6 motores)
        self.action_space = Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        self.state = np.zeros(6)
        self.max_steps = 60
        self.current_step = 0

        # MQTT Configuración
        self.client = mqtt.Client(protocol=mqtt.MQTTv5)
        self.client.on_message = self.on_message
        self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        self.client.loop_start()

    def on_message(self, client, userdata, message, properties=None):
        data = json.loads(message.payload.decode())
        imu_data = np.array(data["imu"])  # Solo usamos datos del IMU
        gps_data = np.array(data["gps"])
        print(gps_data, imu_data)
        self.state = np.concatenate((gps_data, imu_data))

    def step(self, action):
        # 🔹 Enviar solo las acciones de equilibrio
        action_message = json.dumps({"action": action.tolist()})
        self.client.publish(f"robot/{self.robot_id}/action", action_message)

        time.sleep(0.1)
        self.current_step += 1
        balance_reward = 0

        # 🔹 Recompensa: Penalizar inclinaciones grandes
        roll, pitch, yaw = self.state[3], self.state[4], self.state[5]
        gps_z = self.state[2]

        if gps_z < 0.4:
            balance_reward -= 10.0
        else:
            balance_reward += 1.0


        if abs(roll) > np.pi / 4 or abs(pitch) > np.pi / 4:
            balance_reward -= 10.0

        terminated = self.current_step >= self.max_steps
        return self.state, balance_reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.client.publish(f"robot/{self.robot_id}/reset", "reset")
        time.sleep(1)
        self.state = np.zeros(6)
        return self.state, {}


class MqttRobotEnv_Nav(gym.Env):
    def __init__(self, robot_id, balance_model_path):
        super(MqttRobotEnv_Nav, self).__init__()
        self.robot_id = robot_id

        # 🔹 Cargar el modelo de BalanceNet entrenado
        self.balance_model = PPO.load(balance_model_path)  # Cargar BalanceNet entrenado

        # 🔹 Observación: Solo datos del GPS (x, y, z)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # 🔹 Acciones: Control de los motores de locomoción (6 motores)
        self.action_space = Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        self.state = np.zeros(3)
        self.max_steps = 120
        self.current_step = 0

        # MQTT Configuración
        self.client = mqtt.Client(protocol=mqtt.MQTTv5)
        self.client.on_message = self.on_message
        self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        self.client.loop_start()
    
    def on_message(self, client, userdata, message, properties=None):
        data = json.loads(message.payload.decode())
        gps_data = np.array(data["gps"])  # Solo usamos datos del GPS
        imu_data = np.array(data["imu"])  # Obtener datos del IMU también
        self.state = gps_data
        self.imu_state = imu_data  # Guardar la información del IMU para balance

    def step(self, action):
        # 🔹 Obtener la acción de equilibrio de `BalanceNet`
        balance_action, _ = self.balance_model.predict(self.imu_state)  # Usar IMU como entrada

        # 🔹 Combinar la acción de navegación con la de equilibrio
        combined_action = np.concatenate((balance_action, action))  # [Balance(6) + Navegación(6)]

        # 🔹 Enviar la acción combinada al robot
        action_message = json.dumps({"action": combined_action.tolist()})
        self.client.publish(f"robot/{self.robot_id}/action", action_message)

        time.sleep(0.1)
        self.current_step += 1

        # 🔹 Recompensa: Penalizar la distancia al objetivo
        gps_x, gps_y, gps_z = self.state
        target_x, target_y = 1.0, 1.0  # Objetivo deseado
        navigation_reward = -np.sqrt((gps_x - target_x) ** 2 + (gps_y - target_y) ** 2)

        # 🔹 Recompensa por equilibrio (mantenerse estable)
        roll, pitch, yaw = self.imu_state
        balance_reward = - (abs(roll) + abs(pitch))

        # 🔹 Sumar ambas recompensas
        total_reward = navigation_reward + balance_reward

        terminated = self.current_step >= self.max_steps
        return self.state, total_reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.client.publish(f"robot/{self.robot_id}/reset", "reset")
        time.sleep(1)
        self.state = np.zeros(3)
        self.imu_state = np.zeros(3)
        return self.state, {}