import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import paho.mqtt.client as mqtt
import json
import time


# MQTT Configuración
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
ACTION_TOPIC = "robot/action"
STATE_TOPIC = "robot/state"
RESET_TOPIC = "robot/reset"

class MqttRobotEnv(gym.Env):
    def __init__(self, robot_id):
        super(MqttRobotEnv, self).__init__()
        self.robot_id = robot_id  # ID único para cada robot

        # Definir acción y estado
        self.action_space = Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.state = np.zeros(6)
        self.done = False
        self.max_steps = 120
        self.current_step = 0

        # 🔹 Inicializar la posición previa en X
        self.prev_gps_x = 0.0

        # Configurar los tópicos según el ID
        self.action_topic = f"robot/{self.robot_id}/action"
        self.state_topic = f"robot/{self.robot_id}/state"
        self.reset_topic = f"robot/{self.robot_id}/reset"

        # Configuración MQTT
        self.client = mqtt.Client(protocol=mqtt.MQTTv5)
        self.client.on_message = self.on_message
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        self.client.loop_start()

    def on_message(self, client, userdata, message, properties=None):
        data = json.loads(message.payload.decode())
        gps_data = np.array(data["gps"])
        imu_data = np.array(data["imu"])
        self.state = np.concatenate((gps_data, imu_data))

    def on_connect(self, client, userdata, flags, reasonCode, properties=None):
        print(f"[{self.robot_id}] Conectado al broker MQTT con código: {reasonCode}")
        client.subscribe(self.state_topic)

    def on_disconnect(self, client, userdata, reasonCode, properties=None):
        print(f"[{self.robot_id}] Desconectado del broker con código: {reasonCode}")

    def step(self, action):
        action_message = json.dumps({"action": action.tolist()})
        self.client.publish(self.action_topic, action_message)
        time.sleep(0.1)
        self.current_step += 1

        gps_x, gps_z = self.state[0], self.state[2]
        roll, pitch, yaw = self.state[3], self.state[4], self.state[5]
        orientation_penalty = abs(roll) + abs(pitch)

         # 🔹 Recompensa por avanzar en el eje X
        movement_reward = 2.0 if gps_x > self.prev_gps_x else -1.0
        self.prev_gps_x = gps_x  # Actualizar la posición previa

        reward = movement_reward - 5.0 * orientation_penalty 
        if gps_z > 0.4:
            reward += 2
        else:
            reward -= 12

        if abs(roll) > np.pi / 4 or abs(pitch) > np.pi / 4:
            reward -= 10.0
            #terminated = True
            #truncated = False
            #return self.state, reward, terminated, truncated, {}

        terminated = self.current_step >= self.max_steps
        truncated = False
        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.client.publish(self.reset_topic, "reset")
        time.sleep(1)
        self.state = np.zeros(6)

        # 🔹 Reiniciar la posición previa en X
        self.prev_gps_x = 0.0

        return self.state, {}

    def render(self, mode="human"):
        pass
