from controller import Supervisor
import paho.mqtt.client as mqtt
import json
import numpy as np

# MQTT Configuración
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Definir el robot supervisor
robot = Supervisor()
time_step = int(robot.getBasicTimeStep())

# 🔍 Obtener el ID del robot desde Webots
ROBOT_ID = robot.getName()
print(ROBOT_ID)
ACTION_TOPIC = f"robot/{ROBOT_ID}/action"
STATE_TOPIC = f"robot/{ROBOT_ID}/state"
RESET_TOPIC = f"robot/{ROBOT_ID}/reset"

# Configurar dispositivos
motor_names = [
    "front left shoulder abduction motor", "front left shoulder rotation motor", "front left elbow motor",
    "front right shoulder abduction motor", "front right shoulder rotation motor", "front right elbow motor",
    "rear left shoulder abduction motor", "rear left shoulder rotation motor", "rear left elbow motor",
    "rear right shoulder abduction motor", "rear right shoulder rotation motor", "rear right elbow motor"
]

# Limites de posición de los motores según Webots
motor_limits = {
    "shoulder_abduction": (-0.6, 0.499),
    "shoulder_rotation": (-1.0, 1.0),
    "elbow": (-0.45, 0.5)
}

motors = [robot.getDevice(name) for name in motor_names]
for motor in motors:
    motor.setPosition(0.0)

gps = robot.getDevice("gps")
gps.enable(time_step)

# Obtener y habilitar el IMU
imu = robot.getDevice("imu")
imu.enable(time_step)

# MQTT Client Setup (Actualizado para usar MQTT v5)
client = mqtt.Client(protocol=mqtt.MQTTv5)

# ✅ Callback actualizado para recibir mensajes
def on_message(client, userdata, message, properties=None):
    try:
        if message.topic == ACTION_TOPIC:
            data = json.loads(message.payload.decode())
            action = np.clip(np.array(data["action"]), -1.0, 1.0)

            for i, motor in enumerate(motors):
                motor_name = motor_names[i]
                if "shoulder abduction" in motor_name:
                    min_limit, max_limit = motor_limits["shoulder_abduction"]
                elif "shoulder rotation" in motor_name:
                    min_limit, max_limit = motor_limits["shoulder_rotation"]
                elif "elbow" in motor_name:
                    min_limit, max_limit = motor_limits["elbow"]
                else:
                    min_limit, max_limit = -1.0, 1.0  # Valores por defecto

                scaled_action = min_limit + (action[i] + 1) * (max_limit - min_limit) / 2
                motor.setPosition(scaled_action)

        elif message.topic == RESET_TOPIC:
            print("Reiniciando la simulación...")
            robot.simulationReset()
            robot.step(time_step)
            print("Simulación reiniciada.")

    except json.JSONDecodeError:
        print(f"Error al decodificar mensaje en tópico '{message.topic}': {message.payload.decode()}")

# ✅ Callback de conexión
def on_connect(client, userdata, flags, reasonCode, properties=None):
    print(f"[{ROBOT_ID}] Conectado al broker MQTT con código:", reasonCode)
    client.subscribe(ACTION_TOPIC)
    client.subscribe(RESET_TOPIC)

# ✅ Callback de desconexión
def on_disconnect(client, userdata, reasonCode, properties=None):
    print(f"[{ROBOT_ID}] Desconectado del broker, código:", reasonCode)

# Configuración de callbacks
client.on_message = on_message
client.on_connect = on_connect
client.on_disconnect = on_disconnect

# Conexión con el broker MQTT
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_start()

# ✅ Simulación principal con publicación del estado
while robot.step(time_step) != -1:
    orientation = imu.getRollPitchYaw()
    gps_position = gps.getValues()
    state_message = json.dumps({"gps": gps_position, 'imu': orientation})
    client.publish(STATE_TOPIC, state_message)
