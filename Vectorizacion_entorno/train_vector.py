from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from entorno_robot import MqttRobotEnv

def make_env(robot_id):
    return lambda: MqttRobotEnv(robot_id)

if __name__ == "__main__":
    # Definir los entornos de entrenamiento
    robot_ids = ["robot_1", "robot_2", "robot_3", "robot_4", "robot_5", "robot_6",
                 "robot_7", "robot_8", "robot_9","robot_10", "robot_11", "robot_12",
                 "robot_13", "robot_14", "robot_15", "robot_16", "robot_17", "robot_18",
                 "robot_19", "robot_20", "robot_21","robot_22", "robot_23", "robot_24"]
    envs = SubprocVecEnv([make_env(robot_id) for robot_id in robot_ids])

    # Entrenar el modelo con múltiples robots en paralelo
    model = PPO("MlpPolicy", envs, verbose=2, device="cuda")
    model.learn(total_timesteps=1e6, progress_bar=True)
    model.save("robot_ppo_mqtt_multi")
