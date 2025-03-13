from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from Env_HRL import MqttRobotEnv_Balance, MqttRobotEnv_Nav

# 🔹 Función para crear entornos de balance
def make_env_balance(robot_id):
    return lambda: MqttRobotEnv_Balance(robot_id)

# 🔹 Función para crear entornos de navegación con BalanceNet integrado
def make_env_nav(robot_id, balance_model_path):
    return lambda: MqttRobotEnv_Nav(robot_id, balance_model_path)

if __name__ == "__main__":
    robot_ids = ["robot_1", "robot_2", "robot_3", "robot_4", "robot_5", "robot_6",
                 "robot_7", "robot_8", "robot_9","robot_10", "robot_11", "robot_12",
                 "robot_13", "robot_14", "robot_15", "robot_16", "robot_17", "robot_18",
                 "robot_19", "robot_20", "robot_21","robot_22", "robot_23", "robot_24",
                 "robot_25", "robot_26", "robot_27","robot_28", "robot_29", "robot_30"]
    
    # 🔹 Entrenar BalanceNet primero
    envs_balance = SubprocVecEnv([make_env_balance(robot_id) for robot_id in robot_ids])
    
    print("🔹 Entrenando BalanceNet...")
    balance_model = PPO("MlpPolicy", envs_balance, verbose=1, device="cuda")
    balance_model.learn(total_timesteps=2e6, progress_bar=True)
    balance_model.save("BalanceNet.zip")

    envs_balance.close()  # 🔹 Cerrar entornos de balance para liberar memoria

    # 🔹 Entrenar NavigationNet con BalanceNet integrado
    balance_model_path = "BalanceNet.zip"  # Ruta del modelo de balance entrenado
    envs_nav = SubprocVecEnv([make_env_nav(robot_id, balance_model_path) for robot_id in robot_ids])

    print("🔹 Entrenando NavigationNet con BalanceNet...")
    navigation_model = PPO("MlpPolicy", envs_nav, verbose=1, device="cuda")
    navigation_model.learn(total_timesteps=2e6, progress_bar=True)
    navigation_model.save("NavigationNet.zip")

    envs_nav.close()  # 🔹 Cerrar entornos de navegación para liberar memoria
