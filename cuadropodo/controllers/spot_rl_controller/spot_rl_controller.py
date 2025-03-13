from controller import Robot, Motor, Camera, LED, Supervisor
from gymnasium  import Env
from gymnasium.spaces  import Discrete, Box
import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Device names
motor_names = [
    "front left shoulder abduction motor", "front left shoulder rotation motor", "front left elbow motor",
    "front right shoulder abduction motor", "front right shoulder rotation motor", "front right elbow motor",
    "rear left shoulder abduction motor", "rear left shoulder rotation motor", "rear left elbow motor",
    "rear right shoulder abduction motor", "rear right shoulder rotation motor", "rear right elbow motor"
]

# Definir los límites reales de cada motor
motor_limits = {
    "front left shoulder abduction motor": (-0.6, 0.499),
    "front left shoulder rotation motor": (-1.0, 1.0),
    "front left elbow motor": (-1.5, 1.5),
    "front right shoulder abduction motor": (-0.6, 0.499),
    "front right shoulder rotation motor": (-1.0, 1.0),
    "front right elbow motor": (-1.5, 1.5),
    "rear left shoulder abduction motor": (-0.6, 0.499),
    "rear left shoulder rotation motor": (-1.0, 1.0),
    "rear left elbow motor": (-1.5, 1.5),
    "rear right shoulder abduction motor": (-0.6, 0.499),
    "rear right shoulder rotation motor": (-1.0, 1.0),
    "rear right elbow motor": (-0.45, 1.5)  # Límite mínimo ajustado
}
# Convertir los límites en arrays para el espacio de acción
low_limits = np.array([motor_limits[name][0] for name in motor_names])
high_limits = np.array([motor_limits[name][1] for name in motor_names])

class RobotEnv(Env):
    def __init__(self):
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())

        # Cambiar espacio de acción a Box para recibir valores continuos por cada motor
        # Definir el nuevo espacio de acción
        self.action_space = Box(low=low_limits, high=high_limits, dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)  # x, y, z del GPS
        self.estado = None

        # Inicializar dispositivos
        self.motors = [self.robot.getDevice(name) for name in motor_names]

        # Inicializar posiciones de motores
        for motor in self.motors:
            motor.setPosition(0.0)

        # Habilitar GPS
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.time_step)

        self.old_x = 0.0

    def movement_decomposition(self, target, duration):
        n_steps = int(duration * 1000 / self.time_step)
        current_position = [motor.getTargetPosition() for motor in self.motors]
        step_diff = [(t - c) / n_steps for t, c in zip(target, current_position)]

        for _ in range(n_steps):
            for i, motor in enumerate(self.motors):
                current_position[i] += step_diff[i]
                motor.setPosition(current_position[i])
            self.robot_step()

    def robot_step(self):
        if self.robot.step(self.time_step) == -1:
            print("Simulación terminada")

    def step(self, accion):
        # Verificar que la acción sea válida
        accion = np.clip(accion, self.action_space.low, self.action_space.high)

        # Ejecutar el movimiento con la acción recibida
        self.movement_decomposition(accion, duration=1.0)

        # Actualizar el estado con la posición del GPS
        gps_position = self.gps.getValues()
        print(gps_position)
        self.estado = np.array(gps_position)

        if gps_position[0] >= self.old_x:
            reward = 1.0  # Simple recompensa por moverse
        
        else:
            reward = -1.0
        
        self.old_x = gps_position[0]

        # Sistema de recompensa básico
        terminated = False
        truncated = False
        

        return self.estado, reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reiniciar la simulación usando supervisor
        self.robot.simulationReset()

        # Reiniciar el estado del robot y el GPS después del reinicio
        self.robot.step(self.time_step)  # Ejecutar al menos un paso para aplicar el reinicio

        # Reiniciar los motores a posición inicial
        for motor in self.motors:
            motor.setPosition(0.0)

        # Reiniciar el GPS
        self.gps.enable(self.time_step)
        gps_position = self.gps.getValues()
        self.estado = np.array(gps_position)
        self.old_x = gps_position[0]

        # Devolver el estado inicial y un diccionario vacío como información adicional
        return self.estado, {}


    def render(self, mode="human"):
        pass


# Callback personalizado para recolectar recompensas acumuladas
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []  # Lista para guardar las recompensas de cada episodio
        self.current_rewards = 0  # Recompensa acumulada del episodio actual

    def _on_step(self) -> bool:
        # Incrementar la recompensa acumulada con la recompensa del paso actual
        reward = self.locals["rewards"][0]
        self.current_rewards += reward

        # Verificar si el episodio terminó
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_rewards)
            if self.verbose > 0:
                print(
                    f"Episodio terminado en paso {self.num_timesteps}: Recompensa acumulada: {self.current_rewards}"
                )
            self.current_rewards = 0  # Reiniciar recompensa acumulada para el próximo episodio
        return True



# Crear y vectorizar el entorno
env = DummyVecEnv([lambda: RobotEnv()])

# Configurar el modelo DQN
model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-5, device="cuda")


# Entrenamiento
print("Entrenando el modelo...")
reward_callback = RewardCallback(verbose=0)
model.learn(total_timesteps=500000, callback=reward_callback)

# Guardar el modelo
model.save("robot_dqn")