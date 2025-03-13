from tensorflow.keras.callbacks import Callback
import os

class CustomModelCheckpoint(Callback):
    def __init__(self, checkpoint_dir):
        super(CustomModelCheckpoint, self).__init__()
        self.checkpoint_dir = checkpoint_dir

    def on_episode_end(self, episode, logs=None):
        if logs is None:
            logs = {}

        # Eliminar el checkpoint -3 si existe
        checkpoint_to_delete = os.path.join(self.checkpoint_dir, f'model_episode_{episode-3}.h5')
        if os.path.exists(checkpoint_to_delete):
            os.remove(checkpoint_to_delete)

        # Guardar los pesos del modelo después de cada episodio
        model_path = os.path.join(self.checkpoint_dir, f'model_episode_{episode}.h5')
        self.model.save_weights(model_path)


    def get_latest_checkpoint(checkpoint_dir):
        # Obtener la lista de archivos en el directorio de checkpoints
        checkpoint_files = os.listdir(checkpoint_dir)

        # Filtrar los archivos que son checkpoints (por ejemplo, aquellos con extensión .h5)
        checkpoint_files = [file for file in checkpoint_files if file.endswith('.h5')]

        # Ordenar los archivos por fecha de modificación (más reciente primero)
        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)

        # Devolver el último archivo como el último checkpoint
        if checkpoint_files:
            return os.path.join(checkpoint_dir, checkpoint_files[0])
        else:
            return None  # No se encontraron archivos de checkpoint en el directorio
