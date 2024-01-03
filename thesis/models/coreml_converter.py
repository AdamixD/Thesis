import coremltools as ct
import tensorflow as tf


class CoreMLConverter:
    def __init__(self, tf_model_path: str):
        self.tf_model = None
        self.coreml_model = None
        self.load_tensorflow_model(tf_model_path=tf_model_path)

    def load_tensorflow_model(self, tf_model_path: str):
        self.tf_model = tf.keras.models.load_model(tf_model_path)

    def convert_to_coreml(self, labels_path: str, img_size: int, scale_mode: int = 1):
        if scale_mode == 1:
            input_image = ct.ImageType(shape=(1, img_size, img_size, 3), scale=1/255.0)
        elif scale_mode == 2:
            input_image = ct.ImageType(shape=(1, img_size, img_size, 3), scale=2/255.0, bias=[-1, -1, -1])
        else:
            input_image = ct.ImageType(shape=(1, img_size, img_size, 3), scale=1.0)

        self.coreml_model = ct.convert(
            model=self.tf_model,
            inputs=[input_image],
            classifier_config=ct.ClassifierConfig(labels_path),
            convert_to="neuralnetwork"
        )

    def save_coreml_model(self, model_name: str):
        self.coreml_model.save(f"models_coreml/{model_name}.mlmodel")
