import os
import shutil

from kivy.uix.screenmanager import Screen


class MainWindow(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_image = None
        self.last_image = None
        self.source_folder = None
        self.output_folder = None
        self.selected_emotions = []
        self.source_images = []

    def select_emotion(self, emotion):
        if emotion not in self.selected_emotions:
            self.selected_emotions.append(emotion)
        else:
            self.selected_emotions.remove(emotion)

    def confirm_emotions(self):
        self.save_labeled_image()

    def save_labeled_image(self):
        if len(self.selected_emotions) != 0:
            for emotion in self.selected_emotions:
                files_num = len([file for file in os.listdir(os.path.join(self.output_folder, emotion)) if
                             os.path.isfile(os.path.join(self.output_folder, emotion, file))])

                image_name = f'image_{emotion}_{files_num}.png'

                new_image = os.path.join(self.output_folder, emotion, image_name)

                shutil.copy2(self.current_image, new_image)

    def get_images_from_source_folder(self):
        file_list = [os.path.join(self.source_folder, file) for file in os.listdir(self.source_folder) if
                     os.path.isfile(os.path.join(self.source_folder, file))]

        self.source_images = sorted(file_list, key=lambda x: os.path.getmtime(x))
