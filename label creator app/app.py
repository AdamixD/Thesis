import os

from kivy.lang import Builder
from kivymd.app import MDApp

from main_window import MainWindow
from window_manager import WindowManager


class App(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.main_window = MainWindow()
        self.main_window.source_folder = "/Users/adamdabkowski/Desktop/Face Dataset/White People"
        self.main_window.output_folder = "/Users/adamdabkowski/Desktop/Face Dataset"
        self.main_window.current_image = "/Users/adamdabkowski/Desktop/Face Dataset/White People/Screenshot 2023-06-07 at 12.31.23.png"
        self.main_window.get_images_from_source_folder()

    def refresh_widgets(self):
        self.root.children[0].manager.get_screen('main_window').ids.image_label.source = self.main_window.current_image

        self.root.children[0].manager.get_screen('main_window').ids.file_name.text = os.path.basename(self.main_window.current_image)

        self.root.children[0].manager.get_screen('main_window').ids.angry_button.line_width = 0.0001
        self.root.children[0].manager.get_screen('main_window').ids.angry_button.line_color = [0, 0, 0, 1]

        self.root.children[0].manager.get_screen('main_window').ids.disgust_button.line_width = 0.0001
        self.root.children[0].manager.get_screen('main_window').ids.disgust_button.line_color = [0, 0, 0, 1]

        self.root.children[0].manager.get_screen('main_window').ids.fear_button.line_width = 0.0001
        self.root.children[0].manager.get_screen('main_window').ids.fear_button.line_color = [0, 0, 0, 1]

        self.root.children[0].manager.get_screen('main_window').ids.happy_button.line_width = 0.0001
        self.root.children[0].manager.get_screen('main_window').ids.happy_button.line_color = [0, 0, 0, 1]

        self.root.children[0].manager.get_screen('main_window').ids.neutral_button.line_width = 0.0001
        self.root.children[0].manager.get_screen('main_window').ids.neutral_button.line_color = [0, 0, 0, 1]

        self.root.children[0].manager.get_screen('main_window').ids.sad_button.line_width = 0.0001
        self.root.children[0].manager.get_screen('main_window').ids.sad_button.line_color = [0, 0, 0, 1]

        self.root.children[0].manager.get_screen('main_window').ids.surprise_button.line_width = 0.0001
        self.root.children[0].manager.get_screen('main_window').ids.surprise_button.line_color = [0, 0, 0, 1]

        if "angry" in self.main_window.selected_emotions:
            self.root.children[0].manager.get_screen('main_window').ids.angry_button.line_width = 4
            self.root.children[0].manager.get_screen('main_window').ids.angry_button.line_color = [1, 1, 1, 1]

        if "disgust" in self.main_window.selected_emotions:
            self.root.children[0].manager.get_screen('main_window').ids.disgust_button.line_width = 4
            self.root.children[0].manager.get_screen('main_window').ids.disgust_button.line_color = [1, 1, 1, 1]

        if "fear" in self.main_window.selected_emotions:
            self.root.children[0].manager.get_screen('main_window').ids.fear_button.line_width = 4
            self.root.children[0].manager.get_screen('main_window').ids.fear_button.line_color = [1, 1, 1, 1]

        if "happy" in self.main_window.selected_emotions:
            self.root.children[0].manager.get_screen('main_window').ids.happy_button.line_width = 4
            self.root.children[0].manager.get_screen('main_window').ids.happy_button.line_color = [1, 1, 1, 1]

        if "neutral" in self.main_window.selected_emotions:
            self.root.children[0].manager.get_screen('main_window').ids.neutral_button.line_width = 4
            self.root.children[0].manager.get_screen('main_window').ids.neutral_button.line_color = [1, 1, 1, 1]

        if "sad" in self.main_window.selected_emotions:
            self.root.children[0].manager.get_screen('main_window').ids.sad_button.line_width = 4
            self.root.children[0].manager.get_screen('main_window').ids.sad_button.line_color = [1, 1, 1, 1]

        if "surprise" in self.main_window.selected_emotions:
            self.root.children[0].manager.get_screen('main_window').ids.surprise_button.line_width = 4
            self.root.children[0].manager.get_screen('main_window').ids.surprise_button.line_color = [1, 1, 1, 1]

    def select_emotion(self, emotion):
        self.main_window.select_emotion(emotion)
        self.refresh_widgets()

    def confirm_emotions(self):
        self.main_window.confirm_emotions()
        self.get_current_image(self.main_window.current_image)
        self.main_window.selected_emotions = []
        self.refresh_widgets()

    def get_current_image(self, last_image):
        current_image_index = self.main_window.source_images.index(last_image) + 1
        self.main_window.current_image = self.main_window.source_images[current_image_index]

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Orange"

        kv = Builder.load_file("appearance.kv")

        return kv


if __name__ == "__main__":
    App().run()
