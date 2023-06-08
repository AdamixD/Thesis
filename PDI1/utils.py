import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from pandas.io.formats.style import Styler
from typing import List, Dict


def calculate_categories_distribution(dataset_path: str, categories: List) -> Dict:
    distribution = {}

    for category in categories:
        image_list = [os.path.join(dataset_path, category, file) for file in
                      os.listdir(os.path.join(dataset_path, category)) if
                      os.path.isfile(os.path.join(dataset_path, category, file))]

        num_samples = len(image_list)
        distribution[category] = num_samples

    return distribution


def calculate_categories_distribution_multiple_datasets(dataset_paths: List, categories: List) -> Dict:
    multiple_distribution = {}

    for category in categories:
        multiple_distribution[category] = 0

    for dataset_path in dataset_paths:
        distribution = calculate_categories_distribution(
            dataset_path=dataset_path,
            categories=categories
        )

        for category, sample_counts in distribution.items():
            multiple_distribution[category] = multiple_distribution[category] + distribution[category]

    return multiple_distribution


def plot_categories_distribution(dataset_paths: List, categories: List) -> None:
    distribution = calculate_categories_distribution_multiple_datasets(
        dataset_paths=dataset_paths,
        categories=categories
    )

    sample_counts = list(distribution.values())

    plt.bar(categories, sample_counts, facecolor='#ed8932')
    plt.xlabel('Klasa', fontsize=13)
    plt.ylabel('Liczba próbek', fontsize=13)
    plt.title('Rozkład liczności próbek w klasach', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_categories_distribution_table(dataset_paths: List, categories: List) -> Styler:
    distribution = calculate_categories_distribution_multiple_datasets(
        dataset_paths=dataset_paths,
        categories=categories
    )

    sample_counts = list(distribution.values())

    data = pd.DataFrame({'Klasa': categories, 'Liczba próbek': sample_counts})

    styled_table = data.style.set_table_styles([
        {'selector': 'th',
         'props': [('background-color', 'lightgray'),
                   ('color', 'black'),
                   ('font-weight', 'bold'),
                   ('border', '1px solid gray')]},
        {'selector': 'td',
         'props': [('border', '1px solid gray')]}
    ])

    return styled_table


def convert_image_to_square(image):
    height, width = image.shape[:2]
    new_size = max(height, width)
    square_image = np.zeros((new_size, new_size, 3), np.uint8)

    x_start = (new_size - width) // 2
    x_end = x_start + width
    y_start = (new_size - height) // 2
    y_end = y_start + height

    square_image[y_start:y_end, x_start:x_end] = image

    return square_image


def convert_data(source_path: str, save_path: str, categories: List, img_size: int):
    for category in categories:
        path_source = os.path.join(source_path, category)
        path_save = os.path.join(save_path, category)

        for img_name in os.listdir(path_source):
            try:
                file_path_source = os.path.join(path_source, img_name)
                file_path_save = os.path.join(path_save, img_name)

                if os.path.splitext(file_path_source)[1] in [".png", ".jpg"]:
                    img_array = cv2.imread(file_path_source)
                    img_array_square = convert_image_to_square(img_array)
                    new_array = cv2.resize(img_array_square, (img_size, img_size))
                    cv2.imwrite(file_path_save, new_array)

            except Exception as e:
                print("Error during data preparation")