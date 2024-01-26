import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

from typing import Dict, List, Tuple


class ResultsAnalyzer:
    def __init__(self, categories: List[str], results_file_path: str, figure_size: Tuple[str]):
        self.categories = categories
        self.df = self.process_results_file(results_file_path=results_file_path)
        self.figure_size = figure_size

    @staticmethod
    def calculate_statistic_function_per_epoch(df: pd.DataFrame, metric: str = "f1_score", dataset_type: str = "training", statistic_function: str = "mean"):
        if dataset_type == "training":
            selected_columns = [col for col in df.columns if f"{metric}_" in col and not col.startswith('val_')]
        else:
            selected_columns = [col for col in df.columns if f"val_{metric}_" in col]

        if statistic_function == "mean":
            statistic_function_df = df[selected_columns].mean(axis=1)
            statistic_function_df.name = f"{'val_' if dataset_type == 'validation' else ''}{metric}_{statistic_function}"
            return statistic_function_df

        elif statistic_function == "std":
            statistic_function_df = df[selected_columns].std(axis=1)
            statistic_function_df.name = f"{'val_' if dataset_type == 'validation' else ''}{metric}_{statistic_function}"
            return statistic_function_df

    @staticmethod
    def concat_statistic_data(standard_data: pd.Series, balanced_data: pd.Series):
        standard_data.name = "standard_" + standard_data.name
        balanced_data.name = "balanced_" + balanced_data.name
        df = pd.concat([standard_data, balanced_data], axis=1)
        df['Epoch'] = range(1, len(df) + 1)
        return df

    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * (precision * recall) / (precision + recall)
            f1 = np.where(np.isnan(f1), 0, f1)
            return f1

    @staticmethod
    def extract_line_data(line: str) -> Dict[str, float]:
        data = re.findall(r'(\w+): (\d+\.\d+)', line)
        extracted_data = {metric: float(value) for metric, value in data}

        time_match = re.search(r'- (\d+)s', line)
        if time_match:
            epoch_time = int(time_match.group(1))
            extracted_data['epoch_time'] = epoch_time

        return extracted_data

    @classmethod
    def insert_f1_score(cls, df: pd.DataFrame, categories: List[str], dataset_types: List[str]) -> pd.DataFrame:
        for dataset_type in dataset_types:
            dataset_type_prefix = "val_" if dataset_type == 'validation' else ""

            for category in categories:
                precision_metric = f"{dataset_type_prefix}precision_{category}"
                recall_metric = f"{dataset_type_prefix}recall_{category}"
                df[f"{dataset_type_prefix}f1_score_{category}"] = cls.calculate_f1_score(
                    precision=df[precision_metric],
                    recall=df[recall_metric]
                )

            precision_metric = f"{dataset_type_prefix}precision"
            recall_metric = f"{dataset_type_prefix}recall"
            df[f"{dataset_type_prefix}f1_score"] = cls.calculate_f1_score(
                precision=df[precision_metric],
                recall=df[recall_metric]
            )

        return df

    def process_results_file(self, results_file_path: str) -> pd.DataFrame:
        results = []
        epoch = 0

        with open(results_file_path, 'r') as file:
            for line in file:
                if line.startswith('Epoch'):
                    epoch_match = re.search(r'Epoch (\d+)/', line)
                    if epoch_match:
                        epoch = int(epoch_match.group(1))

                elif 'accuracy' in line and 'val_accuracy' in line:
                    epoch_data = self.extract_line_data(line=line)
                    epoch_data['Epoch'] = epoch
                    results.append(epoch_data)

        results_df = pd.DataFrame(results)
        results_df = self.insert_f1_score(
            df=results_df,
            categories=self.categories,
            dataset_types=["training", "validation"]
        )

        return results_df

    @staticmethod
    def plot_statistic_function_comparison(df: pd.DataFrame, figure_size: Tuple[str], statistic_function: str = "mean") -> None:
        if statistic_function == "mean":
            ylabel = "Średnia"
        elif statistic_function == "f1_score":
            ylabel = r"Wynik F$_1$"
        else:
            ylabel = "Odchylenie standardowe"

        custom_labels = [f"{ylabel} (standardowa funkcja straty)", f"{ylabel} (ważona funkcja straty)"]
        # colors = plt.cm.viridis(np.linspace(0, 1, 2))

        plt.figure(figsize=figure_size)
        sns.set(style="whitegrid")

        # plt.plot(df['Epoch'], df[df.columns[0]], label=custom_labels[0], color=colors[0], linestyle='--', marker='o')
        # plt.plot(df['Epoch'], df[df.columns[1]], label=custom_labels[1], color=colors[1], linestyle='--', marker='o')

        plt.plot(df['Epoch'], df[df.columns[0]], label=custom_labels[0], linestyle='--', marker='o')
        plt.plot(df['Epoch'], df[df.columns[1]], label=custom_labels[1], linestyle='--', marker='o')

        plt.xlabel('Epoka', fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.show()

    def plot_metric_comparison(self, metric: str = "accuracy") -> None:
        min_val_loss_epoch = self.df.loc[self.df['val_loss'].idxmin(), 'Epoch']
        metric1 = metric
        metric2 = f"val_{metric}"

        if metric == "accuracy":
            ylabel = "Dokładność"
        elif metric == "precision":
            ylabel = "Precyzja"
        elif metric == "recall":
            ylabel = "Pełność"
        elif metric == "f1_score":
            ylabel = r"Wynik F$_1$"
        else:
            ylabel = "Strata"

        custom_labels = [f"{ylabel} (zbiór treningowy)", f"{ylabel} (zbiór walidacyjny)"]
        # colors = plt.cm.viridis(np.linspace(0, 1, 2))

        plt.figure(figsize=self.figure_size)
        sns.set(style="whitegrid")

        # plt.plot(self.df['Epoch'], self.df[metric1], label=custom_labels[0], color=colors[0], linestyle='--', marker='o')
        # plt.plot(self.df['Epoch'], self.df[metric2], label=custom_labels[1], color=colors[1], linestyle='--', marker='o')

        plt.plot(self.df['Epoch'], self.df[metric1], label=custom_labels[0], linestyle='--', marker='o')
        plt.plot(self.df['Epoch'], self.df[metric2], label=custom_labels[1], linestyle='--', marker='o')

        plt.axvspan(min_val_loss_epoch, self.df['Epoch'].max(), color='blue', alpha=0.05)
        plt.xlabel('Epoka', fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.show()

    def plot_categories_metric_comparison(self, metric: str = "f1_score", dataset_type: str = 'training', color_palette: str = "viridis") -> None:
        min_val_loss_epoch = self.df.loc[self.df['val_loss'].idxmin(), 'Epoch']
        dataset_type_prefix = "val_" if dataset_type == 'validation' else ""

        if metric == "f1_score":
            ylabel = r"Wynik F$_1$"
        elif metric == "precision":
            ylabel = "Precyzja"
        else:
            ylabel = "Pełność"

        plt.figure(figsize=self.figure_size)
        if color_palette == "tab10":
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.categories)+1))
        elif color_palette == "plasma":
            colors = plt.cm.plasma(np.linspace(0, 1, len(self.categories)+1))
        elif color_palette == "inferno":
            colors = plt.cm.inferno(np.linspace(0, 1, len(self.categories)+1))
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(self.categories)+1))

        plt.plot(self.df['Epoch'], self.df[f"{dataset_type_prefix}{metric}"], label="sumarycznie", color=colors[0], linewidth=3, linestyle='--', marker='o')

        i = 1

        categories = ["neutral", "fear", "sad", "happy", "surprise", "angry", "disgust"]

        names_mapping = {
            "angry": "zły",
            "disgust": "zniesmaczony",
            "fear": "przestraszony",
            "happy": "szczęśliwy",
            "neutral": "neutralny",
            "sad": "smutny",
            "surprise": "zaskoczony",
        }

        # for category in self.categories:
        for category in categories:
            category_metric = f"{dataset_type_prefix}{metric}_{category}"
            plt.plot(self.df['Epoch'], self.df[category_metric], label=names_mapping[category], color=colors[i], linestyle='--', marker='o')
            i += 1

        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.axvspan(min_val_loss_epoch, self.df['Epoch'].max(), color='blue', alpha=0.05)
        plt.xlabel('Epoka', fontsize=13, fontweight='bold')
        plt.ylabel(ylabel, fontsize=13, fontweight='bold')
        # plt.legend(fontsize=12)
        plt.legend(loc='lower right', fontsize=12)
        plt.show()

    def display_results(self, epoch: int = None):
        if epoch:
            print(self.df.loc[epoch - 1])
        else:
            print(self.df.loc[self.df['val_loss'].idxmin()])

    def display_average_epoch_training_time(self):
        time = self.df['epoch_time'].mean()
        # hours = int(time // 3600)
        # minutes = int((time % 3600) // 60)
        minutes = int(time // 60)
        seconds = round(time % 60)
        # print(f"{hours} h {minutes} min {seconds} s")
        print(f"{minutes} min {seconds} s")

    def display_total_training_time(self):
        time = self.df['epoch_time'].sum()
        hours = int(time // 3600)
        minutes = int((time % 3600) // 60)
        seconds = round(time % 60)
        print(f"{hours} h {minutes} min {seconds} s")
