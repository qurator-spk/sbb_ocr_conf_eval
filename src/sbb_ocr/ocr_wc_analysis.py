from contextlib import contextmanager
import csv
from typing import Iterator
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt  
from scipy.stats import gaussian_kde  
import requests  
import os  
import xml.etree.ElementTree as ET  
from tqdm import tqdm  

csv.field_size_limit(10**6)  # Set the CSV field size limit

def statistics(confidences):
    confidences_array = np.array(confidences)
    mean = round(np.mean(confidences_array), 3) 
    median = round(np.median(confidences_array), 3)
    variance = round(np.var(confidences_array), 3)
    standard_deviation = round(np.std(confidences_array), 3)
    
    return mean, median, variance, standard_deviation  

def plot_histogram_with_density(ax, data, bins, title, xlabel, ylabel, color, density_color, legend_loc):
    ax.hist(data, bins=bins, color=color, edgecolor="black", alpha=0.6, density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis="y", alpha=0.75)
    max_density = ax.get_ylim()[1]
    kde = gaussian_kde(data)
    x_range = np.linspace(0, 1, 100)
    density_values = kde(x_range)
    max_kde_density = np.max(density_values) 
    ax.plot(x_range, kde(x_range), color=density_color, lw=2, label="Density Plot")
    round_up_half_integer = (int(max_kde_density * 2) + (max_kde_density * 2 % 1 > 0)) / 2
    ax.set_ylim(0, round_up_half_integer)
    ax.set_yticks(np.arange(0, round_up_half_integer+0.01, 0.5))
    ax.legend(loc=legend_loc)

def plot_boxplot(ax, data, title, ylabel, box_colors):
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", linestyle="--",))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)   
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([1, 2], ["Mean", "Standard Deviation"])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis="y", alpha=0.75)

@contextmanager
def load_csv(csv_file):
    with open(csv_file, 'r') as f:
        yield csv.reader(f)

def plot_everything(csv_files : list[str], plot_file="statistics_results.jpg"):
    all_results = []
    with tqdm(total=len(csv_files)) as progbar:
        for csv_file in csv_files:
            progbar.set_description(f"Processing file: {csv_file}")
            try:
                with load_csv(csv_file) as rows:
                    for i, row in enumerate(rows):
                        if i == 0:
                            continue
                        try:
                            textline_confs = list(map(float, row[3].split(' ')))
                            word_confs = list(map(float, row[4].split(' ')))
                        except ValueError:
                            # TODO properly catch errors in the data
                            continue
                        mean_textline, median_textline, variance_textline, standard_deviation_textline = statistics(textline_confs)
                        mean_word, median_word, variance_word, standard_deviation_word = statistics(word_confs)
                        ppn_page = f'{row[0]}_{row[1]}_{row[2]}'
                        all_results.append([ppn_page, mean_word, median_word, variance_word, standard_deviation_word, mean_textline, median_textline, variance_textline, standard_deviation_textline])
            except csv.Error as e:
                print(f"CSV error: {e} in file: {csv_file}. \nIncrease the CSV field size limit!")
                break
            progbar.update(1) 
                    
    results_df = pd.DataFrame(all_results, columns=["ppn_page", "mean_word", "median_word", "variance_word", "standard_deviation_word", "mean_textline", "median_textline", "variance_textline", "standard_deviation_textline"])

    print("Statistics results:")
    print(results_df)

    # Main plotting function  
    fig, axs = plt.subplots(2, 3, figsize=(16.5, 11.0))
    
    plot_colors = {
        "word": {
            "mean": "lightblue",
            "mean_density": "blue",
            "std": "salmon",
            "std_density": "red"
        },
        "textline": {
            "mean": "lightgreen",
            "mean_density": "darkgreen",
            "std": "wheat",
            "std_density": "sienna"
        }
    }
    
    plot_histogram_with_density(axs[0, 0], results_df["mean_word"], np.arange(0, 1.1, 0.05), 
                                "Mean Word Confidence Scores", 
                                "Mean Word Confidence", "Frequency", 
                                plot_colors["word"]["mean"], plot_colors["word"]["mean_density"], legend_loc="upper left")

    plot_boxplot(axs[0, 1], [results_df["mean_word"], results_df["standard_deviation_word"]], 
                 "Mean and Standard Deviation of Word Confidence Scores", 
                 "Confidence Scores", [plot_colors["word"]["mean"], plot_colors["word"]["std"]])

    plot_histogram_with_density(axs[0, 2], results_df["standard_deviation_word"], np.arange(0, 1.1, 0.05), 
                                "Standard Deviation of Word Confidence Scores", 
                                "Standard Deviation", "Frequency", 
                                plot_colors["word"]["std"], plot_colors["word"]["std_density"], legend_loc="upper right")

    plot_histogram_with_density(axs[1, 0], results_df["mean_textline"], np.arange(0, 1.1, 0.05), 
                                "Mean Textline Confidence Scores", 
                                "Mean Textline Confidence", "Frequency", 
                                plot_colors["textline"]["mean"], plot_colors["textline"]["mean_density"], legend_loc="upper left")

    plot_boxplot(axs[1, 1], [results_df["mean_textline"], results_df["standard_deviation_textline"]], 
                 "Mean and Standard Deviation of Textline Confidence Scores", 
                 "Confidence Scores", [plot_colors["textline"]["mean"], plot_colors["textline"]["std"]])

    plot_histogram_with_density(axs[1, 2], results_df["standard_deviation_textline"], np.arange(0, 1.1, 0.05), 
                                "Standard Deviation of Textline Confidence Scores", 
                                "Standard Deviation", "Frequency", 
                                plot_colors["textline"]["std"], plot_colors["textline"]["std_density"], legend_loc="upper right")
    
    plt.tight_layout(pad=1.5)
    plt.savefig(plot_file)
    plt.show()
