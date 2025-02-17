import csv
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt  
from scipy.stats import gaussian_kde  
import requests  
import os  
import xml.etree.ElementTree as ET  


def statistics(confidences):
    confidences_array = np.array(confidences)
    mean = round(np.mean(confidences_array), 3) 
    median = round(np.median(confidences_array), 3)
    variance = round(np.var(confidences_array), 3)
    standard_deviation = round(np.std(confidences_array), 3)
    
    return mean, median, variance, standard_deviation  

def plot_histogram_with_density(ax, data, bins, title, xlabel, ylabel, color, density_color):
    ax.hist(data, bins=bins, color=color, edgecolor="black", alpha=0.6, density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis="y", alpha=0.75)
    
    kde = gaussian_kde(data)
    x_range = np.linspace(0, 1, 100)
    ax.plot(x_range, kde(x_range), color=density_color, lw=2, label="Density Plot")
    ax.legend()

def plot_boxplot(ax, data, title, ylabel):
    ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([1], ["Mean Confidence"])
    ax.grid(axis="y", alpha=0.75)

def load_csv(csv_file):
    with open(csv_file, 'r') as f:
        return list(csv.reader(f))

def plot_everything(csv_file, plot_file="statistics_results.jpg"):
    all_results = []
    for row in load_csv(csv_file)[1:]:
        try:
            mean, median, variance, standard_deviation = statistics(list(map(float, row[-1].split(' '))))
        except ValueError:
            continue
        ppn_page = f'{row[0]}_{row[1]}_{row[2]}'
        all_results.append([ppn_page, mean, median, variance, standard_deviation]) 
            
    results_df = pd.DataFrame(all_results, columns=["ppn_page", "mean", "median", "variance", "standard_deviation"])

    print(results_df)

    # Main plotting function  
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Histogram and density for means  
    plot_histogram_with_density(axs[0], results_df["mean"], np.arange(0, 1.1, 0.05), 
                                "Histogram of Mean Word Confidence Scores", 
                                "Mean Word Confidence", "Frequency", 
                                "lightblue", "blue")

    # Boxplot for means  
    plot_boxplot(axs[1], results_df["mean"], 
                 "Box Plot of Mean Word Confidence Scores", 
                 "Mean Word Confidence Scores")

    # Histogram and density for standard deviations  
    plot_histogram_with_density(axs[2], results_df["standard_deviation"], np.arange(0, 1.1, 0.05), 
                                "Histogram of Standard Deviation of Word Confidence Scores", 
                                "Standard Deviation", "Frequency", 
                                "salmon", "red")

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.show()
