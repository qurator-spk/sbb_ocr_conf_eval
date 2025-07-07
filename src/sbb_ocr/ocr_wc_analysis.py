from contextlib import contextmanager
import csv
from typing import IO, Optional, Tuple
import pandas as pd  
import numpy as np 
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt  
from scipy.stats import gaussian_kde  
from tqdm import tqdm  
import json
from rich import print
import os
import subprocess
import logging
from datetime import datetime
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objects as go
import re
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

csv.field_size_limit(10**9)  # Set the CSV field size limit

def setup_logging(command):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f'log_{command}_{timestamp}.txt'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format=f'%(asctime)s:\n %(message)s\n', filemode='w')

def statistics(confidences):
    confidences_array = np.array(confidences)
    mean = round(np.mean(confidences_array), 3) 
    median = round(np.median(confidences_array), 3)
    standard_deviation = round(np.std(confidences_array), 3)
    
    return mean, median, standard_deviation  

def plot_histogram(ax, data, weights, bins, xlabel, ylabel, color, histogram_info):
    bins = np.array(bins)

    # Categorize the data into discrete bins and use right-closed intervals
    binned_data = pd.cut(data, bins=bins, right=True, include_lowest=True) # include_lowest ensures 0.0 is captured # type: ignore

    # Extract bin labels as intervals
    all_intervals = binned_data.cat.categories

    # Count values or sum weights per bin
    if weights is not None:
        df = pd.DataFrame({'bin': binned_data, 'weight': weights})
        bin_counts = df.groupby('bin', observed=False)['weight'].sum()
        
        # Error bars: sqrt(sum of squared weights per bin)
        bin_errors = df.groupby('bin', observed=False)['weight'].apply(lambda w: np.sqrt(np.sum(w**2)))
    else:
        bin_counts = binned_data.value_counts().sort_index()
        
        # Error bars: sqrt(count)
        bin_errors = np.sqrt(bin_counts)

    # Reindex to ensure all bins appear even if count is 0
    bin_counts = bin_counts.reindex(all_intervals, fill_value=0)
    bin_errors = bin_errors.reindex(all_intervals, fill_value=0)
    bin_lefts = [interval.left for interval in bin_counts.index]
    bin_widths = [interval.right - interval.left for interval in bin_counts.index]

    ax.bar(bin_lefts, bin_counts.values, width=bin_widths, align='edge',
           color=color, edgecolor='black', alpha=0.6,
           yerr=bin_errors.values, capsize=4, error_kw={'elinewidth': 1, 'ecolor': 'gray'})
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.grid(axis="y", alpha=0.75)
    ax.ticklabel_format(style="plain")
    
    # Set y-ticks to only include integer values
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if histogram_info:
        header = f"{'Weighted ' if weights is not None else ''}Histogram: {ylabel}( {xlabel} )"
        print(f"\n{header}")
        logging.info(f"\n{header}")

        for i, interval in enumerate(bin_counts.index):
            # Avoid "-0.00" in bin edge display
            left = max(0.0, interval.left)
            right = interval.right

            if i == 0:
                left_bracket = "["
            else:
                left_bracket = "("

            right_bracket = "]"

            bin_label = f"Bin {left_bracket}{left:.2f}, {right:.2f}{right_bracket}: {bin_counts[interval]} \u00B1 {int(round(bin_errors[interval]))}"
            print(bin_label)
            logging.info(bin_label)
            
def weighted_mean(data, weights):
    return np.average(data, weights=weights)
    
def weighted_std(deviations, weights):
    deviations = np.array(deviations)
    weights = np.array(weights)
    
    # Compute weighted variance
    weighted_variance = np.sum(weights * deviations**2) / np.sum(weights)
    
    # Return standard deviation
    return np.sqrt(weighted_variance)

def weighted_standard_error_of_the_mean(data, weights):
    data = np.array(data)
    weights = np.array(weights)
    
    mean = weighted_mean(data, weights)
    deviations = data - mean
    std = weighted_std(deviations, weights)
    
    # Effective sample size
    effective_n = np.sum(weights)**2 / np.sum(weights**2)
    
    return std / np.sqrt(effective_n)
    
def weighted_percentile(data, weights, percentiles):
    data = np.array(data)
    if weights is None:
        return np.percentile(data, percentiles)
    
    weights = np.array(weights)
    
    # Get the indices that sort the data array
    sorter = np.argsort(data)
    
    # Sort the data and weights according to the indices
    data, weights = data[sorter], weights[sorter]
    
    # Calculate the cumulative sum of the weights
    cumulative_weights = np.cumsum(weights)
    
    # Normalize the cumulative weights to a scale of 0 to 100
    normalized_cumsum = 100 * cumulative_weights / cumulative_weights[-1]
    
    # Interpolate the data at the percentile positions based on the normalized cumulative weights
    return np.interp(percentiles, normalized_cumsum, data)
    
def plot_density(ax, data, weights, xlabel, ylabel, density_color, stats_collector):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    
    try:
        data = np.asarray(data)
        weights = np.asarray(weights) if weights is not None else None
        is_weighted = weights is not None
        
        # Create a kernel density estimation (KDE) using the provided data and weights
        kde = gaussian_kde(data, weights=weights)
        x_range = np.linspace(0, 1, 100)
        
        # Evaluate the density values
        density_values = kde(x_range)
        ax.set_ylim(bottom=0, top=np.max(density_values) * 1.1)
        
        min_val, max_val = np.min(data), np.max(data)
        mean = weighted_mean(data, weights) if is_weighted else np.mean(data)
        std = weighted_std(data, weights) if is_weighted else np.std(data)
        q25, q50, q75 = weighted_percentile(data, weights, [25, 50, 75]) if is_weighted else np.percentile(data, [25, 50, 75])

        ax.axvline(mean, color="black", linestyle="solid", linewidth=1, label="Mean")
        ax.axvline(q25, color="black", linestyle="dashed", linewidth=1, label="Q1: 25%")
        ax.axvline(q50, color="black", linestyle="dotted", linewidth=1, label="Q2: 50% (Median)")
        ax.axvline(q75, color="black", linestyle="dashdot", linewidth=1, label="Q3: 75%")

        ax.plot(x_range, density_values, color=density_color, lw=2)
        max_density_index = np.argmax(density_values)
        max_density_x = x_range[max_density_index]

        # Set legend location based on the maximum position
        legend_loc = 'upper right' if max_density_x < 0.5 else 'upper left'
        ax.legend(loc=legend_loc)
        
        header = f"{'Weighted ' if is_weighted else ''}Density Plot: {xlabel}"
        print(f"\n{header}")
        logging.info(f"\n{header}")
        
        stats_dict = {
            'Statistic': ['Mean', 'Std Dev', 'Min', 'Q1: 25%', 'Q2: 50% (Median)', 'Q3: 75%', 'Max'],
            'Value': [mean, std, min_val, q25, q50, q75, max_val]
        }
        stats_df = pd.DataFrame(stats_dict)
        stats_collector[header] = stats_df["Value"].values
        print(stats_df.to_string(index=False))
        logging.info(stats_df.to_string(index=False))
        
        if is_weighted:
            uw_mean = np.mean(data)
            uw_std = np.std(data)
            uw_q25, uw_q50, uw_q75 = np.percentile(data, [25, 50, 75])
            uw_stats_dict = {
                'Statistic': ['Mean', 'Std Dev', 'Min', 'Q1: 25%', 'Q2: 50% (Median)', 'Q3: 75%', 'Max'],
                'Value': [uw_mean, uw_std, min_val, uw_q25, uw_q50, uw_q75, max_val]
            }
            unweighted_header = f"Unweighted Density Plot: {xlabel}"
            stats_collector[unweighted_header] = pd.DataFrame(uw_stats_dict)["Value"].values
        
    except LinAlgError as e:
        msg = (
            "Cannot plot the data!\n"
            "LinAlgError encountered while performing KDE:\n"
            f"{e}\n"
            "The data does not have enough variation in its dimensions to accurately "
            "estimate a continuous probability density function.\n"
            "Increase the number of PPNs to be filtered!\n"
        )
        logging.info(msg)
        print(msg)

    except ValueError as v:
        msg = (
            "Cannot plot the data!\n"
            "ValueError encountered while performing KDE:\n"
            f"{v}\n"
            "Increase the number of PPNs to be filtered!\n"
        )
        logging.info(msg)
        print(msg)
        
def create_plots(results_df, weights_word, weights_textline, plot_file, histogram_info, general_title):
    density_stats = {}
    _, axs = plt.subplots(2, 4, figsize=(20.0, 10.0))
    plt.suptitle(general_title, fontsize=16, fontweight='bold')
    
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
    
    bins = [0.0, 0.05] + list(np.arange(0.1, 1.0, 0.05)) + [1.0]

    plot_histogram(axs[0, 0], results_df["mean_word"], weights_word, bins, 
                   "Mean of Word Confidence Scores", 
                   "Frequency", 
                   plot_colors["word"]["mean"], histogram_info=histogram_info)
    plot_density(axs[0, 1], results_df["mean_word"], weights_word, 
                 "Mean of Word Confidence Scores", 
                 "Density", 
                 plot_colors["word"]["mean_density"], density_stats)      

    plot_histogram(axs[0, 2], results_df["standard_deviation_word"], weights_word, bins, 
                   "Standard Deviation of Word Confidence Scores", 
                   "Frequency", 
                   plot_colors["word"]["std"], histogram_info=histogram_info)
    plot_density(axs[0, 3], results_df["standard_deviation_word"], weights_word, 
                 "Standard Deviation of Word Confidence Scores", 
                 "Density", 
                 plot_colors["word"]["std_density"], density_stats)

    plot_histogram(axs[1, 0], results_df["mean_textline"], weights_textline, bins, 
                   "Mean of Textline Confidence Scores", 
                   "Frequency", 
                   plot_colors["textline"]["mean"], histogram_info=histogram_info)
    plot_density(axs[1, 1], results_df["mean_textline"], weights_textline, 
                 "Mean of Textline Confidence Scores", 
                 "Density", 
                 plot_colors["textline"]["mean_density"], density_stats)      

    plot_histogram(axs[1, 2], results_df["standard_deviation_textline"], weights_textline, bins, 
                   "Standard Deviation of Textline Confidence Scores", 
                   "Frequency", 
                   plot_colors["textline"]["std"], histogram_info=histogram_info)
    plot_density(axs[1, 3], results_df["standard_deviation_textline"], weights_textline, 
                 "Standard Deviation of Textline Confidence Scores", 
                 "Density", 
                 plot_colors["textline"]["std_density"], density_stats)

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.945, hspace=0.17)
    plt.savefig(plot_file)
    plt.close()
    
    stats_index = ['Mean', 'Std Dev', 'Min', 'Q1: 25%', 'Q2: 50% (Median)', 'Q3: 75%', 'Max']
    density_stats_df = pd.DataFrame(density_stats, index=stats_index)
    density_stats_df.to_csv("density_plot_summary_statistics.csv")

@contextmanager
def load_csv(csv_file):
    with open(csv_file, 'r') as f:
        yield csv.reader(f)
        
def load_csv_to_list(csv_file):
    with open(csv_file, 'r') as f:
        return list(csv.reader(f))
    
def create_weighted_means_dates_barplot(plot_df, results_df, groupby_col, label_col, title, filename, ha):
    word_errors = []
    textline_errors = []

    for label in plot_df[label_col]:
        group_ppns = results_df[results_df[groupby_col] == label]
        word_se = weighted_standard_error_of_the_mean(group_ppns["mean_word"], group_ppns["weight_word"]) if len(group_ppns) > 1 else 0
        textline_se = weighted_standard_error_of_the_mean(group_ppns["mean_textline"], group_ppns["weight_textline"]) if len(group_ppns) > 1 else 0
        word_errors.append(word_se)
        textline_errors.append(textline_se)

    x = np.arange(len(plot_df))
    width = 0.35
    plt.figure(figsize=(max(13, len(plot_df) * 0.5), 7))
    plt.bar(x - width / 2, plot_df['Weighted_Mean_Word'], width,
            yerr=word_errors, capsize=5, label='Weighted Mean Word', color='skyblue')
    plt.bar(x + width / 2, plot_df['Weighted_Mean_Textline'], width,
            yerr=textline_errors, capsize=5, label='Weighted Mean Textline', color='salmon')

    plt.xlabel(label_col, fontsize=13)
    plt.ylabel('Confidence Score', fontsize=13)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(x, plot_df[label_col], rotation=45, ha=ha)
    plt.tick_params(axis='x', length=10)
    plt.ylim(0, 1)
    plt.xlim(-0.5, len(plot_df) - 0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
def extract_genre_and_subgenre(x):
    parts = x.split(':', 1)
    genre = parts[0]
    if len(parts) > 1:
        subgenre = parts[1].strip()
        return genre, subgenre
    else:
        return genre, None
        
def get_sizefactor(n):
    return 0.45 if n > 200 else 0.6 if n > 150 else 1.0
    
def create_publication_count_horizontal_barplot(labels, counts, title, ylabel, filename, fontsize_scale=1.0):
    sizefactor = get_sizefactor(len(labels))
    # Descending order
    labels, counts = labels[::-1], counts[::-1]
    base_font = 20 * sizefactor * fontsize_scale

    plt.figure(figsize=(20, 30))
    bars = plt.barh(labels, counts, color=plt.cm.tab10.colors)  # type: ignore
    plt.ylabel(ylabel, fontsize=26 * sizefactor * fontsize_scale)
    plt.xlabel('Counts', fontsize=26 * sizefactor * fontsize_scale)
    plt.title(title, fontsize=30 * sizefactor * fontsize_scale, fontweight='bold')
    plt.xticks(fontsize=base_font)
    plt.yticks(fontsize=base_font)
    plt.grid(axis='x', linestyle='--', alpha=1.0)
    plt.ylim(-0.5, len(labels) - 0.5)
    
    # Add data labels next to bars
    for bar in bars:
        xval = bar.get_width()
        plt.text(xval, bar.get_y() + bar.get_height() / 2, str(int(xval)), ha='left', va='center', fontsize=base_font)

    plt.tight_layout(pad=2.0)
    plt.savefig(filename)
    plt.close()
    
def create_weighted_means_genre_and_subgenre_barplot(plot_df, label_col, title, filename, ha, word_errors, textline_errors):
    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(max(13, len(plot_df) * 0.5), 7))
    plt.bar(x - width / 2, plot_df['Weighted_Mean_Word'], width,
            yerr=word_errors, capsize=5, label='Weighted Mean Word', color='skyblue')
    plt.bar(x + width / 2, plot_df['Weighted_Mean_Textline'], width,
            yerr=textline_errors, capsize=5, label='Weighted Mean Textline', color='salmon')

    plt.xlabel(label_col, fontsize=13)
    plt.ylabel('Confidence Score', fontsize=13)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(x, plot_df[label_col], rotation=45, ha=ha)
    plt.tick_params(axis='x', length=10)
    plt.ylim(0, 1)
    plt.xlim(-0.5, len(plot_df) - 0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
def process_weighted_means(data_dict, label_name, filename_prefix, counts_dict):
    # Create DataFrame first with all data
    data_list = []
    for label, data in data_dict.items():
        if not data["mean_word"] or not data["weight_word"]:
            continue

        wm_word = weighted_mean(data["mean_word"], data["weight_word"])
        wm_textline = weighted_mean(data["mean_textline"], data["weight_textline"])
        word_se = weighted_standard_error_of_the_mean(data["mean_word"], data["weight_word"]) if len(data["mean_word"]) > 1 else 0
        textline_se = weighted_standard_error_of_the_mean(data["mean_textline"], data["weight_textline"]) if len(data["mean_textline"]) > 1 else 0

        data_list.append({
            label_name: label,
            'Weighted_Mean_Word': wm_word,
            'Weighted_Mean_Textline': wm_textline,
            'Word_Error': word_se,
            'Textline_Error': textline_se,
            'Count': counts_dict.get(label, 0)
        })

    # Create DataFrame and sort by count (descending) and then by label name (ascending)
    df = pd.DataFrame(data_list)
    df = df.sort_values(by=['Count', label_name], ascending=[False, True]).reset_index(drop=True)

    df_save = df[[label_name, 'Weighted_Mean_Word', 'Weighted_Mean_Textline']].copy()
    df_save.to_csv(f"{filename_prefix}_weighted_mean_scores.csv", index=False)

    create_weighted_means_genre_and_subgenre_barplot(
        plot_df=df,
        label_col=label_name,
        title=f'{label_name}-based Weighted Means of Word and Textline Confidence Scores',
        filename=f"{filename_prefix}_weighted_mean_scores.png",
        ha='right',
        word_errors=df['Word_Error'].tolist(),
        textline_errors=df['Textline_Error'].tolist()
    )
                
def genre_evaluation(metadata_df, results_df, use_threshold=False):
    matching_ppn_mods = set(results_df["ppn"].unique())
    filtered_genres = metadata_df[metadata_df["PPN"].isin(matching_ppn_mods)]

    # Create dicts for fast access
    ppn_to_genres_raw = filtered_genres.groupby("PPN")["genre-aad"].apply(list).to_dict()
    
    # Determine aggregation mode
    is_ppn_page_mode = "ppn_page" in results_df.columns
    if not is_ppn_page_mode:
        ppn_to_results = results_df.groupby("ppn").first().to_dict(orient="index")

    genre_weighted_data = {}
    genre_counts = {}
    subgenre_weighted_data = {}
    subgenre_counts = {}
    genre_to_subgenre_counts = {}
    count_multiple_genres = 0
    count_single_genres = 0

    for ppn in matching_ppn_mods:
        current_genres_raw_list = ppn_to_genres_raw.get(ppn, [])
        
        # Get results based on aggregation mode
        if is_ppn_page_mode:
            ppn_results = results_df[results_df["ppn"] == ppn]
        else:
            ppn_results = pd.DataFrame([ppn_to_results.get(ppn)]) if ppn_to_results.get(ppn) is not None else pd.DataFrame()
            
        if ppn_results.empty:
            continue

        counted_genres = set()

        for genre_raw in current_genres_raw_list:
            if not genre_raw:
                continue
            genres_json = genre_raw.replace('{', '[').replace('}', ']').replace("'", '"')

            try:
                genres = json.loads(genres_json)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error for PPN {ppn}: {e}")
                print(f"JSON decode error for PPN {ppn}: {e}")
                continue
            
            genres_with_subgenres = [extract_genre_and_subgenre(x) for x in genres]
            genres = [g[0] for g in genres_with_subgenres]
            subgenres = [sub for _, sub in genres_with_subgenres if sub]

            if len(genres) > 1:
                count_multiple_genres += 1
            elif len(genres) == 1:
                count_single_genres += 1
                
            for genre, subgenre in genres_with_subgenres:
                if subgenre:
                    if genre not in genre_to_subgenre_counts:
                        genre_to_subgenre_counts[genre] = {}
                    if subgenre not in genre_to_subgenre_counts[genre]:
                        genre_to_subgenre_counts[genre][subgenre] = 0
                    genre_to_subgenre_counts[genre][subgenre] += 1
                
            for sub in subgenres:
                subgenre_counts[sub] = subgenre_counts.get(sub, 0) + 1
                    
                if sub not in subgenre_weighted_data:
                    subgenre_weighted_data[sub] = {
                        "mean_word": [], "weight_word": [],
                        "mean_textline": [], "weight_textline": []
                    }
                
                # Add all data for this subgenre
                subgenre_weighted_data[sub]["mean_word"].extend(ppn_results["mean_word"].tolist())
                subgenre_weighted_data[sub]["weight_word"].extend(ppn_results["weight_word"].tolist())
                subgenre_weighted_data[sub]["mean_textline"].extend(ppn_results["mean_textline"].tolist())
                subgenre_weighted_data[sub]["weight_textline"].extend(ppn_results["weight_textline"].tolist())

            for genre in set(genres):
                if genre in counted_genres:
                    continue

                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                counted_genres.add(genre)

                if genre not in genre_weighted_data:
                    genre_weighted_data[genre] = {
                        "mean_word": [], "weight_word": [],
                        "mean_textline": [], "weight_textline": []
                    }

                # Add all data for this genre
                genre_weighted_data[genre]["mean_word"].extend(ppn_results["mean_word"].tolist())
                genre_weighted_data[genre]["weight_word"].extend(ppn_results["weight_word"].tolist())
                genre_weighted_data[genre]["mean_textline"].extend(ppn_results["mean_textline"].tolist())
                genre_weighted_data[genre]["weight_textline"].extend(ppn_results["weight_textline"].tolist())

    logging.info(f"\nNumber of PPNs: {len(matching_ppn_mods)}")
    print(f"\nNumber of PPNs: {len(matching_ppn_mods)}")

    logging.info(f"Number of PPNs with one genre: {count_single_genres}")
    print(f"Number of PPNs with one genre: {count_single_genres}")

    logging.info(f"Number of PPNs with more than one genre: {count_multiple_genres}")
    print(f"Number of PPNs with more than one genre: {count_multiple_genres}")

    all_genres_reduced = set(genre_counts.keys())
    logging.info(f"\nNumber of all unique genres (without subgenres): {len(all_genres_reduced)}")
    print(f"\nNumber of all unique genres (without subgenres): {len(all_genres_reduced)}")

    # Sort genre counts by count (descending) and genre name (ascending)
    genre_counts_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
    genre_counts_df_sorted = genre_counts_df.sort_values(by=['Count', 'Genre'], ascending=[False, True])
    genre_counts_df_sorted.to_csv("genre_publications.csv", index=False)

    if not genre_counts_df.empty:
        logging.info("\nUnique genres and their counts:\n")
        logging.info(genre_counts_df_sorted.to_string(index=False))
        print("\nUnique genres and their counts:\n")
        print(genre_counts_df_sorted.to_string(index=False))

    # Sort subgenre counts by count (descending) and subgenre name (ascending)
    subgenre_counts_df = pd.DataFrame(list(subgenre_counts.items()), columns=['Subgenre', 'Count'])
    subgenre_counts_df_sorted = subgenre_counts_df.sort_values(by=['Count', 'Subgenre'], ascending=[False, True])
    
    logging.info(f"\nNumber of all unique subgenres: {len(subgenre_counts_df_sorted)}")
    print(f"\nNumber of all unique subgenres: {len(subgenre_counts_df_sorted)}")
    
    if not subgenre_counts_df.empty:    
        subgenre_counts_df_sorted.to_csv("subgenre_publications.csv", index=False)        
        logging.info("\nUnique subgenres and their counts:\n")
        logging.info(subgenre_counts_df_sorted.to_string(index=False))
        print("\nUnique subgenres and their counts:\n")
        print(subgenre_counts_df_sorted.to_string(index=False))
        
        genre_subgenre_summary = []

        # Sort genre-subgenre combinations
        for genre in sorted(genre_to_subgenre_counts.keys()):
            subgenre_dict = genre_to_subgenre_counts[genre]
            # Sort subgenres by count (descending) and name (ascending)
            sorted_items = sorted(subgenre_dict.items(), key=lambda x: (-x[1], x[0]))
            subgenre_str = "; ".join([f"{sub} ({count})" for sub, count in sorted_items])
            genre_subgenre_summary.append((genre, subgenre_str))

        genre_subgenre_df = pd.DataFrame(genre_subgenre_summary, columns=["Genre", "Subgenres (with counts)"])
        genre_subgenre_df_sorted = genre_subgenre_df.sort_values(by="Genre")
        genre_subgenre_df_sorted.to_csv("genre_subgenre_combinations.csv", index=False)

        logging.info(f"\nNumber of genres with subgenre associations: {len(genre_subgenre_df_sorted)}")
        print(f"\nNumber of genres with subgenre associations: {len(genre_subgenre_df_sorted)}")
        
        logging.info("\nGenre-subgenre combinations:\n")
        logging.info(genre_subgenre_df_sorted.to_string(index=False))
        print("\nGenre-subgenre combinations:\n")
        print(genre_subgenre_df_sorted.to_string(index=False))
        
        # Use sorted values for subgenre plot
        subgenres, sub_counts = zip(*subgenre_counts_df_sorted.values)
        create_publication_count_horizontal_barplot(
            labels=subgenres,
            counts=sub_counts,
            title='Counts of Unique Subgenres',
            ylabel='Subgenres',
            filename='subgenre_publications.png'
        )

    # Sort genre counts for plotting
    sorted_genre_counts = sorted(genre_counts.items(), key=lambda x: (-x[1], x[0]))

    if sorted_genre_counts:
        _, highest_count = sorted_genre_counts[0]
        plot_threshold = highest_count * 0.04
    else:
        logging.info("No genre available to calculate the threshold.")
        print("No genre available to calculate the threshold.")
        plot_threshold = 0
    
    if use_threshold:
        filtered_genre_counts = [(genre, count) for genre, count in sorted_genre_counts if count > plot_threshold]
    else:
        filtered_genre_counts = sorted_genre_counts

    if not filtered_genre_counts:
        logging.info("No genre exceeds the threshold.")
        print("No genre exceeds the threshold.")
    else:
        genres, counts = zip(*filtered_genre_counts)
        create_publication_count_horizontal_barplot(
            labels=genres,
            counts=counts,
            title='Counts of Unique Genres',
            ylabel='Genres',
            filename='genre_publications.png'
        )

        process_weighted_means(genre_weighted_data, label_name='Genre', filename_prefix='genre', counts_dict=genre_counts)
        
        if len(subgenre_counts) >= 1:
            process_weighted_means(subgenre_weighted_data, label_name='Subgenre', filename_prefix='subgenre', counts_dict=subgenre_counts)
            
            # Create flattened and sorted genre-subgenre combinations
            flattened_data = []
            for genre in sorted(genre_to_subgenre_counts.keys()):
                subgenre_dict = genre_to_subgenre_counts[genre]
                
                # Sort subgenres by count (descending) and name (ascending)
                subgenre_items = list(subgenre_dict.items())
                subgenre_items.sort(key=lambda x: (-x[1], x[0]))
                
                for subgenre, count in subgenre_items:
                    flattened_data.append({
                        'label': f"{genre}: {subgenre}",
                        'count': count
                    })

            # Sort by count (descending) and full label (ascending) for ties
            flattened_data.sort(key=lambda x: (-x['count'], x['label']))
            flattened_labels = [item['label'] for item in flattened_data]
            flattened_counts = [item['count'] for item in flattened_data]

            if flattened_labels and flattened_counts:
                create_publication_count_horizontal_barplot(
                    labels=flattened_labels,
                    counts=flattened_counts,
                    title='Counts of Genre-Subgenre Combinations',
                    ylabel='Genre: Subgenre',
                    filename='genre_subgenre_combinations.png'
                )
        
def dates_evaluation(metadata_df, results_df):
    matching_ppn_mods = set(results_df["ppn"].unique())
    metadata_df = metadata_df[metadata_df["PPN"].isin(matching_ppn_mods)].copy()
    metadata_df.loc[:, 'publication_date'] = pd.to_numeric(metadata_df['publication_date'], errors='coerce')
    try:
        min_year = metadata_df["publication_date"].min()
        max_year = metadata_df["publication_date"].max()
        
        if pd.isna(min_year) and pd.isna(max_year) is None:
            logging.info(f"\nEarliest year: {min_year}")
            logging.info(f"Latest year: {max_year}")
            print(f"\nEarliest year: {min_year}")
            print(f"Latest year: {max_year}")
        
        unique_years = metadata_df["publication_date"].unique()
        num_unique_years = len(unique_years)
        logging.info(f"\nNumber of unique years: {num_unique_years}")
        print(f"\nNumber of unique years: {num_unique_years}")
        
        full_year_range = pd.Series(np.arange(min_year, max_year + 1))
        
        year_counts = metadata_df["publication_date"].value_counts().reindex(full_year_range, fill_value=0)
        year_counts_df = year_counts.reset_index() 
        year_counts_df.columns = ['Year', 'Count']
        year_counts_df['Year'] = year_counts_df['Year'].astype(int)
        year_counts_df['Count'] = year_counts_df['Count'].astype(int)
        year_counts_df.to_csv(f"date_range_{min_year}-{max_year}_publications_per_year.csv", index=False)
        
        num_years_with_zero = (year_counts_df['Count'] == 0).sum()
        logging.info(f"Number of years with no publications: {num_years_with_zero}")
        print(f"Number of years with no publications: {num_years_with_zero}")
        
        logging.info("\nUnique years and their counts:\n")
        logging.info(year_counts_df.to_string(index=False))
        print("\nUnique years and their counts:\n")
        print(year_counts_df.to_string(index=False))
        
        plt.figure(figsize=(max(30, len(full_year_range) * 0.25), 15))
        plt.bar(year_counts_df['Year'], year_counts_df['Count'], color=plt.cm.tab10.colors, width=0.6) # type: ignore
        plt.title('Publication Counts per Year', fontsize=18, fontweight='bold')
        plt.tick_params(axis='x', length=10)
        plt.xticks(full_year_range, fontsize=12, rotation=45)
        plt.yticks(fontsize=13)
        plt.xlabel('Year', fontsize=16)
        plt.ylabel('Count', fontsize=16)
        plt.xlim(min_year - 0.5, max_year + 0.5)
        plt.ylim(0.0, max(year_counts_df['Count']) + 0.01)
        
        if 800 > max(year_counts_df['Count']) >= 400:
            plt.yticks(np.arange(0, max(year_counts_df['Count']) + 1, 50))
        elif 400 > max(year_counts_df['Count']) > 30:
            plt.yticks(np.arange(0, max(year_counts_df['Count']) + 1, 10))
        else:
            plt.yticks(np.arange(0, max(year_counts_df['Count']) + 1, 1))
            
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.tight_layout(pad=1.0)
        plt.savefig(f"date_range_{min_year}-{max_year}_publications_per_year.png")
        plt.close()
        
        grouped = metadata_df.groupby("publication_date")
        years = []
        weighted_mean_word_list = []
        weighted_mean_textline_list = []

        for year, group in grouped:
            group_ppns = group["PPN"]
            subgroup = results_df[results_df["ppn"].isin(group_ppns)]
            if len(subgroup) > 0:
                wm_word = weighted_mean(subgroup["mean_word"], subgroup["weight_word"])
                wm_textline = weighted_mean(subgroup["mean_textline"], subgroup["weight_textline"])
            else:
                wm_word = np.nan
                wm_textline = np.nan
            years.append(year)
            weighted_mean_word_list.append(wm_word)
            weighted_mean_textline_list.append(wm_textline)
        
        plot_df = pd.DataFrame({
            'Year': years,
            'Weighted_Mean_Word': weighted_mean_word_list,
            'Weighted_Mean_Textline': weighted_mean_textline_list
        }).dropna()
        plot_df.to_csv(f"date_range_{min_year}-{max_year}_yearly_weighted_means.csv", index=False)
        
        results_df = results_df.merge(metadata_df[["PPN", "publication_date"]], how="left", left_on="ppn", right_on="PPN")
        
        create_weighted_means_dates_barplot(
            plot_df,
            results_df=results_df,
            groupby_col='publication_date',
            label_col='Year',
            title='Yearly Weighted Means of Word and Textline Confidence Scores',
            filename=f"date_range_{min_year}-{max_year}_yearly_weighted_means.png",
            ha='center'
        )
        
    except ValueError as e:
        logging.info(f"Invalid publication dates: {e}")
        print(f"Invalid publication dates.")
        return

def compute_unweighted_stats(df, bin_col, value_col):
    means = []
    errors = []
    labels = []
    for label, group in df.groupby(bin_col, observed=False):
        data = group[value_col]
        if len(data) > 1:
            mean = data.mean()
            sem = data.std(ddof=1) / np.sqrt(len(data))
        elif len(data) == 1:
            mean = data.iloc[0]
            sem = 0
        else:
            mean = sem = np.nan
        means.append(mean)
        errors.append(sem)
        labels.append(str(label))
    return labels, means, errors

def create_weights_and_num_pages_barplot(data_pairs, titles, xlabels, ylabels, filename, errors_pairs=None):
    plt.figure(figsize=(36, 18))

    for i, (labels, values) in enumerate(data_pairs):
        plt.subplot(1, 2, i + 1)
        
        if errors_pairs is not None:
            yerr = errors_pairs[i]
            plt.bar(labels, values, yerr=yerr, capsize=5, alpha=0.7, edgecolor='black')
        else:
            plt.bar(labels, values, capsize=5, alpha=0.7, edgecolor='black')

        plt.xlabel(xlabels[i], fontsize=18)
        plt.ylabel(ylabels[i], fontsize=18)
        plt.title(titles[i], fontsize=22)
        plt.xticks(rotation=45, ha='right')
        plt.xlim(-0.5, len(values) - 0.5)
        plt.ylim(0, 1)

    plt.tight_layout(pad=2.0)
    plt.savefig(filename)
    plt.close()

def weights_evaluation(results_df):
    cols = ['weight_word', 'weight_textline', 'mean_word', 'mean_textline']
    results_df[cols] = results_df[cols].apply(pd.to_numeric, errors='coerce')

    min_word_weight = results_df['weight_word'].min()
    max_word_weight = results_df['weight_word'].max()
    min_textline_weight = results_df['weight_textline'].min()
    max_textline_weight = results_df['weight_textline'].max()
    
    logging.info(f"\nWord weight range: {min_word_weight} - {max_word_weight}")
    print(f"\nWord weight range: {min_word_weight} - {max_word_weight}")
    logging.info(f"\nTextline weight range: {min_textline_weight} - {max_textline_weight}")
    print(f"\nTextline weight range: {min_textline_weight} - {max_textline_weight}")
    
    word_bins = np.arange(0, results_df['weight_word'].max() + 7500, 7500)
    textline_bins = np.arange(0, results_df['weight_textline'].max() + 750, 750)

    results_df['word_bin'] = pd.cut(results_df['weight_word'], bins=word_bins)
    results_df['textline_bin'] = pd.cut(results_df['weight_textline'], bins=textline_bins)

    word_labels, word_means, word_errors = compute_unweighted_stats(results_df, 'word_bin', 'mean_word')
    textline_labels, textline_means, textline_errors = compute_unweighted_stats(results_df, 'textline_bin', 'mean_textline')

    create_weights_and_num_pages_barplot(
        data_pairs=[(word_labels, word_means), (textline_labels, textline_means)],
        errors_pairs=[word_errors, textline_errors],
        titles=['Mean Confidence per Word Count', 'Mean Confidence per Textline Count'],
        xlabels=['Word Count (Weight)', 'Textline Count (Weight)'],
        ylabels=['Mean Confidence Score', 'Mean Confidence Score'],
        filename="barplot_confs_weights.png"
    )
    
def num_pages_evaluation(results_df):
    cols = ['num_pages', 'mean_word', 'mean_textline', 'weight_word', 'weight_textline']
    results_df[cols] = results_df[cols].apply(pd.to_numeric, errors='coerce')
    
    min_pages = results_df['num_pages'].min()
    max_pages = results_df['num_pages'].max()
    print(f"\nNumber of pages range: {min_pages} - {max_pages}")
    logging.info(f"\nNumber of pages range: {min_pages} - {max_pages}")

    bins = np.arange(0, results_df['num_pages'].max() + 20, 20)
    results_df['pages_bin'] = pd.cut(results_df['num_pages'], bins=bins)

    grouped = results_df.groupby('pages_bin', observed=False)

    word_bin_means = []
    textline_bin_means = []
    word_bin_errors = []
    textline_bin_errors = []
    bin_labels = []
    
    # Calculate weighted mean confidence scores per bin
    for bin_label, group in grouped:
        if len(group) > 0:
            wm_word = weighted_mean(group["mean_word"], group["weight_word"])
            wm_textline = weighted_mean(group["mean_textline"], group["weight_textline"])
            word_se = weighted_standard_error_of_the_mean(group["mean_word"], group["weight_word"])
            textline_se = weighted_standard_error_of_the_mean(group["mean_textline"], group["weight_textline"])
        else:
            wm_word = word_se = np.nan
            wm_textline = textline_se = np.nan

        word_bin_means.append(wm_word)
        textline_bin_means.append(wm_textline)
        word_bin_errors.append(word_se)
        textline_bin_errors.append(textline_se)
        bin_labels.append(str(bin_label))

    create_weights_and_num_pages_barplot(
        data_pairs=[(bin_labels, word_bin_means), (bin_labels, textline_bin_means)],
        errors_pairs=[word_bin_errors, textline_bin_errors],
        titles=['Weighted Mean Word Confidence by Page Count', 'Weighted Mean Textline Confidence by Page Count'],
        xlabels=['Number of Pages', 'Number of Pages'],
        ylabels=['Weighted Mean Word Confidence', 'Weighted Mean Textline Confidence'],
        filename="barplot_weighted_means_by_page_count.png"
    )    
    
def get_ppn_subdirectory_names(results_df, parent_dir, conf_filename):
    if not os.path.exists(parent_dir):
            logging.info(f"Directory does not exist: {parent_dir}")
            print(f"Directory does not exist: {parent_dir}")
            return

    ppn_subdirectory_names = []

    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path) and item.startswith('PPN'):
            ppn_subdirectory_names.append(item)
            
    ppn_df = pd.DataFrame(ppn_subdirectory_names, columns=['PPN'])
    logging.info("\nPPNs found:\n")
    logging.info(ppn_df)
    print("\nPPNs subdirectories found:\n")
    print(ppn_df)
    
    results_df = results_df[results_df["ppn"].isin(ppn_df["PPN"])]
    
    unique_ppns = results_df["ppn"].unique()
    unique_ppn_df = pd.DataFrame(unique_ppns, columns=["PPN"])
    logging.info("\nPPNs from subdirectories with confidence scores:\n")
    logging.info(unique_ppn_df)
    print("\nPPNs from subdirectories with confidence scores:\n")
    print(unique_ppn_df)
    
    unique_ppn_count = results_df["ppn"].nunique()
    logging.info(f"\nNumber of unique PPNs: {unique_ppn_count}\n")
    print(f"\nNumber of unique PPNs: {unique_ppn_count}\n")
    results_df = results_df.sort_values(by='mean_word', ascending=True)
    results_df.to_csv(conf_filename, index=False)
    
def use_dinglehopper(parent_dir, gt_dir, ocr_dir, report_dir):
    # Check if parent_dir exists
    if not os.path.exists(parent_dir):
        logging.info(f"Directory does not exist: {parent_dir}")
        print(f"Directory does not exist: {parent_dir}")
        return

    # Special characters excluded for safety reasons
    special_chars = [';', '&', '|', '`', '(', ')', '{', '}', '~', '>', '>>', '<', '\'', '\"', '\\', ' ', '$', '?', '*', '!', ':', '=', '#', '^']
    
    for param in [parent_dir, gt_dir, ocr_dir, report_dir]:
        if any(char in param for char in special_chars):
            logging.info(f"\nInvalid parameters: special characters {special_chars} are not allowed.\n")
            print(f"\nInvalid parameters: special characters {special_chars} are not allowed.\n")
            return
        
    os.chdir(parent_dir)
    valid_directory_count = 0
    
    for ppn_name in os.listdir():
        full_path = os.path.join(parent_dir, ppn_name)
        if os.path.isdir(full_path) and ppn_name.startswith("PPN"):
            valid_directory_count += 1

    with tqdm(total=valid_directory_count) as progbar:
        for ppn_name in os.listdir():
            full_path = os.path.join(parent_dir, ppn_name)
            
            if os.path.isdir(full_path) and ppn_name.startswith("PPN"):
                progbar.set_description(f"Processing directory: {ppn_name}")
                
                # Change to the subdirectory
                os.chdir(full_path)

                # Check if gt_dir and ocr_dir exist in the current PPN subdirectory
                if not os.path.exists(gt_dir) or not os.path.exists(ocr_dir):
                    logging.info(f"Missing subdirectories in {ppn_name}: {gt_dir} or {ocr_dir}")
                    print(f"Missing subdirectories in {ppn_name}: {gt_dir} or {ocr_dir}")
                    os.chdir(parent_dir)
                    progbar.update(1)
                    return

                command_string = f"ocrd-dinglehopper -I {gt_dir},{ocr_dir} -O {report_dir}"
                command_list = command_string.split()
                try:
                    subprocess.run(command_list, check=True, capture_output=True, text=True, shell=False)
                except subprocess.CalledProcessError as e:
                    logging.info(f"Failed to run command in {ppn_name}. Exit code: {e.returncode}, Error: {e.stderr}")
                    print(f"Failed to run command in {ppn_name}. Exit code: {e.returncode}, Error: {e.stderr}")
                    
                # Change back to the parent directory
                os.chdir(parent_dir)
                progbar.update(1)
    progbar.close()
    
def generate_error_rates(parent_dir_error, report_dir_error, error_rates_filename):
    # Check if parent directory exists
    if not os.path.exists(parent_dir_error):
        logging.info(f"Directory does not exist: {parent_dir_error}")
        print(f"Directory does not exist: {parent_dir_error}")
        return
    
    data = []
    valid_directory_count = 0
    
    # Count valid directories first
    for ppn_name in os.listdir(parent_dir_error):
        full_path = os.path.join(parent_dir_error, ppn_name)
        if os.path.isdir(full_path) and ppn_name.startswith("PPN"):
            valid_directory_count += 1

    with tqdm(total=valid_directory_count) as progbar:
        for ppn_name in os.listdir(parent_dir_error):
            full_path = os.path.join(parent_dir_error, ppn_name)

            if os.path.isdir(full_path) and ppn_name.startswith("PPN"):
                progbar.set_description(f"Processing directory: {ppn_name}")
                
                # Construct path to evaluation directory
                eval_dir = os.path.join(full_path, report_dir_error)

                if os.path.exists(eval_dir) and os.path.isdir(eval_dir):
                    for json_file in os.listdir(eval_dir):
                        if json_file.endswith(".json"):
                            json_path = os.path.join(eval_dir, json_file)
                            
                            try:
                                with open(json_path, 'r') as f:
                                    json_data = json.load(f)
                                    
                                gt = json_data.get('gt', None)
                                ocr = json_data.get('ocr', None)
                                n_characters = json_data.get('n_characters', None)
                                n_words = json_data.get('n_words', None)
                                cer = json_data.get('cer', None)
                                wer = json_data.get('wer', None)
                                
                                # Set invalid error rates to 1
                                if cer is None or cer == "inf" or cer == "infinity" or float(cer) > 1:
                                    cer = 1.0
                                else:
                                    cer = float(cer)

                                if wer is None or wer == "inf" or wer == "infinity" or float(wer) > 1:
                                    wer = 1.0
                                else:
                                    wer = float(wer)
                                    
                                # Extract page number from gt path
                                if gt:
                                    page = gt.split('/')[-1].replace(".page", "")
                                    ppn_page = f'{ppn_name}_{page}'

                                    ppn_name_data = {
                                        'ppn': ppn_name,
                                        'ppn_page': ppn_page,
                                        'gt': gt,
                                        'ocr': ocr,
                                        'cer': cer,
                                        'wer': wer,
                                        'n_characters': n_characters,
                                        'n_words': n_words
                                    }
                                    
                                    data.append(ppn_name_data)
                            except json.JSONDecodeError as e:
                                logging.info(f"Error reading JSON file {json_path}: {e}")
                                print(f"Error reading JSON file {json_path}: {e}")
                                continue
                else:
                    logging.info(f"Evaluation directory not found for {ppn_name}: {eval_dir}")
                    print(f"Evaluation directory not found for {ppn_name}: {eval_dir}")
                
                progbar.update(1)
    progbar.close()

    if data:
        # Create DataFrame and save results
        error_rates_df = pd.DataFrame(data)
        error_rates_df.sort_values(by='ppn_page', ascending=True, inplace=True)
        
        # Save to current directory
        output_path = os.path.join(os.getcwd(), error_rates_filename)
        error_rates_df.to_csv(output_path, index=False)
        
        logging.info("\nResults:\n")
        logging.info(error_rates_df)
        print("\nResults:\n")
        print(error_rates_df)
        print(f"\nResults saved to: {output_path}")
    else:
        logging.info("No data found to process")
        print("No data found to process")
    
def merge_csv(conf_df, error_rates_df, wcwer_filename):
    # Check if files exist
    for filename in [conf_df, error_rates_df]:
        if not os.path.exists(filename):
            logging.info(f"File does not exist: {filename}")
            print(f"File does not exist: {filename}")
            return
    
    try:
        # Read the confidence scores CSV
        ppn_conf_df = pd.read_csv(conf_df)
        
        # Read the error rates CSV
        ppn_error_rates_df = pd.read_csv(error_rates_df)
        
        # Drop the redundant 'ppn' column from error rates
        if 'ppn' in ppn_error_rates_df.columns:
            ppn_error_rates_df.drop(columns=["ppn"], inplace=True)
        
        # Filter confidence scores to only include pages that exist in error rates
        ppn_conf_df = ppn_conf_df[ppn_conf_df['ppn_page'].isin(ppn_error_rates_df['ppn_page'])]
        
        # Merge confidence scores data with the error rates data
        wcwer_df = pd.merge(ppn_conf_df, ppn_error_rates_df, on='ppn_page', how='inner')
        wcwer_df.sort_values(by='ppn_page', ascending=True, inplace=True)
        
        logging.info("\nResults:\n")
        logging.info(wcwer_df)
        print("\nResults:\n")
        print(wcwer_df)
        
        wcwer_df.to_csv(wcwer_filename, index=False)
        print(f"\nMerged data saved to: {wcwer_filename}")
        
    except pd.errors.EmptyDataError:
        logging.info("One or both CSV files are empty")
        print("One or both CSV files are empty")
    except Exception as e:
        logging.info(f"Error processing CSV files: {str(e)}")
        print(f"Error processing CSV files: {str(e)}")
    
def plot_wer_vs_wc(wcwer_csv, plot_filename):
    if not os.path.exists(wcwer_csv):
        logging.info(f"File does not exist: {wcwer_csv}")
        print(f"File does not exist: {wcwer_csv}")
        return
            
    try:
        wcwer_df = pd.read_csv(wcwer_csv)
        
        # Check if required columns exist
        required_columns = ['mean_word', 'wer', 'ppn_page']
        missing_columns = [col for col in required_columns if col not in wcwer_df.columns]
        if missing_columns:
            logging.info(f"Missing required columns: {missing_columns}")
            print(f"Missing required columns: {missing_columns}")
            return
        
        wcwer_df['mean_word'] = pd.to_numeric(wcwer_df['mean_word'], errors='coerce')
        wcwer_df['wer'] = pd.to_numeric(wcwer_df['wer'], errors='coerce')
        wcwer_df = wcwer_df.dropna(subset=['mean_word', 'wer'])
        
        if wcwer_df.empty:
            logging.info("No valid data to plot after processing")
            print("No valid data to plot after processing")
            return
        
        # Sort values
        wcwer_df.sort_values(by='mean_word', ascending=True, inplace=True)
        
        # Create plot
        ppn_pages_count = wcwer_df["ppn_page"].nunique()
        plt.figure(figsize=(ppn_pages_count * 0.1, ppn_pages_count * 0.1))
        
        plt.scatter(wcwer_df["mean_word"], wcwer_df["wer"], 
                   color='blue', marker='x', s=100)
        
        plt.xlabel('Mean Word Confidence Score (WC)', fontsize=16)
        plt.ylabel('Word Error Rate (WER)', fontsize=16)
        plt.title('WER(WC)', fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid(linestyle='--', alpha=0.8)
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.tight_layout(pad=1.0)
        plt.savefig(plot_filename)
        plt.close()
        
        print(f"Plot saved as: {plot_filename}")
        
    except pd.errors.EmptyDataError:
        logging.info("CSV file is empty")
        print("CSV file is empty")
    except Exception as e:
        logging.info(f"Error processing CSV file: {str(e)}")
        print(f"Error processing CSV file: {str(e)}")
        
def plot_wer_vs_wc_interactive(wcwer_csv_inter, plot_filename_inter):
    if not os.path.exists(wcwer_csv_inter):
        logging.info(f"File does not exist: {wcwer_csv_inter}")
        print(f"File does not exist: {wcwer_csv_inter}")
        return
            
    try:
        wcwer_df = pd.read_csv(wcwer_csv_inter)
        
        # Check if required columns exist
        required_columns = ['mean_word', 'wer', 'ppn_page']
        missing_columns = [col for col in required_columns if col not in wcwer_df.columns]
        if missing_columns:
            logging.info(f"Missing required columns: {missing_columns}")
            print(f"Missing required columns: {missing_columns}")
            return
        
        wcwer_df['mean_word'] = pd.to_numeric(wcwer_df['mean_word'], errors='coerce')
        wcwer_df['wer'] = pd.to_numeric(wcwer_df['wer'], errors='coerce')
        wcwer_df = wcwer_df.dropna(subset=['mean_word', 'wer', 'ppn_page'])
        
        if wcwer_df.empty:
            logging.info("No valid data to plot after processing")
            print("No valid data to plot after processing")
            return
        
        X = wcwer_df['mean_word'].values.reshape(-1, 1)
        y = wcwer_df['wer'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Calculate correlations on training data
        pearson_corr = np.corrcoef(X_train.flatten(), y_train)[0, 1]
        spearman_corr = spearmanr(X_train.flatten(), y_train)[0]
        
        # Linear regression on training data
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        linear_r2_train = linear_model.score(X_train, y_train)
        linear_r2_test = linear_model.score(X_test, y_test)
        
        # Calculate MSE for linear regression
        y_pred_linear_train = linear_model.predict(X_train)
        y_pred_linear_test = linear_model.predict(X_test)
        mse_linear_train = np.mean((y_train - y_pred_linear_train) ** 2)
        mse_linear_test = np.mean((y_test - y_pred_linear_test) ** 2)
        
        # Polynomial regression (degree 2) on training data
        poly = PolynomialFeatures(degree=2)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.transform(X_test)
        
        poly_model = LinearRegression()
        poly_model.fit(X_poly_train, y_train)
        poly_r2_train = poly_model.score(X_poly_train, y_train)
        poly_r2_test = poly_model.score(X_poly_test, y_test)
        
        # Calculate MSE for polynomial regression
        y_pred_poly_train = poly_model.predict(X_poly_train)
        y_pred_poly_test = poly_model.predict(X_poly_test)
        mse_poly_train = np.mean((y_train - y_pred_poly_train) ** 2)
        mse_poly_test = np.mean((y_test - y_pred_poly_test) ** 2)
        
        stats_data = {
            'Metric': [
                'Number of Training Points',
                'Number of Test Points',
                'Pearson Correlation Coefficient (Train)',
                'Spearman Correlation Coefficient (Train)',
                'Linear Regression R^2 (Train)',
                'Linear Regression R^2 (Test)',
                'Linear Regression MSE (Train)',
                'Linear Regression MSE (Test)',
                'Linear Regression Slope',
                'Linear Regression Intercept',
                'Polynomial Regression R^2 (Train)',
                'Polynomial Regression R^2 (Test)',
                'Polynomial Regression MSE (Train)',
                'Polynomial Regression MSE (Test)',
                'Polynomial Coefficient (x^2)',
                'Polynomial Coefficient (x)',
                'Polynomial Coefficient (intercept)'
            ],
            'Value': [
                len(X_train),
                len(X_test),
                pearson_corr,
                spearman_corr,
                linear_r2_train,
                linear_r2_test,
                mse_linear_train,
                mse_linear_test,
                linear_model.coef_[0],
                linear_model.intercept_,
                poly_r2_train,
                poly_r2_test,
                mse_poly_train,
                mse_poly_test,
                poly_model.coef_[2],
                poly_model.coef_[1],
                poly_model.coef_[0]
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        
        # Save statistical analysis to CSV
        stats_filename = plot_filename_inter.replace('.html', '_statistics.csv')
        stats_df.to_csv(stats_filename, index=False)
        
        # Generate points for regression lines
        X_smooth = np.linspace(0, 1, 100).reshape(-1, 1)
        y_linear = linear_model.predict(X_smooth)
        y_poly = poly_model.predict(poly.transform(X_smooth))
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add training scatter plot
        fig.add_trace(go.Scatter(
            x=X_train.flatten(),
            y=y_train,
            mode='markers',
            name='Training Points',
            marker=dict(
                size=10,
                color='blue',
                line=dict(width=2, color='DarkSlateGrey')
            ),
            hovertemplate=(
                "PPN Page: %{hovertext}<br>"
                "Mean Word Confidence: %{x:.3f}<br>" +
                "WER: %{y:.3f}"
            ),
            hovertext=wcwer_df["ppn_page"]
        ))
        
        # Add test scatter plot
        fig.add_trace(go.Scatter(
            x=X_test.flatten(),
            y=y_test,
            mode='markers',
            name='Test Points',
            marker=dict(
                size=10,
                color='orange',
                line=dict(width=2, color='DarkSlateGrey')
            ),
            hovertemplate=(
                "PPN Page: %{hovertext}<br>"
                "Mean Word Confidence: %{x:.3f}<br>" +
                "WER: %{y:.3f}"
            ),
            hovertext=wcwer_df["ppn_page"]
        ))
        
        # Add linear regression line
        fig.add_trace(go.Scatter(
            x=X_smooth.flatten(),
            y=y_linear,
            mode='lines',
            name=f'Linear Regression (Test R^2 = {linear_r2_test:.3f}, MSE = {mse_linear_test:.3f})',
            line=dict(color='red', dash='solid')
        ))
        
        # Add polynomial regression line
        fig.add_trace(go.Scatter(
            x=X_smooth.flatten(),
            y=y_poly,
            mode='lines',
            name=f'Polynomial Regression (Test R^2 = {poly_r2_test:.3f}, MSE = {mse_poly_test:.3f})',
            line=dict(color='green', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'WER vs WC',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Mean Word Confidence Score (WC)",
            yaxis_title="Word Error Rate (WER)",
            width=1000,
            height=1000,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        fig.update_xaxes(range=[-0.01, 1.01])
        fig.update_yaxes(range=[-0.01, 1.01])
        
        # Save interactive plot
        pyo.plot(fig, filename=plot_filename_inter, auto_open=False)
        
        # Create and save static plot
        plt.figure(figsize=(12, 12))
        plt.scatter(X_train, y_train, color='blue', marker='x', s=100, label='Training Points')
        plt.scatter(X_test, y_test, color='orange', marker='x', s=100, label='Test Points')
        plt.plot(X_smooth, y_linear, color='red', 
                label=f'Linear Regression (Test R^2 = {linear_r2_test:.3f}, MSE = {mse_linear_test:.3f})')
        plt.plot(X_smooth, y_poly, color='green', linestyle='--', 
                label=f'Polynomial Regression (Test R^2 = {poly_r2_test:.3f}, MSE = {mse_poly_test:.3f})')
        plt.xlabel('Mean Word Confidence Score (WC)', fontsize=12)
        plt.ylabel('Word Error Rate (WER)', fontsize=12)
        plt.title('WER vs WC', fontsize=14)
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        static_image = plot_filename_inter.replace('.html', '.png')
        plt.savefig(static_image, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nStatistical Analysis:")
        print(stats_df.to_string(index=False))
        print(f"\nFiles saved:")
        print(f"Statistical analysis: {stats_filename}")
        print(f"Interactive plot (HTML): {plot_filename_inter}")
        print(f"Static plot (PNG): {static_image}")
        logging.info("\nStatistical Analysis:")
        logging.info(stats_df.to_string(index=False))
        logging.info(f"\nFiles saved:")
        logging.info(f"Statistical analysis: {stats_filename}")
        logging.info(f"Interactive plot (HTML): {plot_filename_inter}")
        logging.info(f"Static plot (PNG): {static_image}")
        
    except pd.errors.EmptyDataError:
        logging.info("CSV file is empty")
        print("CSV file is empty")
    except Exception as e:
        logging.info(f"Error processing CSV file: {str(e)}")
        print(f"Error processing CSV file: {str(e)}")
    
def filter_range(df, column, value_range):
    if value_range[0] == 0:
        return df[(df[column] >= value_range[0]) & (df[column] <= value_range[1])]
    else:
        return df[(df[column] > value_range[0]) & (df[column] <= value_range[1])]


def generate_dataframes(
    csv_files: list[str],
    metadata_csv: str,
    check_value_errors: bool = False,
    check_duplicates: bool = False,
    check_raw_genres: bool = False,
    aggregate_mode: str = "ppn_page"
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    for file in csv_files:
        if not os.path.exists(file):
            exc = FileNotFoundError(file)
            logging.exception(exc)
            raise exc

    if not os.path.exists(metadata_csv):
        exc = FileNotFoundError(metadata_csv)
        logging.exception(exc)
        raise exc
        
    all_results = []
    value_error_pages = []
    with tqdm(total=len(csv_files)) as progbar:
        for csv_file in csv_files:
            progbar.set_description(f"Processing file: {csv_file}")
            try:
                with load_csv(csv_file) as rows:
                    for i, row in enumerate(rows):
                        if i == 0:
                            continue
                            
                        ppn = f'{row[0]}'
                        ppn_page = f'{row[0]}_{row[1]}'
                        
                        try:
                            textline_confs = list(map(float, row[3].split(' ')))
                            word_confs = list(map(float, row[4].split(' ')))
                        except ValueError:
                            if check_value_errors:
                                value_error_pages.append([ppn, ppn_page])
                            continue
                        
                        # Set the number of words and text lines as the weights for a PPN_PAGE
                        weight_word = len(word_confs)
                        weight_textline = len(textline_confs)
                        
                        mean_textline, median_textline, standard_deviation_textline = statistics(textline_confs)
                        mean_word, median_word, standard_deviation_word = statistics(word_confs)
                        all_results.append([
                            ppn,
                            ppn_page,
                            mean_word,
                            median_word,
                            standard_deviation_word,
                            mean_textline,
                            median_textline,
                            standard_deviation_textline,
                            weight_word,
                            weight_textline,
                        ])
                                               
            except csv.Error as e:
                exc = ValueError(f"CSV error: {e} in file: {csv_file}. \nIncrease the CSV field size limit!")
                logging.exception(exc)
                raise exc
            progbar.update(1)
    progbar.close()
    
    results_df = pd.DataFrame(all_results, columns=[
        "ppn",
        "ppn_page",
        "mean_word",
        "median_word",
        "standard_deviation_word",
        "mean_textline",
        "median_textline",
        "standard_deviation_textline",
        "weight_word",
        "weight_textline"
    ])
    
    if aggregate_mode == "ppn":
        page_counts = results_df.groupby("ppn")["ppn_page"].nunique().reset_index()
        page_counts.rename(columns={"ppn_page": "num_pages"}, inplace=True)

        grouped = results_df.groupby("ppn")
        aggregated_results = []

        for ppn, group in grouped:
            word_weights = group["weight_word"].to_numpy()
            textline_weights = group["weight_textline"].to_numpy()

            mean_word = weighted_mean(group["mean_word"], word_weights)
            median_word = weighted_percentile(group["median_word"], word_weights, [50])[0]
            std_word = weighted_std(group["standard_deviation_word"], word_weights)

            mean_textline = weighted_mean(group["mean_textline"], textline_weights)
            median_textline = weighted_percentile(group["median_textline"], textline_weights, [50])[0]
            std_textline = weighted_std(group["standard_deviation_textline"], textline_weights)

            total_word_weight = word_weights.sum()
            total_textline_weight = textline_weights.sum()

            num_pages = page_counts.loc[page_counts["ppn"] == ppn, "num_pages"].values[0]

            aggregated_results.append([
                ppn, num_pages, mean_word, median_word, std_word,
                mean_textline, median_textline, std_textline,
                total_word_weight, total_textline_weight
            ])

        results_df = pd.DataFrame(aggregated_results, columns=[
            "ppn", "num_pages", "mean_word", "median_word", "standard_deviation_word",
            "mean_textline", "median_textline", "standard_deviation_textline",
            "weight_word", "weight_textline"
        ])
    
    # "originInfo-publication0_dateIssued" changed to "publication_date"
    metadata_df = pd.DataFrame(load_csv_to_list(metadata_csv)[1:], columns=["PPN", "genre-aad", "publication_date"])
    
    if check_value_errors:
        value_error_df = pd.DataFrame(value_error_pages, columns=["ppn", "ppn_page"])
        value_error_df = value_error_df[value_error_df["ppn"].isin(metadata_df["PPN"])]
        value_error_df = value_error_df.sort_values(by='ppn_page', ascending=True)
        ppn_counts = value_error_df['ppn'].nunique()
        logging.info(f"\nNumber of PPNs excluded because of a ValueError: {ppn_counts}")
        print(f"\nNumber of PPNs excluded because of a ValueError: {ppn_counts}")
        ppn_page_counts = value_error_df['ppn_page'].value_counts()
        logging.info(f"Number of PPN_PAGEs excluded because of a ValueError: {ppn_page_counts.sum()}")
        print(f"Number of PPN_PAGEs excluded because of a ValueError: {ppn_page_counts.sum()}")
        value_error_df.to_csv("value_error_pages.csv", index=False)
    
    # Reduce the results dataframe to include only those PPNs that are in the PPN list ppns_pipeline_batch_01_2024.txt
    results_df = results_df[results_df["ppn"].isin(metadata_df["PPN"])]
    
    # Change all years that are empty strings or "18XX" to "2025"
    metadata_df.loc[metadata_df["publication_date"].isin(["", "18XX"]), "publication_date"] = "2025"
    
    # Change all genres that are empty strings to "Unbekannt"
    metadata_df.loc[metadata_df["genre-aad"].isin([""]), "genre-aad"] = "{'Unbekannt'}"
    
    # Change the genre separation from slashes to commas
    metadata_df['genre-aad'] = metadata_df['genre-aad'].apply(
        lambda genre: "{" + genre.strip().strip("{ }")
                                 .replace("  / ", "', '")
                                 .replace(" / ", "', '") + "}"
    )

    # Fill incomplete genre names
    metadata_df['genre-aad'] = metadata_df['genre-aad'].apply(
        lambda genre: genre.replace("'Ars'", "'Ars moriendi'")
                           .replace("'moriendi'", "'Ars moriendi'")
                           .strip()
    )
    
    # Merge loose subgenres with their genre
    metadata_df['genre-aad'] = metadata_df['genre-aad'].apply(
        lambda genre: genre.replace("'jur.'", "'Kommentar:jur.'")
                           .replace("'hist.'", "'Kommentar:hist.'")
                           .replace("'theol.'", "'Kommentar:theol.'")
                           .replace("'lit.'", "'Kommentar:lit.'")
                           .replace("Gelegenheitsschrift.Fest", "Gelegenheitsschrift:Fest")
                           .strip()
    )
    
    if check_raw_genres:
        metadata_df_unique = metadata_df["genre-aad"].unique()
        metadata_df_unique_df = pd.DataFrame(metadata_df_unique, columns=["genre-aad"])
        logging.info(f"\nAll raw genres in {metadata_csv}: \n")
        print(f"\nAll raw genres in {metadata_csv}: \n")
        logging.info(metadata_df_unique_df.to_string(index=False))
        print(metadata_df_unique_df.to_string(index=False))
        metadata_df_unique_df.to_csv("genres_raw.csv", index=False)
    
    if check_duplicates:
        ppn_page_counts = results_df['ppn_page'].value_counts()
        logging.info(f"\nNumber of PPN_PAGEs: {ppn_page_counts.sum()}")
        print(f"\nNumber of PPN_PAGEs: {ppn_page_counts.sum()}")
        
        singles_summary = ppn_page_counts[ppn_page_counts == 1]
        logging.info(f"Number of PPN_PAGEs with a single occurrence: {singles_summary.sum()}")
        print(f"Number of PPN_PAGEs with a single occurrence: {singles_summary.sum()}")

        duplicates_summary = ppn_page_counts[ppn_page_counts > 1]
        num_unique_duplicates = len(duplicates_summary)
        logging.info(f"Number of PPN_PAGEs with multiple occurrences: {num_unique_duplicates}")
        print(f"Number of PPN_PAGEs with multiple occurrences: {num_unique_duplicates}")
        non_duplicated_ppn_pages = ppn_page_counts[ppn_page_counts == 1].index

        # Exclude PPN_PAGEs that are not duplicated
        filtered_df = results_df[~results_df['ppn_page'].isin(non_duplicated_ppn_pages)]
        metadata_filtered = metadata_df[['PPN', 'publication_date', 'genre-aad']]
        filtered_df = filtered_df.merge(metadata_filtered, left_on='ppn', right_on='PPN')
        filtered_df.drop(columns=['PPN'], inplace=True)
        filtered_df = filtered_df.sort_values(by='ppn_page', ascending=True)
        filtered_df.to_csv("duplicates.csv", index=False)

    return results_df, metadata_df
    
def plot_everything(
    csv_files: list[str],
    metadata_csv: str,
    search_genre,
    search_subgenre,
    plot_file="statistics_results.jpg",
    search_ppn=None,
    search_date=None,
    date_range: Optional[Tuple[int, int]] = None,
    use_top_ppns_word=False,
    use_bottom_ppns_word=False,
    num_top_ppns_word=1,
    num_bottom_ppns_word=1,
    use_top_ppns_textline=False,
    use_bottom_ppns_textline=False,
    num_top_ppns_textline=1,
    num_bottom_ppns_textline=1,
    mean_word_conf=None,
    mean_textline_conf=None,
    mean_word_confs_range: Optional[Tuple[float, float]] = None,
    mean_textline_confs_range: Optional[Tuple[float, float]] = None,
    show_genre_evaluation=False,
    output: Optional[IO] = None,
    show_dates_evaluation=False,
    show_results=False,
    use_best_mean_word_confs=False,
    use_worst_mean_word_confs=False,
    num_best_mean_word_confs=1,
    num_worst_mean_word_confs=1,
    use_best_mean_textline_confs=False,
    use_worst_mean_textline_confs=False,
    num_best_mean_textline_confs=1,
    num_worst_mean_textline_confs=1,
    parent_dir=None,
    conf_filename=None,
    use_logging=None,
    histogram_info: bool =False,
    check_value_errors: bool = False,
    check_duplicates: bool = False,
    check_raw_genres: bool = False,
    aggregate_mode='ppn_page',
    weighting_method: str = "both",
    search_weight_word=None,
    search_weight_textline=None,
    weight_word_range: Optional[Tuple[int, int]] = None,
    weight_textline_range: Optional[Tuple[int, int]] = None,
    show_weights_evaluation=False,
    search_number_of_pages=False,
    number_of_pages_range: Optional[Tuple[int, int]] = None,
    show_number_of_pages_evaluation=False
):
    if use_logging:
        setup_logging("plot")

    results_df, metadata_df = generate_dataframes(
        csv_files,
        metadata_csv,
        check_value_errors=check_value_errors,
        check_duplicates=check_duplicates,
        check_raw_genres=check_raw_genres,
        aggregate_mode=aggregate_mode
    )
            
    # Count the number of unique PPNs in the results dataframe
    all_ppns = results_df["ppn"].unique()
    results_df = results_df.sort_values(by='mean_word', ascending=True)
    
    if search_number_of_pages is not None:
        if aggregate_mode == "ppn":
            results_df = results_df[(results_df['num_pages'] == search_number_of_pages)]
        else: 
            logging.info("\n'-np' can only be used if '-a ppn' is also provided.")
            print("\n'-np' can only be used if '-a ppn' is also provided.")
            return
            
    if number_of_pages_range is not None:
        if aggregate_mode == "ppn":
            results_df = filter_range(results_df, 'num_pages', number_of_pages_range)
        else: 
            logging.info("\n'-npr' can only be used if '-a ppn' is also provided.")
            print("\n'-npr' can only be used if '-a ppn' is also provided.")
            return
        
    if search_ppn:
        results_df = results_df[results_df["ppn"].isin(
        metadata_df.loc[(metadata_df["PPN"].astype(str) == search_ppn), "PPN"])] # type: ignore
    
    if search_date is not None: # "is not None" enables zero as input
        results_df = results_df[results_df["ppn"].isin(
        metadata_df.loc[(metadata_df["publication_date"].astype(int) == search_date), "PPN"])] # type: ignore
        
    if search_weight_word is not None:
        results_df = results_df[(results_df['weight_word'] == search_weight_word)]
        
    if search_weight_textline is not None:
        results_df = results_df[(results_df['weight_textline'] == search_weight_textline)]
    
    if date_range:
        results_df = results_df[results_df["ppn"].isin(
        metadata_df.loc[
            (metadata_df["publication_date"].astype(int) >= date_range[0]) &
            (metadata_df["publication_date"].astype(int) <= date_range[1]),
            "PPN"])]
            
    if mean_word_conf is not None:
        results_df = results_df[(results_df['mean_word'] == mean_word_conf)]
        
    if mean_textline_conf is not None:
        results_df = results_df.sort_values(by='mean_textline', ascending=True)
        results_df = results_df[(results_df['mean_textline'] == mean_textline_conf)]
            
    if mean_word_confs_range:
        results_df = filter_range(results_df, 'mean_word', mean_word_confs_range)
            
    if mean_textline_confs_range:
        results_df = results_df.sort_values(by='mean_textline', ascending=True)
        results_df = filter_range(results_df, 'mean_textline', mean_textline_confs_range)
                
    if weight_word_range:
        results_df = filter_range(results_df, 'weight_word', weight_word_range)
                
    if weight_textline_range:
        results_df = filter_range(results_df, 'weight_textline', weight_textline_range)
            
    if search_genre:
        # Escape special characters in the search_genre string
        escaped_genre = re.escape(search_genre)
        pattern = r"\{\s*[^}]*?\b" + escaped_genre + r"\b[^}]*?\}"
        results_df = results_df[results_df["ppn"].isin(metadata_df.loc[metadata_df["genre-aad"].str.match(pattern, na=False), "PPN"])]
        
    if search_subgenre:
        escaped_subgenre = re.escape(search_subgenre)
        pattern = r":[\s]*" + escaped_subgenre + r"(?!\w)"
        results_df = results_df[results_df["ppn"].isin(
            metadata_df.loc[metadata_df["genre-aad"].str.contains(pattern, na=False), "PPN"]
        )]
        
    if use_top_ppns_word:
        results_df = results_df[((results_df["mean_word"] >= 0.95) & (results_df["mean_word"] <= 1.0))]
        results_df = results_df.sort_values(by='mean_word', ascending=False)
        results_df = results_df.head(num_top_ppns_word)
    elif use_bottom_ppns_word:
        results_df = results_df[((results_df["mean_word"] >= 0.0) & (results_df["mean_word"] <= 0.05))]
        results_df = results_df.head(num_bottom_ppns_word)
        
    if use_top_ppns_textline:
        results_df = results_df[((results_df["mean_textline"] >= 0.95) & (results_df["mean_textline"] <= 1.0))]
        results_df = results_df.sort_values(by='mean_textline', ascending=False)
        results_df = results_df.head(num_top_ppns_textline)
    elif use_bottom_ppns_textline:
        results_df = results_df[((results_df["mean_textline"] >= 0.0) & (results_df["mean_textline"] <= 0.05))]
        results_df = results_df.sort_values(by='mean_textline', ascending=True)
        results_df = results_df.head(num_bottom_ppns_textline)
        
    if use_best_mean_word_confs:
        results_df = results_df.sort_values(by='mean_word', ascending=False)
        results_df = results_df.head(num_best_mean_word_confs)
    elif use_worst_mean_word_confs:
        results_df = results_df.head(num_worst_mean_word_confs)
        
    if use_best_mean_textline_confs:
        results_df = results_df.sort_values(by='mean_textline', ascending=False)
        results_df = results_df.head(num_best_mean_textline_confs)
    elif use_worst_mean_textline_confs:
        results_df = results_df.sort_values(by='mean_textline', ascending=True)
        results_df = results_df.head(num_worst_mean_textline_confs)
        
    if parent_dir and conf_filename:
        get_ppn_subdirectory_names(results_df=results_df, parent_dir=parent_dir, conf_filename=conf_filename)
        
    results_df_unique = results_df["ppn"].unique()
    logging.info(f"\nResults: {len(results_df_unique)} of {len(all_ppns)} PPNs contained in {len(csv_files)} CSV_FILES match the applied filter:\n")
    print(f"\nResults: {len(results_df_unique)} of {len(all_ppns)} PPNs contained in {len(csv_files)} CSV_FILES match the applied filter:\n")
    sum_weight_word = results_df["weight_word"].sum()
    sum_weight_textline = results_df["weight_textline"].sum()
    logging.info(f"\nNumber of all words: {sum_weight_word}")
    print(f"\nNumber of all words: {sum_weight_word}")
    logging.info(f"Number of all textlines: {sum_weight_textline}\n")
    print(f"Number of all textlines: {sum_weight_textline}\n")
    
    if show_results:
        if len(results_df_unique) > 0:
            if aggregate_mode == 'ppn_page':
                filtered_results_df = results_df[['ppn', 'ppn_page', 'mean_word', 'mean_textline', 'weight_word', 'weight_textline']]
            elif aggregate_mode == 'ppn':
                filtered_results_df = results_df[['ppn', 'num_pages', 'mean_word', 'mean_textline', 'weight_word', 'weight_textline']]
            metadata_filtered = metadata_df[['PPN', 'publication_date', 'genre-aad']]
            filtered_results_df = filtered_results_df.merge(metadata_filtered, left_on='ppn', right_on='PPN')
            filtered_results_df.drop(columns=['PPN'], inplace=True)
            logging.info(filtered_results_df.to_string(index=False))
            print(filtered_results_df.to_string(index=False))
        else:
            logging.info("\nNo PPNs found for the applied filters.")
            print("\nNo PPNs found for the applied filters.")
        
    if show_genre_evaluation:
        genre_evaluation(metadata_df, results_df)
        
    if show_dates_evaluation:
        dates_evaluation(metadata_df, results_df)
    
    if show_weights_evaluation:
        weights_evaluation(results_df)
    
    if show_number_of_pages_evaluation:
        if aggregate_mode == "ppn":
            num_pages_evaluation(results_df)
        else: 
            logging.info("\n'-ne' can only be used if '-a ppn' is also provided.")
            print("\n'-ne' can only be used if '-a ppn' is also provided.")
            return    
    
    if results_df.empty:
        logging.info("\nThere are no results matching the applied filters.")
        print("\nThere are no results matching the applied filters.")
        return

    metadata_filtered = metadata_df[['PPN', 'publication_date', 'genre-aad']]
    results_df = results_df.merge(metadata_filtered, left_on='ppn', right_on='PPN')
    results_df.drop(columns=['PPN'], inplace=True)
    results_df_description = results_df.describe(include='all')
    logging.info("\nResults description: \n")
    logging.info(results_df_description)
    print("\nResults description: \n")
    print(results_df_description)
        
    if output:
        results_df.to_csv(output, index=False)
        logging.info(f"\nSaved results to: {output.name}")
        print(f"\nSaved results to: {output.name}")
        output_desc = output.name.split(".")[0] + "_desc.csv" 
        results_df_description.to_csv(output_desc, index=False)
        logging.info(f"\nSaved results description to: {output_desc}")
        print(f"\nSaved results description to: {output_desc}")
        
    plot_file_weighted = plot_file.split(".")[0] + "_weighted." + plot_file.split(".")[1]

    if aggregate_mode == "ppn_page":
        if weighting_method == "both":
            create_plots(
                results_df,
                None,
                None,
                plot_file=plot_file,
                histogram_info=histogram_info,
                general_title="Analysis of Confidence Scores per Page",
            )
            create_plots(
                results_df,
                weights_word=results_df["weight_word"],
                weights_textline=results_df["weight_textline"],
                plot_file=plot_file_weighted,
                histogram_info=histogram_info,
                general_title="Analysis of Confidence Scores per Page (Weighted)",
            )
        elif weighting_method == "unweighted":
            create_plots(
                results_df,
                None,
                None,
                plot_file=plot_file,
                histogram_info=histogram_info,
                general_title="Analysis of Confidence Scores per Page",
            )
        elif weighting_method == "weighted":
            create_plots(
                results_df,
                weights_word=results_df["weight_word"],
                weights_textline=results_df["weight_textline"],
                plot_file=plot_file_weighted,
                histogram_info=histogram_info,
                general_title="Analysis of Confidence Scores per Page (Weighted)",
            )
    elif aggregate_mode == "ppn":
        if weighting_method == "both":
            create_plots(
                results_df,
                None,
                None,
                plot_file=plot_file,
                histogram_info=histogram_info,
                general_title="Analysis of Confidence Scores per PPN",
            )
            create_plots(
                results_df,
                weights_word=results_df["weight_word"],
                weights_textline=results_df["weight_textline"],
                plot_file=plot_file_weighted,
                histogram_info=histogram_info,
                general_title="Analysis of Confidence Scores per PPN (Weighted)",
            )
        elif weighting_method == "unweighted":
            create_plots(
                results_df,
                None,
                None,
                plot_file=plot_file,
                histogram_info=histogram_info,
                general_title="Analysis of Confidence Scores per PPN",
            )
        elif weighting_method == "weighted":
            create_plots(
                results_df,
                weights_word=results_df["weight_word"],
                weights_textline=results_df["weight_textline"],
                plot_file=plot_file_weighted,
                histogram_info=histogram_info,
                general_title="Analysis of Confidence Scores per PPN (Weighted)",
            )


def evaluate_everything(
    parent_dir=None,
    gt_dir=None,
    ocr_dir=None,
    report_dir=None,
    parent_dir_error=None,
    report_dir_error=None,
    error_rates_filename=None,
    use_logging=None,
    conf_df=None,
    error_rates_df=None,
    wcwer_filename=None,
    wcwer_csv=None,
    plot_filename=None,
    wcwer_csv_inter=None,
    plot_filename_inter=None
):
    if use_logging:
        setup_logging("evaluate")
        
    if parent_dir and gt_dir and ocr_dir and report_dir:
        use_dinglehopper(parent_dir=parent_dir, gt_dir=gt_dir, ocr_dir=ocr_dir, report_dir=report_dir)

    if parent_dir_error and report_dir_error and error_rates_filename:
        generate_error_rates(parent_dir_error=parent_dir_error, report_dir_error=report_dir_error, error_rates_filename=error_rates_filename)
        
    if conf_df and error_rates_df and wcwer_filename:
        merge_csv(conf_df=conf_df, error_rates_df=error_rates_df, wcwer_filename=wcwer_filename)
                
    if wcwer_csv and plot_filename:
        plot_wer_vs_wc(wcwer_csv=wcwer_csv, plot_filename=plot_filename)
        
    if wcwer_csv_inter and plot_filename_inter:
        plot_wer_vs_wc_interactive(wcwer_csv_inter=wcwer_csv_inter, plot_filename_inter=plot_filename_inter)
        
