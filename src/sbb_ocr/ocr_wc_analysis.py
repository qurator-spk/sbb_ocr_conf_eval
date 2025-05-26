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
import re
from matplotlib.ticker import MaxNLocator

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
    else:
        bin_counts = binned_data.value_counts().sort_index()

    # Reindex to ensure all bins appear even if count is 0
    bin_counts = bin_counts.reindex(all_intervals, fill_value=0)
    bin_lefts = [interval.left for interval in bin_counts.index]
    bin_widths = [interval.right - interval.left for interval in bin_counts.index]

    ax.bar(bin_lefts, bin_counts.values, width=bin_widths, align='edge', color=color, edgecolor='black', alpha=0.6)
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

            bin_label = f"Bin {left_bracket}{left:.2f}, {right:.2f}{right_bracket}: {bin_counts[interval]}"
            print(bin_label)
            logging.info(bin_label)
            
def weighted_mean(data, weights):
    return np.average(data, weights=weights)
    
def weighted_std(std_devs, weights):
    std_devs = np.array(std_devs)
    weights = np.array(weights)
    variances = np.square(std_devs)
    
    # Compute the weighted mean of variances
    weighted_variance = weighted_mean(variances, weights)
    
    # Calculate the weighted pooled standard deviation
    pooled_std = np.sqrt(weighted_variance)
    return pooled_std
    
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
    
def plot_density(ax, data, weights, xlabel, ylabel, density_color):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    
    try:
        # Create a kernel density estimation (KDE) using the provided data and weights
        kde = gaussian_kde(data, weights=weights)
        x_range = np.linspace(0, 1, 100)
        
        # Evaluate the density values
        density_values = kde(x_range)
        ax.set_ylim(bottom=0, top=np.max(density_values) * 1.1)

        mean = weighted_mean(data, weights) if weights is not None else np.mean(data)
        q25, q50, q75 = weighted_percentile(data, weights, [25, 50, 75])

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
        
    except LinAlgError as e:
        logging.info(f"Cannot plot the data!\nLinAlgError encountered while performing KDE: \n{e}. \nThe data does not have enough variation in its dimensions to accurately estimate a continuous probability density function. \nIncrease the number of PPNs to be filtered!\n")
        print(f"Cannot plot the data!\nLinAlgError encountered while performing KDE: \n{e}. \nThe data does not have enough variation in its dimensions to accurately estimate a continuous probability density function. \nIncrease the number of PPNs to be filtered!\n")
    except ValueError as v:
        logging.info(f"Cannot plot the data!\nValueError encountered while performing KDE: \n{v}. \nIncrease the number of PPNs to be filtered!\n")
        print(f"Cannot plot the data!\nValueError encountered while performing KDE: \n{v}. \nIncrease the number of PPNs to be filtered!\n")
        
def create_plots(results_df, weights_word, weights_textline, plot_file, histogram_info, general_title):
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
                 plot_colors["word"]["mean_density"])      

    plot_histogram(axs[0, 2], results_df["standard_deviation_word"], weights_word, bins, 
                   "Standard Deviation of Word Confidence Scores", 
                   "Frequency", 
                   plot_colors["word"]["std"], histogram_info=histogram_info)
    plot_density(axs[0, 3], results_df["standard_deviation_word"], weights_word, 
                 "Standard Deviation of Word Confidence Scores", 
                 "Density", 
                 plot_colors["word"]["std_density"])

    plot_histogram(axs[1, 0], results_df["mean_textline"], weights_textline, bins, 
                   "Mean of Textline Confidence Scores", 
                   "Frequency", 
                   plot_colors["textline"]["mean"], histogram_info=histogram_info)
    plot_density(axs[1, 1], results_df["mean_textline"], weights_textline, 
                 "Mean of Textline Confidence Scores", 
                 "Density", 
                 plot_colors["textline"]["mean_density"])      

    plot_histogram(axs[1, 2], results_df["standard_deviation_textline"], weights_textline, bins, 
                   "Standard Deviation of Textline Confidence Scores", 
                   "Frequency", 
                   plot_colors["textline"]["std"], histogram_info=histogram_info)
    plot_density(axs[1, 3], results_df["standard_deviation_textline"], weights_textline, 
                 "Standard Deviation of Textline Confidence Scores", 
                 "Density", 
                 plot_colors["textline"]["std_density"])

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.945, hspace=0.17)
    plt.savefig(plot_file)
    plt.close()

@contextmanager
def load_csv(csv_file):
    with open(csv_file, 'r') as f:
        yield csv.reader(f)
        
def load_csv_to_list(csv_file):
    with open(csv_file, 'r') as f:
        return list(csv.reader(f))
        
def create_weighted_means_barplot(plot_df, label_col, title, filename, ha):
    x = np.arange(len(plot_df))
    width = 0.35
    plt.figure(figsize=(max(13, len(plot_df) * 0.5), 7))
    plt.bar(x - width / 2, plot_df['Weighted_Mean_Word'], width, label='Weighted Mean Word', color='skyblue')
    plt.bar(x + width / 2, plot_df['Weighted_Mean_Textline'], width, label='Weighted Mean Textline', color='salmon')
    plt.xlabel(label_col, fontsize=13)
    plt.ylabel('Confidence Score', fontsize=13)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(x, plot_df[label_col], rotation=45, ha=ha)
    plt.tick_params(axis='x', length=10)
    plt.ylim(0, 1)
    plt.xlim(-0.5, len(plot_df))
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
    labels, counts = list(labels)[::-1], list(counts)[::-1]

    plt.figure(figsize=(100, 150))
    bars = plt.barh(labels, counts, color=plt.cm.tab10.colors)  # type: ignore
    plt.ylabel(ylabel, fontsize=130 * sizefactor * fontsize_scale)
    plt.xlabel('Counts', fontsize=130 * sizefactor * fontsize_scale)
    plt.title(title, fontsize=150 * sizefactor * fontsize_scale, fontweight='bold')
    plt.xticks(fontsize=100 * sizefactor * fontsize_scale)
    plt.yticks(fontsize=100 * sizefactor * fontsize_scale)
    plt.grid(axis='x', linestyle='--', alpha=1.0)
    plt.ylim(-0.5, len(labels) - 0.5)
    
    # Add data labels next to bars
    for bar in bars:
        xval = bar.get_width()
        plt.text(xval, bar.get_y() + bar.get_height() / 2, str(int(xval)), ha='left', va='center', fontsize=100 * sizefactor * fontsize_scale)

    plt.tight_layout(pad=2.0)
    plt.savefig(filename)
    plt.close()
    
def process_weighted_means(data_dict, label_name, filename_prefix):
    labels = []
    mean_words = []
    mean_textlines = []

    for label, data in data_dict.items():
        if not data["mean_word"] or not data["weight_word"]:
            continue
        wm_word = weighted_mean(data["mean_word"], data["weight_word"])
        wm_textline = weighted_mean(data["mean_textline"], data["weight_textline"])
        labels.append(label)
        mean_words.append(wm_word)
        mean_textlines.append(wm_textline)

    df = pd.DataFrame({
        label_name: labels,
        'Weighted_Mean_Word': mean_words,
        'Weighted_Mean_Textline': mean_textlines
    }).dropna()
    df.to_csv(f"{filename_prefix}_weighted_mean_scores.csv", index=False)

    create_weighted_means_barplot(
        df,
        label_col=label_name,
        title=f'{label_name}-based Weighted Means of Word and Textline Confidence Scores',
        filename=f"{filename_prefix}_weighted_mean_scores.png",
        ha='right'
    )
 
def genre_evaluation(metadata_df, results_df, use_threshold=False):
    matching_ppn_mods = results_df["ppn"].unique()
    filtered_genres = metadata_df[metadata_df["PPN"].isin(matching_ppn_mods)]

    # Create dicts for fast access
    ppn_to_genres_raw = filtered_genres.groupby("PPN")["genre-aad"].apply(list).to_dict()
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
        result_entry = ppn_to_results.get(ppn)
        if result_entry is None:
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
                subgenre_weighted_data[sub]["mean_word"].append(result_entry["mean_word"])
                subgenre_weighted_data[sub]["weight_word"].append(result_entry["weight_word"])
                subgenre_weighted_data[sub]["mean_textline"].append(result_entry["mean_textline"])
                subgenre_weighted_data[sub]["weight_textline"].append(result_entry["weight_textline"])

            for genre in set(genres): # Avoid duplicates
                if genre in counted_genres:
                    continue

                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                counted_genres.add(genre)

                mean_word = result_entry["mean_word"]
                mean_textline = result_entry["mean_textline"]
                weight_word = result_entry["weight_word"]
                weight_textline = result_entry["weight_textline"]

                if genre not in genre_weighted_data:
                    genre_weighted_data[genre] = {
                        "mean_word": [], "weight_word": [],
                        "mean_textline": [], "weight_textline": []
                    }

                genre_weighted_data[genre]["mean_word"].append(mean_word)
                genre_weighted_data[genre]["weight_word"].append(weight_word)
                genre_weighted_data[genre]["mean_textline"].append(mean_textline)
                genre_weighted_data[genre]["weight_textline"].append(weight_textline)

    logging.info(f"\nNumber of PPNs: {len(matching_ppn_mods)}")
    print(f"\nNumber of PPNs: {len(matching_ppn_mods)}")

    logging.info(f"Number of PPNs with one genre: {count_single_genres}")
    print(f"Number of PPNs with one genre: {count_single_genres}")

    logging.info(f"Number of PPNs with more than one genre: {count_multiple_genres}")
    print(f"Number of PPNs with more than one genre: {count_multiple_genres}")

    all_genres_reduced = set(genre_counts.keys())
    logging.info(f"\nNumber of all unique genres (without subgenres): {len(all_genres_reduced)}")
    print(f"\nNumber of all unique genres (without subgenres): {len(all_genres_reduced)}")

    genre_counts_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
    genre_counts_df_sorted = genre_counts_df.sort_values(by='Count', ascending=False)
    genre_counts_df_sorted.to_csv("genre_publications.csv", index=False)

    if not genre_counts_df.empty:
        logging.info("\nUnique genres and their counts:\n")
        logging.info(genre_counts_df_sorted.to_string(index=False))
        print("\nUnique genres and their counts:\n")
        print(genre_counts_df_sorted.to_string(index=False))

    subgenre_counts_df = pd.DataFrame(list(subgenre_counts.items()), columns=['Subgenre', 'Count'])
    subgenre_counts_df_sorted = subgenre_counts_df.sort_values(by='Count', ascending=False)
    
    logging.info(f"\nNumber of all unique subgenres: {len(subgenre_counts_df_sorted)}")
    print(f"\nNumber of all unique subgenres: {len(subgenre_counts_df_sorted)}")
    
    if not subgenre_counts_df.empty:    
        subgenre_counts_df_sorted.to_csv("subgenre_publications.csv", index=False)        
        logging.info("\nUnique subgenres and their counts:\n")
        logging.info(subgenre_counts_df_sorted.to_string(index=False))
        print("\nUnique subgenres and their counts:\n")
        print(subgenre_counts_df_sorted.to_string(index=False))
        
        genre_subgenre_summary = []

        for genre, subgenre_dict in genre_to_subgenre_counts.items():
            subgenre_str = "; ".join([f"{sub} ({count})" for sub, count in sorted(subgenre_dict.items(), key=lambda x: -x[1])])
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
        
        subgenres, sub_counts = zip(*subgenre_counts_df_sorted.values)
        create_publication_count_horizontal_barplot(
            labels=subgenres,
            counts=sub_counts,
            title='Counts of Unique Subgenres',
            ylabel='Subgenres',
            filename='subgenre_publications.png'
        )

    sorted_genre_counts = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)

    if sorted_genre_counts:
        _, highest_count = sorted_genre_counts[0]
        plot_threshold = highest_count * 0.04
    else:
        logging.info("No genre available to calculate the threshold.")
        print("No genre available to calculate the threshold.")
        plot_threshold = 0
    
    if use_threshold:
        # Filter genres by threshold
        filtered_genre_counts = [(genre, count) for genre, count in sorted_genre_counts if count > plot_threshold]
    else:
        # Include all genres
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

        process_weighted_means(genre_weighted_data, label_name='Genre', filename_prefix='genre')
        
        if len(subgenre_counts) >= 1:
            process_weighted_means(subgenre_weighted_data, label_name='Subgenre', filename_prefix='subgenre')
            
            # Flatten and sort within genre in order to plot the genre-subgenre combinations
            flattened_labels = []
            flattened_counts = []
            for genre in sorted(genre_to_subgenre_counts.keys()):
                subgenre_dict = genre_to_subgenre_counts[genre]
                
                # Use descending order
                sorted_subgenres = sorted(subgenre_dict.items(), key=lambda x: -x[1])
                for subgenre, count in sorted_subgenres:
                    flattened_labels.append(f"{genre}: {subgenre}")
                    flattened_counts.append(count)

            if flattened_labels and flattened_counts:
                create_publication_count_horizontal_barplot(
                    labels=flattened_labels,
                    counts=flattened_counts,
                    title='Counts of Genre-Subgenre Combinations',
                    ylabel='Genre: Subgenre',
                    filename='genre_subgenre_combinations.png'
                )
        
def dates_evaluation(metadata_df, results_df):
    matching_ppn_mods = results_df["ppn"].unique()
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

        plot_weighted_means_barplot(
            plot_df,
            label_col='Year',
            title='Yearly Weighted Means of Word and Textline Confidence Scores',
            filename=f"date_range_{min_year}-{max_year}_yearly_weighted_means.png",
            ha='center'
        )
        
    except ValueError as e:
        logging.info(f"Invalid publication dates: {e}")
        print(f"Invalid publication dates.")
        return
    
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
    for directory in [parent_dir, gt_dir, ocr_dir]:
        if not os.path.exists(directory):
            logging.info(f"Directory does not exist: {directory}")
            print(f"Directory does not exist: {directory}")
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
                command_string = f"ocrd-dinglehopper -I {gt_dir},{ocr_dir} -O {report_dir}"
                command_list = command_string.split()
                try:
                    subprocess.run(command_list, check=True, capture_output=True, text=True, shell=False)
                except subprocess.CalledProcessError as e:
                    logging.info(f"Failed to run command in {ppn_name}. Exit code: {e.returncode}, Error: {e.stderr}")
                    print(f"Failed to run command in {ppn_name}. Exit code: {e.returncode}, Error: {e.stderr}")
                # Change to the parent directory
                os.chdir(parent_dir)
                progbar.update(1)
    progbar.close()
    
def generate_error_rates(parent_dir_error, report_dir_error, error_rates_filename):
    for directory in [parent_dir_error, report_dir_error]:
        if not os.path.exists(directory):
            logging.info(f"Directory does not exist: {directory}")
            print(f"Directory does not exist: {directory}")
            return
    
    data = []
    valid_directory_count = 0
    os.chdir(parent_dir_error)
    for ppn_name in os.listdir():
        full_path = os.path.join(parent_dir_error, ppn_name)
        if os.path.isdir(full_path) and ppn_name.startswith("PPN"):
            valid_directory_count += 1

    with tqdm(total=valid_directory_count) as progbar:
        for ppn_name in os.listdir():
            full_path = os.path.join(parent_dir_error, ppn_name)

            if os.path.isdir(full_path) and ppn_name.startswith("PPN"):
                progbar.set_description(f"Processing directory: {ppn_name}")
                eval_dir = os.path.join(full_path, report_dir_error)

                if os.path.exists(eval_dir) and os.path.isdir(eval_dir):
                    for json_file in os.listdir(eval_dir):
                        if json_file.endswith(".json"):
                            json_path = os.path.join(eval_dir, json_file)
                            
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
                progbar.update(1)
    progbar.close()

    error_rates_df = pd.DataFrame(data)
    error_rates_df.sort_values(by='ppn_page', ascending=True, inplace=True)
    logging.info("\nResults:\n")
    logging.info(error_rates_df)
    print("\nResults:\n")
    print(error_rates_df)
    os.chdir(os.pardir)
    error_rates_df.to_csv(error_rates_filename, index=False)
    
def merge_csv(conf_df, error_rates_df, wcwer_filename):
    for filename in [conf_df, error_rates_df]:
        if not os.path.exists(filename):
            logging.info(f"File does not exist: {filename}")
            print(f"File does not exist: {filename}")
            return
    
    ppn_conf_df = pd.DataFrame(load_csv_to_list(conf_df)[1:], columns=["ppn", "ppn_page", "mean_word", "median_word", "standard_deviation_word", "mean_textline", "median_textline", "standard_deviation_textline", "weight_word", "weight_textline"])
    ppn_error_rates_df = pd.DataFrame(load_csv_to_list(error_rates_df)[1:], columns=["ppn", "ppn_page", "gt", "ocr", "cer", "wer", "n_characters", "n_words"])
    ppn_error_rates_df.drop(columns=["ppn"], inplace=True)
    ppn_conf_df = ppn_conf_df[ppn_conf_df['ppn_page'].isin(ppn_error_rates_df['ppn_page'])]
    
    # Merge confidence scores data with the error rates data
    wcwer_df = pd.merge(ppn_conf_df, ppn_error_rates_df, on='ppn_page', how='inner')
    wcwer_df.sort_values(by='ppn_page', ascending=True, inplace=True)
    logging.info("\nResults:\n")
    logging.info(wcwer_df)
    print("\nResults:\n")
    print(wcwer_df)
    wcwer_df.to_csv(wcwer_filename, index=False)
    
def plot_wer_vs_wc(wcwer_csv, plot_filename):
    if not os.path.exists(wcwer_csv):
            logging.info(f"File does not exist: {wcwer_csv}")
            print(f"File does not exist: {wcwer_csv}")
            return
            
    wcwer_df = pd.DataFrame(load_csv_to_list(wcwer_csv)[1:], columns=["ppn", "ppn_page", "mean_word", "median_word", "standard_deviation_word", "mean_textline", "median_textline", "standard_deviation_textline", "weight_word", "weight_textline", "gt", "ocr", "cer", "wer", "n_characters", "n_words"])
    
    wcwer_df['mean_word'] = pd.to_numeric(wcwer_df['mean_word'])
    wcwer_df['wer'] = pd.to_numeric(wcwer_df['wer'])
    wcwer_df.sort_values(by='mean_word', ascending=True, inplace=True)
    
    ppn_pages_count = wcwer_df["ppn_page"].nunique()
    plt.figure(figsize=(ppn_pages_count * 0.1, ppn_pages_count * 0.1)) 
    plt.scatter(wcwer_df["mean_word"], wcwer_df["wer"], color='blue', marker='x', s=100)
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
    
def plot_wer_vs_wc_interactive(wcwer_csv_inter, plot_filename_inter):
    if not os.path.exists(wcwer_csv_inter):
            logging.info(f"File does not exist: {wcwer_csv_inter}")
            print(f"File does not exist: {wcwer_csv_inter}")
            return
            
    wcwer_df = pd.DataFrame(load_csv_to_list(wcwer_csv_inter)[1:], columns=["ppn", "ppn_page", "mean_word", "median_word", "standard_deviation_word", "mean_textline", "median_textline", "standard_deviation_textline", "weight_word", "weight_textline", "gt", "ocr", "cer", "wer", "n_characters", "n_words"])
    
    wcwer_df['mean_word'] = pd.to_numeric(wcwer_df['mean_word'])
    wcwer_df['wer'] = pd.to_numeric(wcwer_df['wer'])
    
    fig = px.scatter(wcwer_df, x="mean_word", y="wer", title="WER(WC)", labels={'mean_word': 'Mean Word Confidence Score (WC)', 'wer': 'Word Error Rate (WER)'}, template='plotly_white', hover_name="ppn_page")
    
    # Show information about the PPN_PAGE on hover
    fig.update_traces(marker=dict(size=10, color='blue', line=dict(width=2, color='DarkSlateGrey')),
                      hovertemplate='PPN Page: %{hovertext}<br>Mean Word Confidence: %{x}<br>WER: %{y}<extra></extra>',
                      hovertext=wcwer_df['ppn_page'])

    fig.update_xaxes(range=[-0.01, 1.01])
    fig.update_yaxes(range=[-0.01, 1.01])
    fig.update_layout(title=dict(text='WER(WC)', x=0.5, xanchor='center'))
    pyo.plot(fig, filename=plot_filename_inter, auto_open=False)


def generate_dataframes(
    csv_files: list[str],
    metadata_csv: str,
    check_value_errors: bool = False,
    check_duplicates: bool = False,
    check_raw_genres: bool = False,
    aggregate_mode: str = "ppn_page",
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
                        all_results.append([ppn, ppn_page, mean_word, median_word, standard_deviation_word, mean_textline, median_textline, standard_deviation_textline, weight_word, weight_textline])
                                               
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

            aggregated_results.append([
                ppn, mean_word, median_word, std_word,
                mean_textline, median_textline, std_textline,
                total_word_weight, total_textline_weight
            ])

        results_df = pd.DataFrame(aggregated_results, columns=[
            "ppn", "mean_word", "median_word", "standard_deviation_word",
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
    use_best_mean_word_confs_unique=False,
    use_worst_mean_word_confs_unique=False,
    num_best_mean_word_confs_unique=1,
    num_worst_mean_word_confs_unique=1,
    use_best_mean_textline_confs_unique=False,
    use_worst_mean_textline_confs_unique=False,
    num_best_mean_textline_confs_unique=1,
    num_worst_mean_textline_confs_unique=1,
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
    aggregate_mode='ppn_page'
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
    
    if search_ppn:
        results_df = results_df[results_df["ppn"].isin(
        metadata_df.loc[(metadata_df["PPN"].astype(str) == search_ppn), "PPN"])] # type: ignore
    
    if search_date is not None: # "is not None" enables zero as input
        results_df = results_df[results_df["ppn"].isin(
        metadata_df.loc[(metadata_df["publication_date"].astype(int) == search_date), "PPN"])] # type: ignore
    
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
        if mean_word_confs_range[0] == 0:
            results_df = results_df[
                (results_df['mean_word'] >= mean_word_confs_range[0]) &
                (results_df['mean_word'] <= mean_word_confs_range[1])]
        else:
            results_df = results_df[
                (results_df['mean_word'] > mean_word_confs_range[0]) &
                (results_df['mean_word'] <= mean_word_confs_range[1])]
            
    if mean_textline_confs_range:
        results_df = results_df.sort_values(by='mean_textline', ascending=True)
        if mean_textline_confs_range[0] == 0:
            results_df = results_df[
                (results_df['mean_textline'] >= mean_textline_confs_range[0]) &
                (results_df['mean_textline'] <= mean_textline_confs_range[1])]
        else:
            results_df = results_df[
                (results_df['mean_textline'] > mean_textline_confs_range[0]) &
                (results_df['mean_textline'] <= mean_textline_confs_range[1])]
            
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
        
    if use_best_mean_word_confs_unique:
        results_df = results_df.sort_values(by='mean_word', ascending=False)
        best_unique_ppns = results_df['ppn'].drop_duplicates().head(num_best_mean_word_confs_unique)
        results_df = results_df[results_df['ppn'].isin(best_unique_ppns)]
    elif use_worst_mean_word_confs_unique:
        worst_unique_ppns = results_df['ppn'].drop_duplicates().head(num_worst_mean_word_confs_unique)
        results_df = results_df[results_df['ppn'].isin(worst_unique_ppns)]
        
    if use_best_mean_textline_confs_unique:
        results_df = results_df.sort_values(by='mean_textline', ascending=False)
        best_unique_ppns = results_df['ppn'].drop_duplicates().head(num_best_mean_textline_confs_unique)
        results_df = results_df[results_df['ppn'].isin(best_unique_ppns)]
    elif use_worst_mean_textline_confs_unique:
        results_df = results_df.sort_values(by='mean_textline', ascending=True)
        worst_unique_ppns = results_df['ppn'].drop_duplicates().head(num_worst_mean_textline_confs_unique)
        results_df = results_df[results_df['ppn'].isin(worst_unique_ppns)]
        
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
                filtered_results_df = results_df[['ppn', 'mean_word', 'mean_textline', 'weight_word', 'weight_textline']]
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

    if aggregate_mode == 'ppn_page':
        create_plots(results_df, None, None, plot_file=plot_file, histogram_info=histogram_info, general_title="Analysis of Confidence Scores per Page")
        create_plots(results_df, weights_word=results_df["weight_word"], weights_textline=results_df["weight_textline"], plot_file=plot_file_weighted, histogram_info=histogram_info, general_title="Analysis of Confidence Scores per Page (Weighted)")
    elif aggregate_mode == 'ppn':
        create_plots(results_df, None, None, plot_file=plot_file, histogram_info=histogram_info, general_title="Analysis of Confidence Scores per PPN")
        create_plots(results_df, weights_word=results_df["weight_word"], weights_textline=results_df["weight_textline"], plot_file=plot_file_weighted, histogram_info=histogram_info, general_title="Analysis of Confidence Scores per PPN (Weighted)")
    
        
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
        
