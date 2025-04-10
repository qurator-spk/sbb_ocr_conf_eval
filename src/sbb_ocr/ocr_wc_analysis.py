from contextlib import contextmanager
import csv
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

def plot_histogram(ax, data, bins, title, xlabel, ylabel, color):
    ax.hist(data, bins=bins, color=color, edgecolor="black", alpha=0.6, density=False)
    ax.set_title(title)
    ax.ticklabel_format(style="plain")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis="y", alpha=0.75)
    
def plot_density(ax, data, title, xlabel, ylabel, density_color, legend_loc):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    
    try:
        kde = gaussian_kde(data)
        x_range = np.linspace(0, 1, 100)
        density_values = kde(x_range)
        max_kde_density = np.max(density_values) 
        round_up_half_integer = (int(max_kde_density * 2) + (max_kde_density * 2 % 1 > 0)) / 2  
        ax.set_ylim(0, round_up_half_integer)
        ax.set_yticks(np.arange(0, round_up_half_integer + 0.01, 0.5))

        mean_value = np.mean(data)
        median_value = np.median(data)
        quantiles = np.quantile(data, [0.25, 0.75])

        ax.axvline(mean_value, color="black", linestyle="solid", linewidth=1, label="Mean")
        ax.axvline(quantiles[0], color="black", linestyle="dashed", linewidth=1, label="Q1: 25%")
        ax.axvline(median_value, color="black", linestyle="dotted", linewidth=1, label="Q2: 50% (Median)")
        ax.axvline(quantiles[1], color="black", linestyle="dashdot", linewidth=1, label="Q3: 75%")

        ax.plot(x_range, density_values, color=density_color, lw=2)
        ax.legend(loc=legend_loc)
        
    except LinAlgError as e:
        logging.info(f"Cannot plot the data!\nLinAlgError encountered while performing KDE: \n{e}. \nThe data does not have enough variation in its dimensions to accurately estimate a continuous probability density function. \nIncrease the number of PPNs to be filtered!\n")
        print(f"Cannot plot the data!\nLinAlgError encountered while performing KDE: \n{e}. \nThe data does not have enough variation in its dimensions to accurately estimate a continuous probability density function. \nIncrease the number of PPNs to be filtered!\n")
    except ValueError as v:
        logging.info(f"Cannot plot the data!\nValueError encountered while performing KDE: \n{v}. \nIncrease the number of PPNs to be filtered!\n")
        print(f"Cannot plot the data!\nValueError encountered while performing KDE: \n{v}. \nIncrease the number of PPNs to be filtered!\n")

@contextmanager
def load_csv(csv_file):
    with open(csv_file, 'r') as f:
        yield csv.reader(f)
        
def load_csv_to_list(csv_file):
    with open(csv_file, 'r') as f:
        return list(csv.reader(f))

def genre_evaluation(metadata_df, results_df, replace_subgenres=True):
    matching_ppn_mods = results_df["ppn"].unique()
    filtered_genres = metadata_df[metadata_df["PPN"].isin(matching_ppn_mods)]
    all_genres_raw = set(filtered_genres["genre-aad"].tolist())
    
    genre_counts = {}
    count_multiple_genres = 0
    count_single_genres = 0
    for ppn in matching_ppn_mods:
        current_genres_raw = filtered_genres[filtered_genres["PPN"] == ppn]["genre-aad"]
        counted_genres = set()
        
        for genre_raw in current_genres_raw:
            genres_json = genre_raw.replace('{', '[').replace('}', ']').replace("'", '"')
            if not genres_json:
                continue
            
            genres = json.loads(genres_json)

            if replace_subgenres:
                genres = [x.split(':')[0] if ':' in x else x.split('.')[0] for x in genres]
            
            # Count each genre for this PPN
            if len(genres) > 1:
                count_multiple_genres += 1
            elif len(genres) == 1:
                count_single_genres += 1
                
            for genre in set(genres):  # Avoid duplicates
                if genre in counted_genres:
                    continue 
                
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                counted_genres.add(genre)
    
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
    
    logging.info("\nUnique genres and their counts:\n")
    logging.info(genre_counts_df_sorted.to_string(index=False))
    print("\nUnique genres and their counts:\n")
    print(genre_counts_df_sorted.to_string(index=False))

    sorted_genre_counts = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_genre_counts_descending = sorted(genre_counts.items(), key=lambda x: x[1], reverse=False)

    if sorted_genre_counts:
        highest_genre, highest_count = sorted_genre_counts[0]  # Get the genre with the highest count
        plot_threshold = highest_count * 0.04
    else:
        logging.info("No genre available to calculate the threshold.")
        print("No genre available to calculate the threshold.")
        plot_threshold = 0

    # Filter genres by the threshold
    filtered_genre_counts = [(genre, count) for genre, count in sorted_genre_counts_descending if count > plot_threshold]

    if not filtered_genre_counts:
        logging.info("No genre exceeds the threshold.")
        print("No genre exceeds the threshold.")
    else:
        genres, counts = zip(*filtered_genre_counts)

        plt.figure(figsize=(100, 150))
        bars = plt.barh(genres, counts, color=plt.cm.tab10.colors)
        plt.ylabel('Genres', fontsize=130)
        plt.xlabel('Counts', fontsize=130)
        plt.title('Counts of Unique Genres', fontsize=150)
        plt.xticks(fontsize=100)
        plt.yticks(fontsize=100)
        plt.grid(axis='x', linestyle='--', alpha=1.0)
        plt.ylim(-0.5, len(genres) - 0.5)

        # Adding data labels next to bars
        for bar in bars:
            xval = bar.get_width()
            plt.text(xval, bar.get_y() + bar.get_height()/2, int(xval), ha='left', va='center', fontsize=100)  # Display counts next to bars
        
        plt.tight_layout(pad=2.0)
        plt.savefig("bar_plot_of_all_genres.png")
        plt.close()
        
def dates_evaluation(metadata_df, results_df, replace_subgenres=True):
    matching_ppn_mods = results_df["ppn"].unique()
    metadata_df = metadata_df[metadata_df["PPN"].isin(matching_ppn_mods)]
    
    min_year = metadata_df["originInfo-publication0_dateIssued"].min()
    max_year = metadata_df["originInfo-publication0_dateIssued"].max()
    logging.info(f"\nEarliest year: {min_year}")
    logging.info(f"\nLatest year: {max_year}")
    print(f"\nEarliest year: {min_year}")
    print(f"\nLatest year: {max_year}")
    
    unique_years = metadata_df["originInfo-publication0_dateIssued"].unique()
    num_unique_years = len(unique_years)
    logging.info(f"\nNumber of unique years: {num_unique_years}")
    print(f"\nNumber of unique years: {num_unique_years}")
    
    year_counts = metadata_df["originInfo-publication0_dateIssued"].value_counts().sort_index()
    year_counts_df = year_counts.reset_index() 
    year_counts_df.columns = ['Year', 'Count']
    
    logging.info("\nUnique years and their counts:\n")
    logging.info(year_counts_df.to_string(index=False))
    print("\nUnique years and their counts:\n")
    print(year_counts_df.to_string(index=False))
    
    plt.figure(figsize=(max(30, num_unique_years * 0.25), 15))
    plt.bar(year_counts_df['Year'].astype(str), year_counts_df['Count'], color=plt.cm.tab10.colors, width=0.5)
    plt.title('Publication Counts per Year', fontsize=18)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=13)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xlim(-0.5, len(year_counts_df['Year']) - 0.5)
    plt.ylim(0.0, max(year_counts_df['Count']) + 0.01)
    if 800 > max(year_counts_df['Count']) >= 400:
        plt.yticks(np.arange(0, max(year_counts_df['Count']) + 1, 50))
    elif 400 > max(year_counts_df['Count']) > 30:
        plt.yticks(np.arange(0, max(year_counts_df['Count']) + 1, 10))
    else:
        plt.yticks(np.arange(0, max(year_counts_df['Count']) + 1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.8)
    plt.tight_layout(pad=1.0)
    plt.savefig(f"date_range_{min_year}-{max_year}_bar_plot.png")
    plt.close()
    
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
                    result = subprocess.run(command_list, check=True, capture_output=True, text=True, shell=False)
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
    
    ppn_conf_df = pd.DataFrame(load_csv_to_list(conf_df)[1:], columns=["ppn", "ppn_page", "mean_word", "median_word", "standard_deviation_word", "mean_textline", "median_textline", "standard_deviation_textline"])
    ppn_error_rates_df = pd.DataFrame(load_csv_to_list(error_rates_df)[1:], columns=["ppn", "ppn_page", "gt", "ocr", "cer", "wer", "n_characters", "n_words"])
    ppn_error_rates_df.drop(columns=["ppn"], inplace=True)
    ppn_conf_df = ppn_conf_df[ppn_conf_df['ppn_page'].isin(ppn_error_rates_df['ppn_page'])]
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
            
    wcwer_df = pd.DataFrame(load_csv_to_list(wcwer_csv)[1:], columns=["ppn", "ppn_page", "mean_word", "median_word", "standard_deviation_word", "mean_textline", "median_textline", "standard_deviation_textline", "gt", "ocr", "cer", "wer", "n_characters", "n_words"])
    
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
            
    wcwer_df = pd.DataFrame(load_csv_to_list(wcwer_csv_inter)[1:], columns=["ppn", "ppn_page", "mean_word", "median_word", "standard_deviation_word", "mean_textline", "median_textline", "standard_deviation_textline", "gt", "ocr", "cer", "wer", "n_characters", "n_words"])
    
    wcwer_df['mean_word'] = pd.to_numeric(wcwer_df['mean_word'])
    wcwer_df['wer'] = pd.to_numeric(wcwer_df['wer'])
    
    fig = px.scatter(wcwer_df, x="mean_word", y="wer", title="WER(WC)", labels={'mean_word': 'Mean Word Confidence Score (WC)', 'wer': 'Word Error Rate (WER)'}, template='plotly_white', hover_name="ppn_page")

    fig.update_traces(marker=dict(size=10, color='blue', line=dict(width=2, color='DarkSlateGrey')),
                      hovertemplate='PPN Page: %{hovertext}<br>Mean Word Confidence: %{x}<br>WER: %{y}<extra></extra>',
                      hovertext=wcwer_df['ppn_page'])

    fig.update_xaxes(range=[-0.01, 1.01])
    fig.update_yaxes(range=[-0.01, 1.01])
    fig.update_layout(title=dict(text='WER(WC)', x=0.5, xanchor='center'))
    pyo.plot(fig, filename=plot_filename_inter, auto_open=False)
    
def plot_everything(csv_files : list[str], metadata_csv, search_genre, plot_file="statistics_results.jpg", replace_subgenres : bool = True,
                    year_start=None, year_end=None, 
                    use_top_ppns_word=False, use_bottom_ppns_word=False, num_top_ppns_word=1, num_bottom_ppns_word=1, 
                    use_top_ppns_textline=False, use_bottom_ppns_textline=False, num_top_ppns_textline=1, num_bottom_ppns_textline=1,
                    mean_word_start=None, mean_word_end=None, mean_textline_start=None, mean_textline_end=None, show_genre_evaluation=False, 
                    output=False, show_dates_evaluation=False, show_results=False,
                    use_best_mean_word_confs_unique=False, use_worst_mean_word_confs_unique=False, num_best_mean_word_confs_unique=1, num_worst_mean_word_confs_unique=1,
                    use_best_mean_textline_confs_unique=False, use_worst_mean_textline_confs_unique=False, num_best_mean_textline_confs_unique=1, num_worst_mean_textline_confs_unique=1,
                    use_best_mean_word_confs=False, use_worst_mean_word_confs=False, num_best_mean_word_confs=1, num_worst_mean_word_confs=1,
                    use_best_mean_textline_confs=False, use_worst_mean_textline_confs=False, num_best_mean_textline_confs=1, num_worst_mean_textline_confs=1,
                    parent_dir=None, conf_filename=None, use_logging=None, check_value_errors=False, check_duplicates=False, check_raw_genres=False):
    if use_logging:
        setup_logging("plot")
    
    for file in csv_files:
        if not os.path.exists(file):
            logging.info(f"File does not exist: {file}")
            print(f"File does not exist: {file}")
            return
        
    all_results = []
    value_error_pages = []
    with tqdm(total=len(csv_files)) as progbar:
        for ind, csv_file in enumerate(csv_files):
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
                            
                        mean_textline, median_textline, standard_deviation_textline = statistics(textline_confs)
                        mean_word, median_word, standard_deviation_word = statistics(word_confs)
                        all_results.append([ppn, ppn_page, mean_word, median_word, standard_deviation_word, mean_textline, median_textline, standard_deviation_textline])
                                               
            except csv.Error as e:
                logging.info(f"CSV error: {e} in file: {csv_file}. \nIncrease the CSV field size limit!")
                print(f"CSV error: {e} in file: {csv_file}. \nIncrease the CSV field size limit!")
                return
            progbar.update(1)
    progbar.close()
    
    results_df = pd.DataFrame(all_results, columns=["ppn", "ppn_page", "mean_word", "median_word", "standard_deviation_word", "mean_textline", "median_textline", "standard_deviation_textline"])        
    
    if "metadata" in metadata_csv:
        if not os.path.exists(metadata_csv):
            logging.info(f"File does not exist: {metadata_csv}")
            print(f"File does not exist: {metadata_csv}")
            return
        else:
            metadata_df = pd.DataFrame(load_csv_to_list(metadata_csv)[1:], columns=["PPN", "genre-aad", "originInfo-publication0_dateIssued"])
            
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
            metadata_df.loc[metadata_df["originInfo-publication0_dateIssued"].isin(["", "18XX"]), "originInfo-publication0_dateIssued"] = "2025"
            
            # Change all genres that are empty strings to "Unbekannt"
            metadata_df.loc[metadata_df["genre-aad"].isin([""]), "genre-aad"] = "{'Unbekannt'}"
            
            # Change the genre separation from slashes to commas
            metadata_df['genre-aad'] = metadata_df['genre-aad'].apply(lambda genre: "{" + genre.strip().strip("{ }").replace("  / ", "', '").replace(" / ", "', '") + "}")
            
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
                metadata_filtered = metadata_df[['PPN', 'originInfo-publication0_dateIssued', 'genre-aad']]
                filtered_df = filtered_df.merge(metadata_filtered, left_on='ppn', right_on='PPN')
                filtered_df.drop(columns=['PPN'], inplace=True)
                filtered_df = filtered_df.sort_values(by='ppn_page', ascending=True)
                filtered_df.to_csv("duplicates.csv", index=False)
            
    # Count the number of unique PPNs in the results dataframe
    all_ppns = results_df["ppn"].unique()
    
    if year_start is not None and year_end is not None: # "is not None" enables zero as input
        results_df = results_df[results_df["ppn"].isin(
        metadata_df.loc[
            (metadata_df["originInfo-publication0_dateIssued"].astype(int) >= year_start) &
            (metadata_df["originInfo-publication0_dateIssued"].astype(int) <= year_end),
            "PPN"])]
        results_df = results_df.sort_values(by='mean_word', ascending=True)
            
    if mean_word_start is not None and mean_word_end is not None:
        results_df = results_df.sort_values(by='mean_word', ascending=True)
        if mean_word_start == 0:
            results_df = results_df[
                (results_df['mean_word'] >= mean_word_start) &  # Include 0
                (results_df['mean_word'] <= mean_word_end)
            ]
        else:
            results_df = results_df[
                (results_df['mean_word'] > mean_word_start) &
                (results_df['mean_word'] <= mean_word_end)
            ]
            
    if mean_textline_start is not None and mean_textline_end is not None:
        results_df = results_df.sort_values(by='mean_textline', ascending=True)
        results_df = results_df[
            (results_df['mean_textline'] >= mean_textline_start) & 
            (results_df['mean_textline'] <= mean_textline_end)]
            
    if search_genre:
        # Escape special characters in the search_genre string
        escaped_genre = re.escape(search_genre)
        pattern = r"\{\s*[^}]*?\b" + escaped_genre + r"\b[^}]*?\}"
        results_df = results_df[results_df["ppn"].isin(metadata_df.loc[metadata_df["genre-aad"].str.match(pattern, na=False), "PPN"])]
        results_df = results_df.sort_values(by='mean_word', ascending=True)
        
    if use_top_ppns_word:
        results_df = results_df[((results_df["mean_word"] >= 0.95) & (results_df["mean_word"] <= 1.0))]
        results_df = results_df.sort_values(by='mean_word', ascending=False)
        results_df = results_df.head(num_top_ppns_word)
    elif use_bottom_ppns_word:
        results_df = results_df[((results_df["mean_word"] >= 0.0) & (results_df["mean_word"] <= 0.05))]
        results_df = results_df.sort_values(by='mean_word', ascending=True)
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
        results_df = results_df.sort_values(by='mean_word', ascending=True)
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
        results_df = results_df.sort_values(by='mean_word', ascending=True)
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
    
    if show_results:
        if len(results_df_unique) > 0:
            filtered_results_df = results_df[['ppn', 'ppn_page', 'mean_word', 'mean_textline']]
            metadata_filtered = metadata_df[['PPN', 'originInfo-publication0_dateIssued', 'genre-aad']]
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
    else:
        if "metadata" in metadata_csv:
            metadata_filtered = metadata_df[['PPN', 'originInfo-publication0_dateIssued', 'genre-aad']]
            results_df = results_df.merge(metadata_filtered, left_on='ppn', right_on='PPN')
            results_df.drop(columns=['PPN'], inplace=True)
            results_df_description = results_df.describe(include='all')
            logging.info("\nResults description: \n")
            logging.info(results_df_description)
            print("\nResults description: \n")
            print(results_df_description)
        else:
            logging.info("\nResults description: \n")
            logging.info(results_df.describe(include='all'))
            print("\nResults description: \n")
            print(results_df.describe(include='all'))
            
        if output:
            results_df = results_df.sort_values(by='genre-aad', ascending=True)
            results_df.to_csv(output, index=False)
            logging.info(f"\nSaved results to: {output.name}")
            print(f"\nSaved results to: {output.name}")
            output_desc = output.name.split(".")[0] + "_desc.csv" 
            results_df_description.to_csv(output_desc, index=False)
            logging.info(f"\nSaved results description to: {output_desc}")
            print(f"\nSaved results description to: {output_desc}")

        # Main plotting function  
        fig, axs = plt.subplots(2, 4, figsize=(20.0, 10.0))
        
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
        
        bins = np.arange(0, 1.05, 0.05)
        
        plot_histogram(axs[0, 0], results_df["mean_word"], bins, 
                                    "Mean Word Confidence Scores", 
                                    "Mean Word Confidence", "Frequency", 
                                    plot_colors["word"]["mean"])
                                    
        plot_density(axs[0, 1], results_df["mean_word"], 
                                    "Mean Word Confidence Scores", 
                                    "Mean Word Confidence", "Density", 
                                    plot_colors["word"]["mean_density"], legend_loc="upper left")      
        
        plot_histogram(axs[0, 2], results_df["standard_deviation_word"], bins, 
                                    "Standard Deviation Word Confidence Scores", 
                                    "Standard Deviation Word Confidence", "Frequency", 
                                    plot_colors["word"]["std"])
                                    
        plot_density(axs[0, 3], results_df["standard_deviation_word"], 
                                    "Standard Deviation Word Confidence Scores", 
                                    "Standard Deviation Word Confidence", "Density", 
                                    plot_colors["word"]["std_density"], legend_loc="upper right")
                                    
        plot_histogram(axs[1, 0], results_df["mean_textline"], bins, 
                                    "Mean Textline Confidence Scores", 
                                    "Mean Textline Confidence", "Frequency", 
                                    plot_colors["textline"]["mean"])
                                    
        plot_density(axs[1, 1], results_df["mean_textline"], 
                                    "Mean Textline Confidence Scores", 
                                    "Mean Textline Confidence", "Density", 
                                    plot_colors["textline"]["mean_density"], legend_loc="upper left")      
        
        plot_histogram(axs[1, 2], results_df["standard_deviation_textline"], bins, 
                                    "Standard Deviation Textline Confidence Scores", 
                                    "Standard Deviation Textline Confidence", "Frequency", 
                                    plot_colors["textline"]["std"])
                                    
        plot_density(axs[1, 3], results_df["standard_deviation_textline"], 
                                    "Standard Deviation Textline Confidence Scores", 
                                    "Standard Deviation Textline Confidence", "Density", 
                                    plot_colors["textline"]["std_density"], legend_loc="upper right")
        
        plt.tight_layout(pad=1.0)
        plt.savefig(plot_file)
        plt.close()
        
def evaluate_everything(parent_dir=None, gt_dir=None, ocr_dir=None, report_dir=None, parent_dir_error=None, report_dir_error=None, error_rates_filename=None,
                        use_logging=None, conf_df=None, error_rates_df=None, wcwer_filename=None, wcwer_csv=None, plot_filename=None, wcwer_csv_inter=None, plot_filename_inter=None):
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
        