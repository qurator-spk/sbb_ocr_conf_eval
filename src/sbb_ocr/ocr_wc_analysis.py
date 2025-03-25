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

csv.field_size_limit(10**9)  # Set the CSV field size limit

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
        print(f"Cannot plot the data!\nLinAlgError encountered while performing KDE: \n{e}. \nThe data does not have enough variation in its dimensions to accurately estimate a continuous probability density function. \nIncrease the number of PPNs to be filtered!\n")
    except ValueError as v:
        print(f"Cannot plot the data!\nValueError encountered while performing KDE: \n{v}. \nIncrease the number of PPNs to be filtered!\n")

@contextmanager
def load_csv(csv_file):
    with open(csv_file, 'r') as f:
        yield csv.reader(f)
        
def load_csv_to_list(csv_file):
    with open(csv_file, 'r') as f:
        return list(csv.reader(f))

def genre_evaluation(mods_info_df, results_df, replace_subgenres=True):
    matching_ppn_mods = results_df["ppn"].unique()
    filtered_genres = mods_info_df[mods_info_df["PPN"].isin(matching_ppn_mods)]

    all_genres_raw = set(filtered_genres["genre-aad"].tolist())
    print("\nNumber of all genres: ", len(all_genres_raw))
    all_genres = []
    for genre_raw in all_genres_raw:
        genres_json = genre_raw.replace('{', '[').replace('}', ']').replace("'", '"')
        if not genres_json:
            continue
        genres = json.loads(genres_json)
        if replace_subgenres:
            genres = [x.split(':')[0] if ':' in x else x.split('.')[0] for x in genres]

            if any(x in ["lit", "hist", "jur", "theol", "Ars", "moriendi"] for x in genres):
                continue

        all_genres += genres

    all_genres_reduced = set(all_genres)
    print("\nNumber of all genres (without subgenres): ", len(all_genres_reduced))

    genre_counts = {}
    for genre in all_genres:
        if genre in genre_counts:
            genre_counts[genre] += 1
        else:
            genre_counts[genre] = 1
            
    genre_counts_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
    genre_counts_df_sorted = genre_counts_df.sort_values(by='Count', ascending=False)

    print("\nUnique genres and their counts:\n")
    print(genre_counts_df_sorted.to_string(index=False))

    sorted_genre_counts = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_genre_counts_descending = sorted(genre_counts.items(), key=lambda x: x[1], reverse=False)

    if sorted_genre_counts:
        highest_genre, highest_count = sorted_genre_counts[0]  # Get the genre with the highest count
        plot_threshold = highest_count * 0.04
    else:
        print("No genre available to calculate the threshold.")
        plot_threshold = 0

    # Filter genres by the threshold
    filtered_genre_counts = [(genre, count) for genre, count in sorted_genre_counts_descending if count > plot_threshold]

    if not filtered_genre_counts:
        print("No genre exceeds the threshold.")
    else:
        genres, counts = zip(*filtered_genre_counts)

        plt.figure(figsize=(100, 150))
        bars = plt.barh(genres, counts, color=plt.cm.tab10.colors)
        plt.ylabel('Genres', fontsize=100)
        plt.xlabel('Counts', fontsize=100)
        plt.title('Counts of Unique Genres', fontsize=120)
        plt.xticks(fontsize=65)
        plt.yticks(fontsize=65)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.ylim(-0.5, len(genres) - 0.5)
        plt.xlim(0, 600)

        # Adding data labels next to bars
        for bar in bars:
            xval = bar.get_width()
            plt.text(xval, bar.get_y() + bar.get_height()/2, int(xval), ha='left', va='center', fontsize=65)  # Display counts next to bars

        plt.tight_layout(pad=2.0)
        plt.savefig("bar_plot_of_all_genres.png")
        plt.close()
        
def dates_evaluation(mods_info_df, results_df, replace_subgenres=True):
    matching_ppn_mods = results_df["ppn"].unique()
    mods_info_df = mods_info_df[mods_info_df["PPN"].isin(matching_ppn_mods)]
    unique_years = mods_info_df["originInfo-publication0_dateIssued"].unique()
    num_unique_years = len(unique_years)
    print(f"\nNumber of unique years: {num_unique_years}")
    
    year_counts = mods_info_df["originInfo-publication0_dateIssued"].value_counts().sort_index()
    year_counts_df = year_counts.reset_index() 
    year_counts_df.columns = ['Year', 'Count']

    print("\nUnique years and their counts:\n")
    print(year_counts_df.to_string(index=False))
    
    plt.figure(figsize=(30, 15))
    plt.bar(year_counts_df['Year'].astype(str), year_counts_df['Count'], color=plt.cm.tab10.colors, width=0.5)
    plt.title('Publication Counts per Year', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=13)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xticks(rotation=45)
    plt.xlim(-0.5, len(year_counts_df['Year']) - 0.5)
    plt.ylim(0.0, max(year_counts_df['Count']) + 0.01)
    plt.yticks(np.arange(0, max(year_counts_df['Count']) + 1, 1))
    plt.tight_layout(pad=1.0)
    plt.savefig("bar_plot_of_all_years.png")
    plt.close()

def plot_everything(csv_files : list[str], mods_info_csv, search_genre, plot_file="statistics_results.jpg", replace_subgenres : bool = True,
                    year_start=None, year_end=None, 
                    use_top_ppns_word=False, use_bottom_ppns_word=False, num_top_ppns_word=1, num_bottom_ppns_word=1, 
                    use_top_ppns_textline=False, use_bottom_ppns_textline=False, num_top_ppns_textline=1, num_bottom_ppns_textline=1,
                    mean_word_start=None, mean_word_end=None, mean_textline_start=None, mean_textline_end=None, show_genre_evaluation=False, 
                    output=False, show_dates_evaluation=False, show_results=False,
                    use_best_mean_word_confs=False, use_worst_mean_word_confs=False, num_best_mean_word_confs=1, num_worst_mean_word_confs=1,
                    use_best_mean_textline_confs=False, use_worst_mean_textline_confs=False, num_best_mean_textline_confs=1, num_worst_mean_textline_confs=1):
    for file in csv_files:
        if not os.path.exists(file):
            print(f"File does not exist: {file}")
            return

    if not os.path.exists(mods_info_csv):
        print(f"File does not exist: {mods_info_csv}")
        return
        
    all_results = []
    with tqdm(total=len(csv_files)) as progbar:
        for ind, csv_file in enumerate(csv_files):
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
                        mean_textline, median_textline, standard_deviation_textline = statistics(textline_confs)
                        mean_word, median_word, standard_deviation_word = statistics(word_confs)
                        ppn_page = f'{row[0]}_{row[1]}_{row[2]}'
                        ppn = f'{row[0]}'
                        all_results.append([ppn, ppn_page, mean_word, median_word, standard_deviation_word, mean_textline, median_textline, standard_deviation_textline])
                                               
            except csv.Error as e:
                print(f"CSV error: {e} in file: {csv_file}. \nIncrease the CSV field size limit!")
                break
            progbar.update(1)
    progbar.close()
    
    results_df = pd.DataFrame(all_results, columns=["ppn", "ppn_page", "mean_word", "median_word", "standard_deviation_word", "mean_textline", "median_textline", "standard_deviation_textline"])
    results_df_original = results_df.copy()
    results_df.to_csv("results1.csv", index=False)
    
    if "2024-11-27" in mods_info_csv:
        mods_info_df = pd.DataFrame(load_csv_to_list(mods_info_csv)[1:], columns=["accessCondition-use and reproduction", "accessCondition-restriction on access", "classification-ZVDD", "genre-aad", "identifier-purl", "identifier-vd17", "language_languageTerm", "location_physicalLocation", "location_shelfLocator", "name0_displayForm", "name0_namePart-family", "name0_namePart-given", "name0_role_roleTerm", "originInfo-digitization0_dateCaptured", "originInfo-digitization0_edition", "originInfo-digitization0_place_placeTerm", "originInfo-digitization0_publisher", "originInfo-publication0_dateIssued", "originInfo-publication0_place_placeTerm", "originInfo-publication0_publisher", "relatedItem-original_recordInfo_recordIdentifier", "titleInfo_subTitle", "titleInfo_title", "typeOfResource", "mets_fileSec_fileGrp-FULLTEXT-count", "mets_fileSec_fileGrp-DEFAULT-count", "mets_fileSec_fileGrp-PRESENTATION-count", "mets_fileSec_fileGrp-THUMBS-count", "mets_file", "identifier-RISMA2", "originInfo-production0_dateCreated", "originInfo-production0_edition", "identifier-KOPE", "originInfo-production0_place_placeTerm", "genre-sbb", "mets_fileSec_fileGrp-MAX-count", "mets_fileSec_fileGrp-MIN-count", "mets_fileSec_fileGrp-LOCAL-count", "language_scriptTerm", "name0_namePart", "relatedItem-host_recordInfo_recordIdentifier", "classification-sbb", "identifier-vd18", "identifier-ORIE", "originInfo-publication0_edition", "identifier-PPNanalog", "genre-wikidata", "identifier-vd16", "subject-EC1418_genre", "subject_name0_displayForm", "subject_name0_namePart-family", "subject_name0_namePart-given", "classification-ddc", "identifier-MMED", "relatedItem-original_recordInfo_recordIdentifier-dnb-ppn", "name0_namePart-termsOfAddress", "genre-marcgt", "identifier-zdb", "identifier-RISMA1", "identifier-GW", "identifier-doi", "classification-ark", "abstract", "accessCondition-embargo enddate", "titleInfo_partName", "identifier-KSTO", "identifier-ISSN", "genre", "identifier-EC1418"])
        
        mods_info_df["PPN"] = mods_info_df["mets_file"].apply(lambda x: x.split("/")[-1].split(".")[0])
        
        rows_to_drop_4_char = mods_info_df[mods_info_df["originInfo-publication0_dateIssued"].str.len() != 4].index
        mods_info_df.drop(index=rows_to_drop_4_char, inplace=True)
        
        rows_to_drop_XX = mods_info_df[~mods_info_df["originInfo-publication0_dateIssued"].str.isdigit()].index
        mods_info_df.drop(index=rows_to_drop_XX, inplace=True)
        
        mods_info_df["originInfo-publication0_dateIssued"] = pd.to_numeric(mods_info_df["originInfo-publication0_dateIssued"], errors="coerce")
        mods_info_df = mods_info_df.dropna(subset=["originInfo-publication0_dateIssued"])
        mods_info_df["originInfo-publication0_dateIssued"] = mods_info_df["originInfo-publication0_dateIssued"].astype(int)
        
        mods_info_df.to_csv("mods_info_df_2024-11-27_afterdrop.csv", index=False)
    
    elif "2024-09-06" in mods_info_csv:
        mods_info_df = pd.DataFrame(load_csv_to_list(mods_info_csv)[1:], columns=["PPN", "PPN", "accessCondition-embargo enddate", "accessCondition-restriction on access", "accessCondition-use and reproduction", "classification-ZVDD", "classification-ark", "classification-ddc", "classification-sbb", "genre", "genre-aad", "genre-sbb", "genre-wikidata", "identifier-vd16", "identifier-vd17", "identifier-vd18", "language_languageTerm", "language_scriptTerm", "mets_fileSec_fileGrp-FULLTEXT-count", "mets_fileSec_fileGrp-PRESENTATION-count", "originInfo-digitization0_dateCaptured", "originInfo-digitization0_publisher", "originInfo-production0_dateCreated", "originInfo-publication0_dateIssued", "originInfo-publication0_publisher", "recordInfo_recordIdentifier", "subject-EC1418_genre", "titleInfo_title", "typeOfResource", "vd", "vd16", "vd17", "vd18", "columns", "german", "druck"])
        
        mods_info_df.drop(columns=['PPN'], inplace=True)
        mods_info_df["PPN"] = mods_info_df["recordInfo_recordIdentifier"]

        mods_info_df["originInfo-publication0_dateIssued"] = pd.to_numeric(mods_info_df["originInfo-publication0_dateIssued"], errors="coerce")
        mods_info_df = mods_info_df.dropna(subset=["originInfo-publication0_dateIssued"])
        mods_info_df["originInfo-publication0_dateIssued"] = mods_info_df["originInfo-publication0_dateIssued"].astype(int)
        
        mods_info_df.to_csv("mods_info_df_2024-09-06_afterdrop.csv", index=False)
        
    elif "2025-03-07" in mods_info_csv:
        mods_info_df = pd.DataFrame(load_csv_to_list(mods_info_csv)[1:], columns=["PPN", "genre-aad", "originInfo-publication0_dateIssued"])
        
    elif "2025-03-19" in mods_info_csv:
        mods_info_df = pd.DataFrame(load_csv_to_list(mods_info_csv)[1:], columns=["PPN", "genre-aad", "originInfo-publication0_dateIssued"])
        
    elif "2025-03-24" in mods_info_csv:
        mods_info_df = pd.DataFrame(load_csv_to_list(mods_info_csv)[1:], columns=["PPN", "genre-aad", "originInfo-publication0_dateIssued"])
        
    elif "2025-03-25" in mods_info_csv:
        mods_info_df = pd.DataFrame(load_csv_to_list(mods_info_csv)[1:], columns=["PPN", "genre-aad", "originInfo-publication0_dateIssued"])  
    
    results_df = results_df[results_df["ppn"].isin(mods_info_df["PPN"])]
    results_df.to_csv("results2.csv", index=False)
    
    if year_start and year_end:
        results_df = results_df[results_df["ppn"].isin(
        mods_info_df.loc[
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) >= year_start) &
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) <= year_end),
            "PPN"])] 
            
    if mean_word_start and mean_word_end:
        results_df = results_df[
            (results_df['mean_word'] >= mean_word_start) & 
            (results_df['mean_word'] <= mean_word_end)]
            
    if mean_textline_start and mean_textline_end:
        results_df = results_df[
            (results_df['mean_textline'] >= mean_textline_start) & 
            (results_df['mean_textline'] <= mean_textline_end)]
            
    if search_genre:
        results_df = results_df[results_df["ppn"].isin(mods_info_df.loc[mods_info_df["genre-aad"].str.contains(search_genre, na=False), "PPN"])]
        
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
        
    if use_best_mean_word_confs:
        results_df = results_df.sort_values(by='mean_word', ascending=False)
        best_unique_ppns = results_df['ppn'].drop_duplicates().head(num_best_mean_word_confs)
        results_df = results_df[results_df['ppn'].isin(best_unique_ppns)]
    elif use_worst_mean_word_confs:
        results_df = results_df.sort_values(by='mean_word', ascending=True)
        worst_unique_ppns = results_df['ppn'].drop_duplicates().head(num_worst_mean_word_confs)
        results_df = results_df[results_df['ppn'].isin(worst_unique_ppns)]
        
    if use_best_mean_textline_confs:
        results_df = results_df.sort_values(by='mean_textline', ascending=False)
        best_unique_ppns = results_df['ppn'].drop_duplicates().head(num_best_mean_textline_confs)
        results_df = results_df[results_df['ppn'].isin(best_unique_ppns)]
    elif use_worst_mean_textline_confs:
        results_df = results_df.sort_values(by='mean_textline', ascending=True)
        worst_unique_ppns = results_df['ppn'].drop_duplicates().head(num_worst_mean_textline_confs)
        results_df = results_df[results_df['ppn'].isin(worst_unique_ppns)]
        
    results_df_unique = results_df["ppn"].unique()
        
    all_ppns = results_df_original["ppn"].unique()
    
    print(f"\nResults: {len(results_df_unique)} of {len(all_ppns)} PPNs contained in {len(csv_files)} CSV_FILES match the applied filter:\n")
    
    if show_results:
        if len(results_df_unique) > 0:
            filtered_results_df = results_df[['ppn', 'ppn_page', 'mean_word', 'mean_textline']]
            mods_info_filtered = mods_info_df[['PPN', 'originInfo-publication0_dateIssued', 'genre-aad']]
            filtered_results_df = filtered_results_df.merge(mods_info_filtered, left_on='ppn', right_on='PPN')
            filtered_results_df.drop(columns=['PPN'], inplace=True)
            
            print(filtered_results_df.to_string(index=False))
        else:
            print("\nNo PPNs found for the applied filters.")
        
    if show_genre_evaluation:
        genre_evaluation(mods_info_df, results_df)
        
    if show_dates_evaluation:
        dates_evaluation(mods_info_df, results_df)
    
    if results_df.empty:
        print("\nThere are no results matching the applied filters.")
    else:
        mods_info_filtered = mods_info_df[['PPN', 'originInfo-publication0_dateIssued', 'genre-aad']]
        results_df = results_df.merge(mods_info_filtered, left_on='ppn', right_on='PPN')
        results_df.drop(columns=['PPN'], inplace=True)
        results_df_description = results_df.describe(include='all')
        if output:
            results_df.to_csv(output, index=False)
            print(f"\nSaved results to: {output.name}")
            output_desc = output.name.split(".")[0] + "_desc.csv" 
            results_df_description.to_csv(output_desc, index=False)
            print(f"\nSaved results description to: {output_desc}")
            
        print("\nResults description: \n")
        print(results_df_description)

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
        
        plot_histogram(axs[0, 0], results_df["mean_word"], np.arange(0, 1.1, 0.05), 
                                    "Mean Word Confidence Scores", 
                                    "Mean Word Confidence", "Frequency", 
                                    plot_colors["word"]["mean"])
                                    
        plot_density(axs[0, 1], results_df["mean_word"], 
                                    "Mean Word Confidence Scores", 
                                    "Mean Word Confidence", "Density", 
                                    plot_colors["word"]["mean_density"], legend_loc="upper left")      
        
        plot_histogram(axs[0, 2], results_df["standard_deviation_word"], np.arange(0, 1.1, 0.05), 
                                    "Standard Deviation Word Confidence Scores", 
                                    "Standard Deviation Word Confidence", "Frequency", 
                                    plot_colors["word"]["std"])
                                    
        plot_density(axs[0, 3], results_df["standard_deviation_word"], 
                                    "Standard Deviation Word Confidence Scores", 
                                    "Standard Deviation Word Confidence", "Density", 
                                    plot_colors["word"]["std_density"], legend_loc="upper right")
                                    
        plot_histogram(axs[1, 0], results_df["mean_textline"], np.arange(0, 1.1, 0.05), 
                                    "Mean Textline Confidence Scores", 
                                    "Mean Textline Confidence", "Frequency", 
                                    plot_colors["textline"]["mean"])
                                    
        plot_density(axs[1, 1], results_df["mean_textline"], 
                                    "Mean Textline Confidence Scores", 
                                    "Mean Textline Confidence", "Density", 
                                    plot_colors["textline"]["mean_density"], legend_loc="upper left")      
        
        plot_histogram(axs[1, 2], results_df["standard_deviation_textline"], np.arange(0, 1.1, 0.05), 
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
        #plt.show()
