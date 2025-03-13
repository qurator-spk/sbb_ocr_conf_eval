from contextlib import contextmanager
import csv
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt  
from scipy.stats import gaussian_kde  
from tqdm import tqdm  
import json
from rich import print

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

@contextmanager
def load_csv(csv_file):
    with open(csv_file, 'r') as f:
        yield csv.reader(f)
        
def load_csv_to_list(csv_file):
    with open(csv_file, 'r') as f:
        return list(csv.reader(f))

def genre_evaluation(ppn_col, mods_info_df, results_df, replace_subgenres=True):
    matching_ppn_mods = results_df["ppn"].unique()

    filtered_genres = mods_info_df[mods_info_df[ppn_col].isin(matching_ppn_mods)]

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
    print("\nNumber of all genres (after reduction): ", len(all_genres_reduced))

    genre_counts = {}
    for genre in all_genres:
        if genre in genre_counts:
            genre_counts[genre] += 1
        else:
            genre_counts[genre] = 1

    sorted_genre_counts = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_genre_counts_descending = sorted(genre_counts.items(), key=lambda x: x[1], reverse=False)

    print("\nUnique genres and their counts:")
    for genre, count in sorted_genre_counts:
        print(f"{genre}: {count}")

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
        
def best_ppns(results_df, mods_info_df, num_best_ppns):
    best_ppns_df = results_df[((results_df["mean_word"] >= 0.95) & (results_df["mean_word"] <= 1.0)) & ((results_df["mean_textline"] >= 0.95) & (results_df["mean_textline"] <= 1.0))]
    best_ppn_unique = best_ppns_df["ppn"].unique()
    best_ppn_list = best_ppn_unique[:num_best_ppns]
    
    if len(best_ppn_list) > 0:
        filtered_best_ppns_df = best_ppns_df[['ppn', 'ppn_page', 'mean_word', 'mean_textline']]
        mods_info_filtered = mods_info_df[['PPN', 'originInfo-publication0_dateIssued', 'genre-aad']]
        filtered_best_ppns_df = filtered_best_ppns_df.merge(mods_info_filtered, left_on='ppn', right_on='PPN')
        filtered_best_ppns_df.drop(columns=['PPN'], inplace=True)
        print(f"\nList of {len(best_ppn_list)} PPNs found with mean word score & mean textline scores between 0.95 and 1.0:\n")
        print(filtered_best_ppns_df.to_string(index=False))
    else:
        print("\nThere are no PPNs with mean word score & mean textline scores between 0.95 and 1.0 for the applied filters.")
        
def worst_ppns(results_df, mods_info_df, num_worst_ppns):
    worst_ppns_df = results_df[((results_df["mean_word"] >= 0.0) & (results_df["mean_word"] <= 0.05)) & ((results_df["mean_textline"] >= 0.0) & (results_df["mean_textline"] <= 0.05))]
    worst_ppn_unique = worst_ppns_df["ppn"].unique()
    worst_ppn_list = worst_ppn_unique[:num_worst_ppns]
    
    if len(worst_ppn_list) > 0:
        filtered_worst_ppns_df = worst_ppns_df[['ppn', 'ppn_page', 'mean_word', 'mean_textline']]
        mods_info_filtered = mods_info_df[['PPN', 'originInfo-publication0_dateIssued', 'genre-aad']]
        filtered_worst_ppns_df = filtered_best_ppns_df.merge(mods_info_filtered, left_on='ppn', right_on='PPN')
        filtered_worst_ppns_df.drop(columns=['PPN'], inplace=True)
        print(f"\nList of {len(worst_ppn_list)} PPNs found with mean word score & mean textline scores between 0.0 and 0.05:\n")
        print(filtered_worst_ppns_df.to_string(index=False))
    else:
        print("\nThere are no PPNs with mean word score & mean textline scores between 0.0 and 0.05 for the applied filters.")
        
def mean_word_confs(results_df, mods_info_df, mean_word_start, mean_word_end):
    mean_word_confs_df = results_df[(results_df["mean_word"] >= mean_word_start) & (results_df["mean_word"] <= mean_word_end)]
    mean_word_confs_unique = mean_word_confs_df["ppn"].unique()
    
    if len(mean_word_confs_unique) > 0:
        filtered_mean_word_confs_df = mean_word_confs_df[['ppn', 'ppn_page', 'mean_word']]
        mods_info_filtered = mods_info_df[['PPN', 'originInfo-publication0_dateIssued', 'genre-aad']]
        filtered_mean_word_confs_df = filtered_mean_word_confs_df.merge(mods_info_filtered, left_on='ppn', right_on='PPN')
        filtered_mean_word_confs_df.drop(columns=['PPN'], inplace=True)
        print(f"\nList of {len(mean_word_confs_unique)} PPNs with a mean word score between {mean_word_start} and {mean_word_end}:\n")
        print(filtered_mean_word_confs_df.to_string(index=False))
    else:
        print(f"\nThere are no PPNs with a mean word score between {mean_word_start} and {mean_word_end} for the applied filters.")
        
def mean_textline_confs(results_df, mods_info_df, mean_textline_start, mean_textline_end):
    mean_textline_confs_df = results_df[(results_df["mean_textline"] >= mean_textline_start) & (results_df["mean_textline"] <= mean_textline_end)]
    mean_textline_confs_unique = mean_textline_confs_df["ppn"].unique()
    
    if len(mean_textline_confs_unique) > 0:
        filtered_mean_textline_confs_df = mean_textline_confs_df[['ppn', 'ppn_page', 'mean_textline']]
        mods_info_filtered = mods_info_df[['PPN', 'originInfo-publication0_dateIssued', 'genre-aad']]
        filtered_mean_textline_confs_df = filtered_mean_textline_confs_df.merge(mods_info_filtered, left_on='ppn', right_on='PPN')
        filtered_mean_textline_confs_df.drop(columns=['PPN'], inplace=True)
        print(f"\nList of {len(mean_textline_confs_unique)} PPNs with a mean word score between {mean_textline_start} and {mean_textline_end}:\n")
        print(filtered_mean_textline_confs_df.to_string(index=False))
    else:
        print(f"\nThere are no PPNs with a mean word score between {mean_textline_start} and {mean_textline_end} for the applied filters.")
        
def date_ranges(results_df, mods_info_df, year_start, year_end):
    date_range_df = results_df = results_df[results_df["ppn"].isin(mods_info_df.loc[(mods_info_df["originInfo-publication0_dateIssued"].astype(int) >= year_start) &
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) <= year_end), "PPN"])] 
    date_range_df_unique = date_range_df["ppn"].unique()
    
    if len(date_range_df_unique) > 0:
        filtered_date_range_df = date_range_df[['ppn', 'ppn_page', 'mean_word', 'mean_textline']]
        mods_info_filtered = mods_info_df[['PPN', 'originInfo-publication0_dateIssued', 'genre-aad']]
        filtered_date_range_df = filtered_date_range_df.merge(mods_info_filtered, left_on='ppn', right_on='PPN')
        filtered_date_range_df.drop(columns=['PPN'], inplace=True)
        print(f"\nList of {len(date_range_df_unique)} PPNs found in the date range between {year_start} and {year_end}:\n")
        print(filtered_date_range_df.to_string(index=False))
    else:
        print("\nThere are no PPNs in the date range between {year_start} and {year_end}.")

def plot_everything(csv_files : list[str], mods_info_csv, search_genre, plot_file="statistics_results.jpg", replace_subgenres : bool = True,
                    year_start=None, year_end=None, use_best_ppns=False, use_worst_ppns=False, num_best_ppns=50, num_worst_ppns=50, 
                    mean_word_start=None, mean_word_end=None, mean_textline_start=None, mean_textline_end=None):
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
    
    if "2024-11-27" in mods_info_csv:
        mods_info_df = pd.DataFrame(load_csv_to_list(mods_info_csv)[1:], columns=["accessCondition-use and reproduction", "accessCondition-restriction on access", "classification-ZVDD", "genre-aad", "identifier-purl", "identifier-vd17", "language_languageTerm", "location_physicalLocation", "location_shelfLocator", "name0_displayForm", "name0_namePart-family", "name0_namePart-given", "name0_role_roleTerm", "originInfo-digitization0_dateCaptured", "originInfo-digitization0_edition", "originInfo-digitization0_place_placeTerm", "originInfo-digitization0_publisher", "originInfo-publication0_dateIssued", "originInfo-publication0_place_placeTerm", "originInfo-publication0_publisher", "relatedItem-original_recordInfo_recordIdentifier", "titleInfo_subTitle", "titleInfo_title", "typeOfResource", "mets_fileSec_fileGrp-FULLTEXT-count", "mets_fileSec_fileGrp-DEFAULT-count", "mets_fileSec_fileGrp-PRESENTATION-count", "mets_fileSec_fileGrp-THUMBS-count", "mets_file", "identifier-RISMA2", "originInfo-production0_dateCreated", "originInfo-production0_edition", "identifier-KOPE", "originInfo-production0_place_placeTerm", "genre-sbb", "mets_fileSec_fileGrp-MAX-count", "mets_fileSec_fileGrp-MIN-count", "mets_fileSec_fileGrp-LOCAL-count", "language_scriptTerm", "name0_namePart", "relatedItem-host_recordInfo_recordIdentifier", "classification-sbb", "identifier-vd18", "identifier-ORIE", "originInfo-publication0_edition", "identifier-PPNanalog", "genre-wikidata", "identifier-vd16", "subject-EC1418_genre", "subject_name0_displayForm", "subject_name0_namePart-family", "subject_name0_namePart-given", "classification-ddc", "identifier-MMED", "relatedItem-original_recordInfo_recordIdentifier-dnb-ppn", "name0_namePart-termsOfAddress", "genre-marcgt", "identifier-zdb", "identifier-RISMA1", "identifier-GW", "identifier-doi", "classification-ark", "abstract", "accessCondition-embargo enddate", "titleInfo_partName", "identifier-KSTO", "identifier-ISSN", "genre", "identifier-EC1418"])
        
        mods_info_df["PPN"] = mods_info_df["mets_file"].apply(lambda x: x.split("/")[-1].split(".")[0])
        
        results_df = results_df[results_df["ppn"].isin(mods_info_df["PPN"])]
        
        genre_evaluation("PPN", mods_info_df, results_df)
        
        rows_to_drop_4_char = mods_info_df[mods_info_df["originInfo-publication0_dateIssued"].str.len() != 4].index
        mods_info_df.drop(index=rows_to_drop_4_char, inplace=True)
        
        rows_to_drop_XX = mods_info_df[~mods_info_df["originInfo-publication0_dateIssued"].str.isdigit()].index
        mods_info_df.drop(index=rows_to_drop_XX, inplace=True)
        
        mods_info_df["originInfo-publication0_dateIssued"] = pd.to_numeric(mods_info_df["originInfo-publication0_dateIssued"], errors="coerce")
        mods_info_df = mods_info_df.dropna(subset=["originInfo-publication0_dateIssued"])
        mods_info_df["originInfo-publication0_dateIssued"] = mods_info_df["originInfo-publication0_dateIssued"].astype(int)
    
    elif "2024-09-06" in mods_info_csv:
        mods_info_df = pd.DataFrame(load_csv_to_list(mods_info_csv)[1:], columns=["PPN", "PPN", "accessCondition-embargo enddate", "accessCondition-restriction on access", "accessCondition-use and reproduction", "classification-ZVDD", "classification-ark", "classification-ddc", "classification-sbb", "genre", "genre-aad", "genre-sbb", "genre-wikidata", "identifier-vd16", "identifier-vd17", "identifier-vd18", "language_languageTerm", "language_scriptTerm", "mets_fileSec_fileGrp-FULLTEXT-count", "mets_fileSec_fileGrp-PRESENTATION-count", "originInfo-digitization0_dateCaptured", "originInfo-digitization0_publisher", "originInfo-production0_dateCreated", "originInfo-publication0_dateIssued", "originInfo-publication0_publisher", "recordInfo_recordIdentifier", "subject-EC1418_genre", "titleInfo_title", "typeOfResource", "vd", "vd16", "vd17", "vd18", "columns", "german", "druck"])
        
        mods_info_df.drop(columns=['PPN'], inplace=True)
        
        mods_info_df["PPN"] = mods_info_df["recordInfo_recordIdentifier"]
        
        results_df = results_df[results_df["ppn"].isin(mods_info_df["PPN"])]

        genre_evaluation("PPN", mods_info_df, results_df)

        mods_info_df["originInfo-publication0_dateIssued"] = pd.to_numeric(mods_info_df["originInfo-publication0_dateIssued"], errors="coerce")
        mods_info_df = mods_info_df.dropna(subset=["originInfo-publication0_dateIssued"])
        mods_info_df["originInfo-publication0_dateIssued"] = mods_info_df["originInfo-publication0_dateIssued"].astype(int)
        
        ## Create merged_mods_info_df_2025-03-07.csv:
        
        #with open('PPN.list.2024-09-06', 'r') as file:
        #    lines = [line.strip() for line in file.readlines()]

        #ppn_list_df = pd.DataFrame(lines, columns=['PPN'])

        #filtered_mods_info_df = mods_info_df[mods_info_df['recordInfo_recordIdentifier'].isin(ppn_list_df['PPN'])]
        
        #merged_mods_info_df = pd.DataFrame()
        #merged_mods_info_df['PPN'] = ppn_list_df['PPN']
        #merged_mods_info_df = merged_mods_info_df.merge(filtered_mods_info_df[['recordInfo_recordIdentifier', 'genre-aad', 'originInfo-publication0_dateIssued']],
        #                        left_on='PPN', right_on='recordInfo_recordIdentifier', how='left')

        #merged_mods_info_df.drop(columns='recordInfo_recordIdentifier', inplace=True)
        #merged_mods_info_df["originInfo-publication0_dateIssued"] = pd.to_numeric(merged_mods_info_df["originInfo-publication0_dateIssued"], errors="coerce")
        #merged_mods_info_df.dropna(subset=["originInfo-publication0_dateIssued"], inplace=True)
        #merged_mods_info_df["originInfo-publication0_dateIssued"] = merged_mods_info_df["originInfo-publication0_dateIssued"].astype(int)
        #merged_mods_info_df = merged_mods_info_df.reset_index(drop=True)

        #print("\nMerged mods_info_df: \n", merged_mods_info_df.head())
        
        #merged_mods_info_df.to_csv("merged_mods_info_df_2025-03-07.csv", index=False)
        
    elif "2025-03-07" in mods_info_csv:
        mods_info_df = pd.DataFrame(load_csv_to_list(mods_info_csv)[1:], columns=["PPN", "genre-aad", "originInfo-publication0_dateIssued"])
        
        genre_evaluation("PPN", mods_info_df, results_df)

    if search_genre is not None:
        results_df = results_df[results_df["ppn"].isin(mods_info_df.loc[mods_info_df["genre-aad"].str.contains(search_genre, na=False), "PPN"])]
    
    if year_start is not None and year_end is not None:
        date_ranges(results_df, mods_info_df, year_start, year_end)
        results_df = results_df[results_df["ppn"].isin(
        mods_info_df.loc[
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) >= year_start) &
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) <= year_end),
            "PPN"])] 
            
    if search_genre is not None and year_start is not None and year_end is not None:
        results_df = results_df[results_df["ppn"].isin(
        mods_info_df.loc[
            (mods_info_df["genre-aad"].str.contains(search_genre, na=False)) &
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) >= year_start) &
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) <= year_end),
            "PPN"])]
            
    if mean_word_start is not None and mean_word_end is not None:
        mean_word_confs(results_df, mods_info_df, mean_word_start=mean_word_start, mean_word_end=mean_word_end)
        results_df = results_df[
            (results_df['mean_word'] >= mean_word_start) & 
            (results_df['mean_word'] <= mean_word_end)]
            
    if mean_textline_start is not None and mean_textline_end is not None:
        mean_textline_confs(results_df, mods_info_df, mean_textline_start=mean_textline_start, mean_textline_end=mean_textline_end)
        results_df = results_df[
            (results_df['mean_textline'] >= mean_textline_start) & 
            (results_df['mean_textline'] <= mean_textline_end)]
            
    if mean_word_start is not None and mean_word_end is not None and search_genre is not None:
        results_df = results_df[results_df["ppn"].isin(mods_info_df.loc[mods_info_df["genre-aad"].str.contains(search_genre, na=False), "PPN"])]
        results_df = results_df[
            (results_df['mean_word'] >= mean_word_start) & 
            (results_df['mean_word'] <= mean_word_end)]
        
    if mean_word_start is not None and mean_word_end is not None and year_start is not None and year_end is not None:
        results_df = results_df[results_df["ppn"].isin(
        mods_info_df.loc[
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) >= year_start) &
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) <= year_end),
            "PPN"])] 
        results_df = results_df[
            (results_df['mean_word'] >= mean_word_start) & 
            (results_df['mean_word'] <= mean_word_end)]
            
    if mean_word_start is not None and mean_word_end is not None and year_start is not None and year_end is not None and search_genre is not None:
        results_df = results_df[results_df["ppn"].isin(mods_info_df.loc[mods_info_df["genre-aad"].str.contains(search_genre, na=False), "PPN"])]
        results_df = results_df[results_df["ppn"].isin(
        mods_info_df.loc[
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) >= year_start) &
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) <= year_end),
            "PPN"])] 
        results_df = results_df[
            (results_df['mean_word'] >= mean_word_start) & 
            (results_df['mean_word'] <= mean_word_end)]
            
    if mean_textline_start is not None and mean_textline_end is not None and search_genre is not None:
        results_df = results_df[results_df["ppn"].isin(mods_info_df.loc[mods_info_df["genre-aad"].str.contains(search_genre, na=False), "PPN"])]
        results_df = results_df[
            (results_df['mean_textline'] >= mean_textline_start) & 
            (results_df['mean_textline'] <= mean_textline_end)]
        
    if mean_textline_start is not None and mean_textline_end is not None and year_start is not None and year_end is not None:
        results_df = results_df[results_df["ppn"].isin(
        mods_info_df.loc[
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) >= year_start) &
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) <= year_end),
            "PPN"])] 
        results_df = results_df[
            (results_df['mean_textline'] >= mean_textline_start) & 
            (results_df['mean_textline'] <= mean_textline_end)]
            
    if mean_textline_start is not None and mean_textline_end is not None and year_start is not None and year_end is not None and search_genre is not None:
        results_df = results_df[results_df["ppn"].isin(mods_info_df.loc[mods_info_df["genre-aad"].str.contains(search_genre, na=False), "PPN"])]
        results_df = results_df[results_df["ppn"].isin(
        mods_info_df.loc[
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) >= year_start) &
            (mods_info_df["originInfo-publication0_dateIssued"].astype(int) <= year_end),
            "PPN"])] 
        results_df = results_df[
            (results_df['mean_textline'] >= mean_textline_start) & 
            (results_df['mean_textline'] <= mean_textline_end)]
            
    if use_best_ppns:
        best_ppns(results_df, mods_info_df, num_best_ppns=num_best_ppns)
        
    if use_worst_ppns:
        worst_ppns(results_df, mods_info_df, num_worst_ppns=num_worst_ppns)
    
    if results_df.empty:
        print("\nThere are no results matching the applied filters.")
    else:
        print("\nStatistics results:\n", results_df)

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
        plt.show()
