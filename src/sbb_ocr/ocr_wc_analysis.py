from contextlib import contextmanager
import csv
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt  
from scipy.stats import gaussian_kde  
from tqdm import tqdm  
import json

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

def plot_everything(csv_files : list[str], mods_info_csv, plot_file="statistics_results.jpg"):
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
    
    if "mods_info_df_2024-11-27.csv" in mods_info_csv:
        mods_info_df = pd.DataFrame(load_csv_to_list(mods_info_csv)[1:], columns=["accessCondition-use and reproduction", "accessCondition-restriction on access", "classification-ZVDD", "genre-aad", "identifier-purl", "identifier-vd17", "language_languageTerm", "location_physicalLocation", "location_shelfLocator", "name0_displayForm", "name0_namePart-family", "name0_namePart-given", "name0_role_roleTerm", "originInfo-digitization0_dateCaptured", "originInfo-digitization0_edition", "originInfo-digitization0_place_placeTerm", "originInfo-digitization0_publisher", "originInfo-publication0_dateIssued", "originInfo-publication0_place_placeTerm", "originInfo-publication0_publisher", "relatedItem-original_recordInfo_recordIdentifier", "titleInfo_subTitle", "titleInfo_title", "typeOfResource", "mets_fileSec_fileGrp-FULLTEXT-count", "mets_fileSec_fileGrp-DEFAULT-count", "mets_fileSec_fileGrp-PRESENTATION-count", "mets_fileSec_fileGrp-THUMBS-count", "mets_file", "identifier-RISMA2", "originInfo-production0_dateCreated", "originInfo-production0_edition", "identifier-KOPE", "originInfo-production0_place_placeTerm", "genre-sbb", "mets_fileSec_fileGrp-MAX-count", "mets_fileSec_fileGrp-MIN-count", "mets_fileSec_fileGrp-LOCAL-count", "language_scriptTerm", "name0_namePart", "relatedItem-host_recordInfo_recordIdentifier", "classification-sbb", "identifier-vd18", "identifier-ORIE", "originInfo-publication0_edition", "identifier-PPNanalog", "genre-wikidata", "identifier-vd16", "subject-EC1418_genre", "subject_name0_displayForm", "subject_name0_namePart-family", "subject_name0_namePart-given", "classification-ddc", "identifier-MMED", "relatedItem-original_recordInfo_recordIdentifier-dnb-ppn", "name0_namePart-termsOfAddress", "genre-marcgt", "identifier-zdb", "identifier-RISMA1", "identifier-GW", "identifier-doi", "classification-ark", "abstract", "accessCondition-embargo enddate", "titleInfo_partName", "identifier-KSTO", "identifier-ISSN", "genre", "identifier-EC1418"])
        
        mods_info_df["ppn_mods"] = mods_info_df["mets_file"].apply(lambda x: x.split("/")[-1].split(".")[0])
        
        results_df = results_df[results_df["ppn"].isin(mods_info_df["ppn_mods"])]
    
        all_genres = mods_info_df["genre-aad"].unique().tolist()
        print("Number of all genres: ", len(all_genres))
    
        results_df = results_df[results_df["ppn"].isin(mods_info_df.loc[mods_info_df["genre-aad"] == "{'Roman'}", "ppn_mods"])] # Use "Roman" as an example
    
    elif "mods_info_df_2024-09-06.csv" in mods_info_csv:
        
        mods_info_df = pd.DataFrame(load_csv_to_list(mods_info_csv)[1:], columns=["PPN", "PPN", "accessCondition-embargo enddate", "accessCondition-restriction on access", "accessCondition-use and reproduction", "classification-ZVDD", "classification-ark", "classification-ddc", "classification-sbb", "genre", "genre-aad", "genre-sbb", "genre-wikidata", "identifier-vd16", "identifier-vd17", "identifier-vd18", "language_languageTerm", "language_scriptTerm", "mets_fileSec_fileGrp-FULLTEXT-count", "mets_fileSec_fileGrp-PRESENTATION-count", "originInfo-digitization0_dateCaptured", "originInfo-digitization0_publisher", "originInfo-production0_dateCreated", "originInfo-publication0_dateIssued", "originInfo-publication0_publisher", "recordInfo_recordIdentifier", "subject-EC1418_genre", "titleInfo_title", "typeOfResource", "vd", "vd16", "vd17", "vd18", "columns", "german", "druck"])
        
        results_df = results_df[results_df["ppn"].isin(mods_info_df["recordInfo_recordIdentifier"])]
        
        all_genres = mods_info_df["genre-aad"].unique().tolist()
        print("Number of all genres: ", len(all_genres))
        
        results_df = results_df[results_df["ppn"].isin(mods_info_df.loc[mods_info_df["genre-aad"] == "{'Roman'}", "recordInfo_recordIdentifier"])] # Use "Roman" as an example
        
    
    
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
