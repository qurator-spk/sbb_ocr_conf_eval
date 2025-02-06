import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt  
from scipy.stats import gaussian_kde  
import requests  
import os  
import xml.etree.ElementTree as ET  

dir_in = "./xmls/"
dir_in_page = "./PPN740124749/TESS/"
all_results = []
df = pd.read_csv("titlepages_fulltext_info.csv")

fulltext = list(df["fileGrp_FULLTEXT_file_FLocat_href"])
fulltext_light = fulltext[:50]

def download_file(url, destination):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("File downloaded successfully!")
    except requests.exceptions.RequestException as e:
        print("Error downloading the file:", e)
        
def statistics(xml_file):
    xml = ET.parse(xml_file)
    root = xml.getroot()
    print("root.namespace ", root.namespace)
    
    try:
        xmlns = str(root.tag).split("}")[0].strip("{")
    except IndexError:
        xmlns = "No namespace found."
    
    if "alto" in root.tag:
        wc_path = f".//{{{xmlns}}}String"
        wc_attr = "WC"
    elif "PAGE" in root.tag:
        wc_path = f".//{{{xmlns}}}TextEquiv"
        wc_attr = "conf"
    else:
        return 0, 0, 0, 0
    
    confidences = []
    for conf in xml.iterfind(wc_path):
            wc = float(conf.attrib.get(wc_attr))
            if wc is not None:
                confidences.append(wc)
    
    if confidences:
        confidences_array = np.array(confidences)
        mean = round(np.mean(confidences_array), 3) 
        median = round(np.median(confidences_array), 3)
        variance = round(np.var(confidences_array), 3)
        standard_deviation = round(np.std(confidences_array), 3)
        
        return mean, median, variance, standard_deviation  
    else:
        return 0, 0, 0, 0

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

XML_DATA_TYPE = "alto" # Choose between "alto" or "page" 

if XML_DATA_TYPE == "alto":
    for url in fulltext_light:
        xml_name = url.split("/")[4]
        save_path = os.path.join(dir_in, xml_name)
        print(save_path)
        ppn_page = xml_name.split(".")[0].split(".")[0]
        download_file(url, save_path)
        mean, median, variance, standard_deviation = statistics(save_path)
        all_results.append([ppn_page, mean, median, variance, standard_deviation]) 
        
elif XML_DATA_TYPE == "page":
    for xml_file in os.listdir(dir_in_page):
        save_path = os.path.join(dir_in_page, xml_file)
        print(save_path)
        ppn_page = xml_file
        mean, median, variance, standard_deviation = statistics(save_path)
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
plt.savefig("statistics_results.jpg")
plt.show()
