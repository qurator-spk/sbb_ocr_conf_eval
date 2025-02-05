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

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Histogram of the means  
bins_mean = np.arange(0, 1.1, 0.05)
axs[0].hist(results_df["mean"], bins=bins_mean, color="lightblue", edgecolor="black", alpha=0.6, density=True)
axs[0].set_title("Histogram of Mean Word Confidence Scores")
axs[0].set_xlabel("Mean Word Confidence")
axs[0].set_ylabel("Frequency")
axs[0].set_xlim(0, 1.0)
axs[0].set_xticks(np.arange(0, 1.1, 0.1)) 
axs[0].grid(axis="y", alpha=0.75)

# Density distribution of the means  
kde_mean = gaussian_kde(results_df["mean"])
x_range_mean = np.linspace(0, 1, 100)
axs[0].plot(x_range_mean, kde_mean(x_range_mean), color="blue", lw=2, label="Density Distribution")
axs[0].legend()

# Box plot of the mean values  
axs[1].boxplot(results_df["mean"], patch_artist=True, boxprops=dict(facecolor="lightgreen"))
axs[1].set_title("Box Plot of Mean Word Confidence Scores")
axs[1].set_ylabel("Mean Word Confidence Scores")
axs[1].set_xticks([1], ["Mean Confidence"])  
axs[1].grid(axis="y", alpha=0.75)

# Histogram of the standard deviation values
bins_sd = np.arange(0, 1.1, 0.05)
axs[2].hist(results_df["standard_deviation"], bins=bins_sd, color="salmon", edgecolor="black", alpha=0.6, density=True)
axs[2].set_title("Histogram of Standard Deviation of Word Confidence Scores")
axs[2].set_xlabel("Standard Deviation")
axs[2].set_ylabel("Frequency")
axs[2].set_xlim(0, 1.0)
axs[2].set_xticks(np.arange(0, 1.1, 0.1)) 
axs[2].grid(axis="y", alpha=0.75)

# Density distribution of the standard deviation values  
kde_sd = gaussian_kde(results_df["standard_deviation"])
x_range_sd = np.linspace(0, 1, 100)
axs[2].plot(x_range_sd, kde_sd(x_range_sd), color="red", lw=2, label="Density Distribution")
axs[2].legend()

plt.tight_layout()
plt.savefig("statistics_results.jpg")
plt.show()
