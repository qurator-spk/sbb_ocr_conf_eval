# sbb_ocr_conf_eval

> A toolkit for large-scale OCR quality assessment and confidence score analysis.

## Description
`sbb_ocr` (short for `sbb_ocr_conf_eval`) is a command-line toolkit for analyzing OCR confidence scores and evaluating text recognition quality using ground-truth-based metrics. It was developed in the course of the research presented in the paper [*How Scalable is Quality Assessment of Text Recognition? A Combination of Ground Truth and Confidence Scores*](https://anthology.ach.org/volumes/vol0003/how-scalable-is-quality-assessment-of-text-of/), which investigates scalable approaches to assessing text recognition quality across large collections of digitized documents.

The toolkit provides utilities for:

- **OCR confidence score analysis** – extract, aggregate, and summarize word- and textline-level confidence scores from OCR output.
- **Metadata-driven exploration** – filter and compare quality estimates by publication year, genre, language, and other metadata.
- **Visualization** – generate plots and descriptive statistics for confidence scores and collection characteristics.
- **Ground-truth-based benchmarking** – compute error metrics (e.g., Word Error Rate) with `dinglehopper` and combine them with confidence score statistics.
- **Scalable quality assessment** – investigate OCR confidence scores as a proxy for text recognition quality, enabling efficient quality assessment when comprehensive ground truth is unavailable.

## Installation

```sh
# create and/or activate venv if necessary
python3 -m venv venv
source venv/bin/activate
# Install
pip install -e .
```

## Usage
```text
Usage:    
    
    sbb_ocr COMMAND [ARGS] [OPTIONS]...
  
    CSV_FILEs: One or more CSV files containing confidence scores data.
    
    PLOT_FILE: Output file name for the generated plot (e.g., PNG or JPG).


Examples: 

    sbb_ocr plot -m /path/METADATA_FILE /path/CONF_CSV_FILES /path/PLOT_FILE
 
    sbb_ocr plot /path/CONF_CSV_FILES /path/PLOT_FILE -g GENRE -sb SUBGENRE
 
    sbb_ocr plot /path/CONF_CSV_FILES /path/PLOT_FILE -d <YEAR_START> <YEAR_END>
 
    sbb_ocr evaluate -d <PARENT_DIRECTORY> <GT_DIRECTORY> <OCR_DIRECTORY> <REPORT_DIRECTORY>


Commands:

    plot                    Plot confidence metrics from all CONF_CSV_FILES, output to a single PLOT_FILE 
    evaluate                Evaluate OCR word confidence scores with word error rates
    ppn2confs               Get the table of line and word confidences for PPN
    ppn2kitodo              Translate PPN into Kitodo ID
    ppn2pagexml             Get a list of PAGE-XML files for PPN
    ppn2mets                Get METS file for PPN
    create-metadata         Create a lighter version of the METADATA_FILE (e.g., metadata.csv) based on a list of PPNs (e.g., ppns_pipeline_batch_01_2024.txt) and a MODS_INFO_FILE (e.g., mods_info_df_2024-09-06.csv).
    convert-mods-info       Convert mods_info.parquet.sqlite3 to CONF_CSV_FILE and remove all non-zero indexed names
    merge-mods-info         Merge a list of PPNs (e.g., PPN.list.2024-09-06) with a MODS_INFO_FILE (e.g., mods_info_df_2024-09-06.csv) to create a lighter version of the MODS_INFO_FILE (e.g., merged_mods_info_df_2025-03-07.csv).


Options:
    --help                                          Show this message and exit
  
    plot:
       -m,      --metadata                          Add a METADATA_FILE with the PPN metadata (default is metadata.csv)
       -a,      --aggregate-mode                    Choose between aggregation by PPN or PPN_PAGE (default is PPN_PAGE)
       -r,      --show-results                      Show the light version of the results [ppn, ppn_page/num_pages, mean_word, weight_word, weight_textline, originInfo-publication0_dateIssued, genre-aad] (optional)
       -o,      --output                            Save the results and the description of the results to an OUTPUT_CSV_FILE (optional)
       -wm,     --weighting-method                  Choose whether to show only the weighted plots, only the unweighted plots, or both (default is both)
       -g,      --genre                             Choose a GENRE to be evaluated (optional)
       -sg,     --subgenre                          Choose a SUBGENRE to be evaluated (optional)
       -ge,     --show-genre-evaluation             Evaluate the genres in the CSV_FILEs (optional)
       -d,      --search-date                       Filter the data for a specific publication year, specify <YEAR> (optional)
       -dr,     --date-range                        Choose a publication date range for filtering the data, specify <YEAR_START> <YEAR_END> (optional)
       -l,      --search-language                   Filter the data for a specific language, specify <LANGUAGE> (optional)
       -le,     --show-languages-evaluation         Evaluate the languages in the CSV_FILES (optional)
       -ppn,    --search-ppn                        Filter the data for a specific PPN, specify <PPN> (optional)
       -de,     --show-dates-evaluation             Evaluate the publication dates in the CSV_FILEs (optional)
       -wc,     --mean-word-conf                    Filter the data for a specific mean word confidence score, specify <MEAN_WORD> (optional)
       -tc,     --mean-textline-conf                Filter the data for a specific mean textline confidence score, specify <MEAN_TEXTLINE> (optional)
       -wcr,    --mean-word-confs-range             Choose a mean word confidence score range for filtering data, specify <MEAN_WORD_START MEAN_WORD_END> (optional)
       -tcr,    --mean-textline-confs-range         Choose a mean textline confidence score range for filtering data, specify <MEAN_TEXTLINE_START MEAN_TEXTLINE_END> (optional)
       -bmw,    --best-mean-word-confs              Choose a number of PPNs or PPN_PAGEs with the best mean word scores, specify <NUMBER_OF> (optional)
       -wmw,    --worst-mean-word-confs             Choose a number of PPNs or PPN_PAGEs with the worst mean word scores, specify <NUMBER_OF> (optional)
       -bmt,    --best-mean-textline-confs          Choose a number of PPNs or PPN_PAGEs with the best mean textline scores, specify <NUMBER_OF> (optional)
       -wmt,    --worst-mean-textline-confs         Choose a number of PPNs or PPN_PAGEs with the worst mean textline scores, specify <NUMBER_OF> (optional)
       -topw,   --top-ppns-word                     Choose a number of top PPNs or PPN_PAGEs with mean word scores between 0.95 and 1.0, specify <NUMBER_OF> (optional)
       -botw,   --bottom-ppns-word                  Choose a number of bottom PPNs or PPN_PAGEs with mean word nscores between 0.0 and 0.05, specify <NUMBER_OF> (optional)
       -topt,   --top-ppns-textline                 Choose a number of top PPNs or PPN_PAGEs with mean textline scores between 0.95 and 1.0, specify <NUMBER_OF> (optional)
       -bott,   --bottom-ppns-textline              Choose a number of bottom PPNs or PPN_PAGEs with mean textline scores between 0.0 and 0.05, specify <NUMBER_OF> (optional)
       -we,     --weights-evaluation                Evaluate the word and textline counts (weights) in the CSV_FILES (optional)
       -ww,     --search-weight-word                Filter the data for a specific number of words (word weight), specify <NUMBER_OF> (optional)
       -wt,     --search-weight-textline            Filter the data for a specific number of textlines (textline weight), specify <NUMBER_OF> (optional)
       -wwr,    --weight-word-range                 Choose a word count (word weight) range for filtering data, specify <NUMBER_START> <NUMBER_END> (optional)
       -wtr,    --weight-textline-range             Choose a textline count (textline weight) range for filtering data, specify <NUMBER_START> <NUMBER_END> (optional)
       -ne,     --number-of-pages-evaluation        Evaluate the page counts in the CSV_FILES (optional)
       -np,     --search-number-of-pages            Filter the data for a specific number of PPN_PAGEs, specify <NUMBER_OF> (optional)
       -npr,    --number-of-pages-range             Choose a number of PPN_PAGEs range for filtering data, specify <NUMBER_START> <NUMBER_END> (optional)
       -ppndir, --ppn-directory                     Generate a CSV with confidence scores from the names of PPN subdirectories in a <PARENT_DIRECTORY>, specify <PARENT_DIRECTORY> <CONF_CSV> (optional)
       -log,    --use-logging                       Save all log messages to log_plot_{TIMESTAMP}.txt (optional)
       -cve,    --check-value-errors                Check the CSV_FILEs for ValueErrors and save them to value_error_pages.csv (optional)
       -cd,     --check-duplicates                  Check the CSV_FILEs for duplicates and save them to duplicates.csv (optional)
       -crg,    --check-raw-genres                  Check the METADATA_FILE for all raw genres and save them to genres_raw.csv (optional)
       -crl,    --check-raw-languages               Check the METADATA_FILE for all raw languages and save them to languages_raw.csv (optional)
       -hi,     --histogram-info                    Show detailed information about histogram bins (optional)
   
    evaluate:
       -d,      --dinglehopper                      Perform ocrd-dinglehopper on a <PARENT_DIRECTORY>, specify <PARENT_DIRECTORY> <GT_DIRECTORY> <OCR_DIRECTORY> <REPORT_DIRECTORY> (optional)
       -e,      --error-rates                       Generate a CSV with error rates created by ocrd-dinglehopper, specify <PARENT_DIRECTORY> <REPORT_DIRECTORY> <ERROR_RATES_CSV> (optional)
       -m,      --merge-csv                         Generate a CSV with confidence scores and error rates by merging <CONF_CSV> and <ERROR_RATES_CSV>, specify <CONF_CSV> <ERROR_RATES_CSV> <MERGED_CSV> (optional)
       -p,      --plot                              Generate an interactive scatter plot (<HTML_FILE>) showing the relationship between mean word confidence (WC) and word error rate (WER) based on <MERGED_CSV>, and perform a regression analysis., specify <MERGED_CSV> <HTML_PLOT_FILE>
       -log,    --use-logging                       Save all log messages to log_evaluate_{TIMESTAMP}.txt (optional)

    create-metadata:
       -d,      --drop-ppns                         Drop rows from METADATA_FILE with PPNs that are in PPN_LIST, specify <PPN_LIST> <METADATA_CSV_OLD> <METADATA_CSV_NEW> (optional)

    ppn2pagexml:
       --format                                     Whether to output csv or json
       --output                                     Print to this file
 
    ppn2confs:
       --format                                     Whether to output csv or json
       --output                                     Print to this file
```

## How to cite
```bibtex
@incollection{bubula_chr2025,
  author    = {Bubula, Micha{\l} and Baierer, Konstantin and Lehmann, J{\"o}rg and Neudecker, Clemens and Rezanezhad, Vahid and {\v S}kari{\'c}, Doris},
  title     = {How Scalable is Quality Assessment of Text Recognition? A Combination of Ground Truth and Confidence Scores},
  booktitle = {Computational Humanities Research 2025},
  publisher = {Anthology of Computers and the Humanities},
  year      = {2025},
  month     = nov,
  pages     = {1285--1309},
  doi       = {10.63744/gr59c1ixu6wj},
  url       = {https://doi.org/10.63744/gr59c1ixu6wj},
  issn      = {3070-8931}
}
```

## Additional information

### Language codes

Language codes follow the **ISO 639-2 (alpha-3)** standard:

https://www.loc.gov/standards/iso639-2/php/code_list.php

### Ground-truth evaluation

The `evaluate -d` command uses
[dinglehopper](https://github.com/qurator-spk/dinglehopper) for ground-truth-based OCR evaluation. Install it as follows:

Install it as follows:

```sh
git clone https://github.com/qurator-spk/dinglehopper
cd dinglehopper
pip install -e . 'uniseg<0.9'
```

The input directories should be structured as follows:

```text
<PARENT_DIRECTORY>
├── <GT_DIRECTORY>
├── <OCR_DIRECTORY>
└── <REPORT_DIRECTORY>
```
