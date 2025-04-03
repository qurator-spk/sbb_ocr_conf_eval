# sbb_ocr

> Evaluating OCR @StabiBerlin

## METADATA_FILES description

Location: `/nfs/git-annex/michal.bubula/csv/confs_in_archive` 
- Number of PPNS: ***46506***
- Number of PPN_PAGES: ***4632942***
- Number of CSV_FILES: ***10988***

`metadata.csv`:
- Number of rows: ***47716*** (without header row)
- Number of columns: ***3***
- PPNs in `PPN` column
- Genres in `genre-aad` column
- Publication dates in `originInfo-publication0_dateIssued` column ***(with 22 rows, which elements are not 4 digits)***
- Projects: ***missing***
- Titlepages: ***missing***
- Source: Created from `mods_info_df_2024-09-06.csv` and `ppns_pipeline_batch_01_2024.txt` (rows: 47716)


## Installation

```sh
# create and/or activate venv if necessary
python3 -m venv venv
source venv/bin/activate
# Install
pip install -e .
```

## Usage
```
Usage:    
    
    sbb_ocr COMMAND [ARGS] [OPTIONS]...
  
    CSV_FILES: One or more CSV files containing confidence scores data.
    
    PLOT_FILE: Output file name for the generated plot (e.g., PNG or JPG).


Examples: 

    sbb_ocr plot -m /path/METADATA_FILE /path/CONF_CSV_FILES /path/PLOT_FILE
 
    sbb_ocr plot -g GENRE -m /path/METADATA_FILE /path/CONF_CSV_FILES /path/PLOT_FILE
 
    sbb_ocr plot -d <YEAR_START> <YEAR_END> -m /path/METADATA_FILE /path/CONF_CSV_FILES /path/PLOT_FILE
 
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
       -m,      --metadata                          Add a METADATA_FILE with the PPN metadata
       -r,      --show-results                      Show the light version of the results [ppn, ppn_page, mean_word, originInfo-publication0_dateIssued, genre-aad] (optional)
       -o,      --output                            Save the results and the description of the results to an OUTPUT_CSV_FILE (optional)
       -g,      --genre                             Choose a GENRE (optional)
       -ge,     --show-genre-evaluation             Evaluate the number of genres in the CSV_FILES and save the corresponding bar plot (optional)
       -d,      --date-range                        Choose a date range for filtering the data, specify <YEAR_START> <YEAR_END> (optional)
       -de,     --show-dates-evaluation             Evaluate the number of years in the CSV_FILES and save the corresponding bar plot (optional)
       -wc,     --mean-word-confs                   Choose a mean word confidence score range for filtering data, specify <MEAN_WORD_START MEAN_WORD_END> (optional)
       -tc,     --mean-textline-confs               Choose a mean textline confidence score range for filtering data, specify <MEAN_TEXTLINE_START MEAN_TEXTLINE_END> (optional)
       -bmw,    --best-mean-word-confs              Choose a number of PPN_PAGEs with the best mean word scores, specify <NUMBER_OF> (optional)
       -wmw,    --worst-mean-word-confs             Choose a number of PPN_PAGEs with the worst mean word scores, specify <NUMBER_OF> (optional)
       -bmt,    --best-mean-textline-confs          Choose a number of PPN_PAGEs with the best mean textline scores, specify <NUMBER_OF> (optional)
       -wmt,    --worst-mean-textline-confs         Choose a number of PPN_PAGEs with the worst mean textline scores, specify <NUMBER_OF> (optional)
       -topw,   --top-ppns-word                     Choose a number of top PPN_PAGEs with mean word scores between 0.95 and 1.0, specify <NUMBER_OF> (optional)
       -botw,   --bottom-ppns-word                  Choose a number of bottom PPN_PAGEs with mean word nscores between 0.0 and 0.05, specify <NUMBER_OF> (optional)
       -topt,   --top-ppns-textline                 Choose a number of top PPN_PAGEs with mean textline scores between 0.95 and 1.0, specify <NUMBER_OF> (optional)
       -bott,   --bottom-ppns-textline              Choose a number of bottom PPN_PAGEs with mean textline scores between 0.0 and 0.05, specify <NUMBER_OF> (optional)
       -bmwu,   --best-mean-word-confs-unique       Choose a number of unique PPNs whose PPN_PAGEs have the best mean word scores, specify <NUMBER_OF> (optional)
       -wmwu,   --worst-mean-word-confs-unique      Choose a number of unique PPNs whose PPN_PAGEs have the worst mean word scores, specify <NUMBER_OF> (optional)
       -bmtu,   --best-mean-textline-confs-unique   Choose a number of unique PPNs whose PPN_PAGEs have the best mean textline scores, specify <NUMBER_OF> (optional)
       -wmtu,   --worst-mean-textline-confs-unique  Choose a number of unique PPNs whose PPN_PAGEs have the worst mean textline scores, specify <NUMBER_OF> (optional)
       -ppndir, --ppn-directory                     Generate a CSV with confidence scores from the names of PPN subdirectories in a <PARENT_DIRECTORY>, specify <PARENT_DIRECTORY> <CONF_CSV> (optional)
       -log,    --use-logging                       Save all log messages to log_plot_{TIMESTAMP}.txt (optional)
   
    evaluate:
       -d,      --dinglehopper                      Perform ocrd-dinglehopper on a <PARENT_DIRECTORY>, specify <PARENT_DIRECTORY> <GT_DIRECTORY> <OCR_DIRECTORY> <REPORT_DIRECTORY> (optional)
       -e,      --error-rates                       Generate a CSV with error rates created by ocrd-dinglehopper, specify <PARENT_DIRECTORY> <REPORT_DIRECTORY> <ERROR_RATES_CSV> (optional)
       -m,      --merge-csv                         Generate a CSV with confidence scores and error rates by merging <CONF_CSV> and <ERROR_RATES_CSV>, specify <CONF_CSV> <ERROR_RATES_CSV> <MERGED_CSV> (optional)
       -p,      --plot                              Make a scatter plot of the mean word confidence score (WC) and the word error rate (WER) based on a <MERGED_CSV>, specify <MERGED_CSV> <PLOT_FILE>
       -pi      --plot-interactive                  Make an interactive scatter plot (<HTML_FILE>) of the mean word confidence score (WC) and the word error rate (WER) based on a <MERGED_CSV>, specify <MERGED_CSV> <HTML_FILE>
       -log,    --use-logging                       Save all log messages to log_evaluate_{TIMESTAMP}.txt (optional)
 
    ppn2pagexml:
       --format                                     Whether to output csv or json
       --output                                     Print to this file
 
    ppn2confs:
       --format                                     Whether to output csv or json
       --output                                     Print to this file
```

## Extra installation

In order to use `sbb_ocr evaluate -d` you have to properly install `dinglehopper`:
```
git clone https://github.com/qurator-spk/dinglehopper
cd dinglehopper/
pip install -e . 'uniseg<0.9'
```

The directories are structured as follows:
```
<PARENT_DIRECTORY>
    +-- <GT_DIRECTORY>
    +-- <OCR_DIRECTORY>
    +-- <REPORT_DIRECTORY>
```