# sbb_ocr

> Evaluating OCR @StabiBerlin

## MODS_INFO_FILES description

Location: `/nfs/git-annex/michal.bubula/csv/`

`merged_mods_info_df_2025-03-07.csv`:
- Number of rows: ***47694***
- Number of columns: ***3***
- PPNs in `PPN` column
- Genres in `genre-aad` column
- Publication dates in `originInfo-publication0_dateIssued` column ***(dropped NaNs)***
- Projects: ***missing***
- Titlepages: ***missing***
- Source: Merge of `mods_info_df_2024-09-06.csv` and `PPN.list.2024-09-06`

`mods_info_df_2024-09-06.csv`: 
- Number of rows: ***64260***
- Number of columns: ***36***
- PPNs in `recordInfo_recordIdentifier` column ***(renamed to: ***`PPN`***)***
- Publication dates in `originInfo-publication0_dateIssued` column
- Genres in `genre-aad` column
- Projects: ***missing***
- Titlepages: ***missing***
- Source: `T:\QURATOR\2024-08-select-documents-for-mass-digitization\2024-09-06\documents.csv`
	
`mods_info_df_2024-11-27.csv`: 
- Number of rows: ***224183***
- Number of columns: ***69***
- PPNs in `PPN` column ***(via workaround)***
- Publication dates in `originInfo-publication0_dateIssued` column ***(kept strings that are 4 digits long)***
- Genres in `genre-aad` column
- Projects: ***missing***
- Titlepages: ***missing***
- Source: `@lx0246:/data/mike.gerber/2024-11-mods_info-etc/2024-11-27/mods_info_df.parquet.sqlite3`

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

    sbb_ocr plot -m /path/MODS_INFO_FILE /path/CONF_CSV_FILES /path/PLOT_FILE
 
    sbb_ocr plot -g GENRE -m /path/MODS_INFO_FILE /path/CONF_CSV_FILES /path/PLOT_FILE
 
    sbb_ocr plot -d <YEAR_START YEAR_END> -m /path/MODS_INFO_FILE /path/CONF_CSV_FILES /path/PLOT_FILE
 
    sbb_ocr plot -g GENRE -d <YEAR_START YEAR_END> -m /path/MODS_INFO_FILE /path/CONF_CSV_FILES /path/PLOT_FILE


Commands:
    plot                    Plot confidence metrics from all CONF_CSV_FILES, output to a single PLOT_FILE 
    ppn2confs               Get the table of line and word confidences for PPN
    ppn2kitodo              Translate PPN into Kitodo ID
    ppn2pagexml             Get a list of PAGE-XML files for PPN
    ppn2mets                Get METS file for PPN
    convert-mods-info       Convert mods_info.parquet.sqlite3 to CONF_CSV_FILE and remove all non-zero indexed names
    merge-mods-info         Merge a list of PPNs (e.g., PPN.list.2024-09-06) with a MODS_INFO_FILE (e.g., mods_info_df_2024-09-06.csv) to create a lighter version of the MODS_INFO_FILE (e.g., merged_mods_info_df_2025-03-07.csv).


Options:
    --help                             Show this message and exit
  
    plot:
       -m,   --mods-info               Add a MODS_INFO_FILE with the PPN metadata
       -r,   --show-results            Show the light version of the results [ppn, ppn_page, mean_word, originInfo-publication0_dateIssued, genre-aad] (optional)
       -o,   --output                  Save the results and the description of the results to an OUTPUT_CSV_FILE (optional)
       -g,   --genre                   Choose a GENRE (optional)
       -ge,  --show-genre-evaluation   Evaluate the number of genres in the CSV_FILES and save the corresponding bar plot (optional)
       -d,   --date-range              Choose a date range for filtering the data, specify <YEAR_START YEAR_END> (optional)
       -de,  --show-dates-evaluation   Evaluate the number of years in the CSV_FILES and save the corresponding bar plot(optional)
       -top, --top-ppns                Choose a number of top PPN_PAGEs with mean word score & mean textline scores between 0.95 and 1.0, specify <NUMBER_OF> (optional)
       -bot, --bottom-ppns             Choose a number of bottom PPN_PAGEs with mean word score & mean textline scores between 0.0 and 0.05, specify <NUMBER_OF> (optional)
       -wc,  --mean-word-confs         Choose a mean word confidence score range for filtering data, specify <MEAN_WORD_START MEAN_WORD_END> (optional)
       -tc,  --mean-textline-confs     Choose a mean textline confidence score range for filtering data, specify <MEAN_TEXTLINE_START MEAN_TEXTLINE_END> (optional)
 
    ppn2pagexml:
       --format                        Whether to output csv or json
       --output                        Print to this file
 
    ppn2confs:
       --format                        Whether to output csv or json
       --output                        Print to this file
```
