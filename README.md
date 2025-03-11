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
- PPNs in `recordInfo_recordIdentifier` column
- Publication dates in `originInfo-publication0_dateIssued` column
- Genres in `genre-aad` column ***(too many)***
- Projects: ***missing***
- Titlepages: ***missing***
- Source: `T:\QURATOR\2024-08-select-documents-for-mass-digitization\2024-09-06\documents.csv`
	
`mods_info_df_2024-11-27.csv`: 
- Number of rows: ***224183***
- Number of columns: ***69***
- PPNs in `ppn_mods` column ***(via workaround)***
- Publication dates in `originInfo-production0_dateCreated` column ***(needs verification and filtering)***
- Genres in `genre-aad` column ***(too many)***
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

    sbb_ocr plot -t <TOP_NUMBER> /path/MODS_INFO_FILE /path/CONF_CSV_FILES /path/PLOT_FILE


Commands:
    plot                    Plot confidence metrics from all CONF_CSV_FILES, output to a single PLOT_FILE 
    ppn2confs               Get the table of line and word confidences for PPN
    ppn2kitodo              Translate PPN into Kitodo ID
    ppn2pagexml             Get a list of PAGE-XML files for PPN
    convert-mods-info       Convert mods_info.parquet.sqlite3 to CONF_CSV_FILE and remove all non-zero indexed names
    ppn2mets                Get METS file for PPN


Options:
    --help                  Show this message and exit
  
    plot:
        -m, --mods-info     Add MODS_INFO_FILE with the PPN metadata
        -g, --genre         Add GENRE to be evaluated (optional)
        -d, --date-range    Add date range for filtering data, specify <YEAR_START YEAR_END> (optional)
        -t, --top-ppns      Add number of top PPNs with mean word score & mean textline scores between 0.95 and 1.0, specify <TOP_NUMBER> (optional)
 
    ppn2pagexml:
        --format            Whether to output csv or json
        --output            Print to this file
 
    ppn2confs:
        --format            Whether to output csv or json
        --output            Print to this file
```
