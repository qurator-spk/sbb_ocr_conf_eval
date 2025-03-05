# sbb_ocr

> Evaluating OCR @StabiBerlin

## MODS_INFO_FILES description

Location: `/nfs/git-annex/michal.bubula/csv/`

`mods_info_df_2024-09-06.csv`: 
- total number of rows: ***64260***
- PPNs in `recordInfo_recordIdentifier` column
- Publication dates in `originInfo-publication0_dateIssued` column
- Genres in `genre-aad` column ***(too many)***
- Projects: ***missing***
- Titlepages: ***missing***
- Source: `T:\QURATOR\2024-08-select-documents-for-mass-digitization\2024-09-06\documents.csv`
	
`mods_info_df_2024-11-27.csv`: 
- total number of rows: ***224183***
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
Usage: sbb_ocr COMMAND [OPTIONS] [ARGS]...

Example: sbb_ocr plot -M /path/MODS_INFO_FILE.csv /path/CONF_CSV_FILES.csv /path/PLOT_FILE.png

Options:
  --help              Show this message and exit
  -M, --mods-info     Add MODS_INFO_FILE with the PPN metadata

Commands:
  plot                Plot confidence metrics from all CONF_CSV_FILES, output to a single PLOT_FILE 
                      (add -M or --mods-info for the MODS_INFO_FILE with the PPN metadata)
  ppn2confs           Get the table of line and word confidences for PPN
  ppn2kitodo          Translate PPN into Kitodo ID
  ppn2pagexml         Get a list of PAGE-XML files for PPN
  convert-mods-info   Convert mods_info.parquet.sqlite3 to CONF_CSV_FILE and remove all non-zero indexed names
  ppn2mets            Get METS file for PPN
```
