# sbb_ocr

> Evaluating OCR @StabiBerlin

## Installation

```sh
# create and/or activate venv if necessary
python3 -m venv venv
source venv/bin/activate
# Install
pip install -e .
```

## Usage

Usage: `sbb_ocr COMMAND [OPTIONS] [ARGS]...`

Example: `sbb_ocr plot -M /path/MODS_INFO_FILE.csv /path/CONF_CSV_FILES.csv /path/PLOT_FILE .png`

Options:
  --help              Show this message and exit
  -M, --mods-info     Add MODS_INFO_FILE with the PPN metadata

Commands:
  plot                Plot confidence metrics from all CONF_CSV_FILES, output to a single PLOT_FILE 
                      (add -M or --mods-info for the MODS_INFO_FILE with the PPN metadata)
  ppn2confs           Get the table of line and word confidences for PPN
  ppn2kitodo          Translate PPN into Kitodo ID
  ppn2pagexml         Get a list of PAGE-XML files for PPN
  convert-mods-info   Convert mods_info.parquet.sqlite3 to CSV and remove all non-zero indexed names
```
