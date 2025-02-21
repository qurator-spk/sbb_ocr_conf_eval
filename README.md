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

See `sbb_ocr --help`:

```
Usage: sbb_ocr [OPTIONS] COMMAND [ARGS]...

  Evaluating OCR @StabiBerlin

Options:
  --help  Show this message and exit.

Commands:
  plot         Plot confidence metrics from all CSV_FILES, output to a single PLOT_FILE
  ppn2confs    Get the table of line and word confidences for PPN
  ppn2kitodo   Translate PPN into Kitodo ID
  ppn2pagexml  Get a list of PAGE-XML files for PPN
```
