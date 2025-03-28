import json
import sys
import csv
import click
from .ppn_handler import PpnHandler, PpnHandlerConfig
from .ocr_wc_analysis import plot_everything
import pandas as pd
import sqlite3

@click.group('sbb_ocr')
def cli():
    """
    Evaluating OCR @StabiBerlin
    """
    pass


@cli.command('plot')
@click.option('-g', '--genre', 'search_genre', help='Genre to be evaluated (optional)')
@click.option('-m', '--mods-info', 'mods_info_csv', default="mods_info_df_2024-11-27.csv", help='mods_info CSV for PPN metadata')
@click.option('-d', '--date-range', 'date_range', nargs=2, type=(int, int), help='Year range for filtering data, specify <YEAR_START YEAR_END> (optional)')
@click.option('-topw', '--top-ppns-word', type=int, help='Number of top PPN_PAGESs with mean word scores between 0.95 and 1.0, specify <NUMBER_OF> (optional)')
@click.option('-botw', '--bottom-ppns-word', type=int, help='Number of bottom PPN_PAGEs with mean word scores between 0.0 and 0.05, specify <NUMBER_OF> (optional)')
@click.option('-topt', '--top-ppns-textline', type=int, help='Number of top PPN_PAGESs with mean textline scores between 0.95 and 1.0, specify <NUMBER_OF> (optional)')
@click.option('-bott', '--bottom-ppns-textline', type=int, help='Number of bottom PPN_PAGEs with mean textline scores between 0.0 and 0.05, specify <NUMBER_OF> (optional)')
@click.option('-wc', '--mean-word-confs', 'mean_word_confs', nargs=2, type=(float, float), help='Mean word confidence score range for filtering data, specify <MEAN_WORD_START MEAN_WORD_END> (optional)')
@click.option('-tc', '--mean-textline-confs', 'mean_textline_confs', nargs=2, type=(float, float), help='Mean textline confidence score range for filtering data, specify <MEAN_TEXTLINE_START MEAN_TEXTLINE_END> (optional)')
@click.option('-ge', '--show-genre-evaluation', 'show_genre_evaluation', is_flag=True, default=False, help="Evaluate the number of genres in the CSV_FILES (optional)")
@click.option('-o', '--output', 'output', type=click.File('w'), help='Save the results to an OUTPUT_CSV_FILE (optional)')
@click.option('-de', '--show-dates-evaluation', 'show_dates_evaluation', is_flag=True, default=False, help="Evaluate the number of years in the CSV_FILES (optional)")
@click.option('-r', '--show-results', 'show_results', is_flag=True, default=False, help="Show the light version of the results [ppn, ppn_page, mean_word, originInfo-publication0_dateIssued, genre-aad] (optional)")
@click.option('-bmwu', '--best-mean-word-confs-unique', type=int, help='Number of unique PPNs whose PPN_PAGEs have the best mean word scores, specify <NUMBER_OF> (optional)')
@click.option('-wmwu', '--worst-mean-word-confs-unique', type=int, help='Number of unique PPNs whose PPN_PAGEs have the worst mean word scores, specify <NUMBER_OF> (optional)')
@click.option('-bmtu', '--best-mean-textline-confs-unique', type=int, help='Number of unique PPNs whose PPN_PAGEs have the best mean textline scores, specify <NUMBER_OF> (optional)')
@click.option('-wmtu', '--worst-mean-textline-confs-unique', type=int, help='Number of unique PPNs whose PPN_PAGEs have the worst mean textline scores, specify <NUMBER_OF> (optional)')
@click.option('-bmw', '--best-mean-word-confs', type=int, help='Number of PPN_PAGEs with the best mean word scores, specify <NUMBER_OF> (optional)')
@click.option('-wmw', '--worst-mean-word-confs', type=int, help='Number of PPN_PAGEs with the worst mean word scores, specify <NUMBER_OF> (optional)')
@click.option('-bmt', '--best-mean-textline-confs', type=int, help='Number of PPN_PAGEs with the best mean textline scores, specify <NUMBER_OF> (optional)')
@click.option('-wmt', '--worst-mean-textline-confs', type=int, help='Number of PPN_PAGEs with the worst mean textline scores, specify <NUMBER_OF> (optional)')
@click.argument('CSV_FILES', nargs=-1)
@click.argument('PLOT_FILE')
def plot_cli(search_genre, mods_info_csv, csv_files, plot_file, date_range, 
             top_ppns_word, bottom_ppns_word, top_ppns_textline, bottom_ppns_textline, 
             mean_word_confs, mean_textline_confs, show_genre_evaluation, output, show_dates_evaluation, show_results,
             best_mean_word_confs_unique, worst_mean_word_confs_unique, best_mean_textline_confs_unique, worst_mean_textline_confs_unique,
             best_mean_word_confs, worst_mean_word_confs, best_mean_textline_confs, worst_mean_textline_confs):
    """
    Plot confidence metrics from all CSV_FILES, output to PLOT_FILE.
    """
    
    if date_range is None:
        year_start, year_end = (None, None)
    else:
        year_start, year_end = date_range
        
    if mean_word_confs is None:
        mean_word_start, mean_word_end = (None, None)
    else:
        mean_word_start, mean_word_end = mean_word_confs
        
    if mean_textline_confs is None:
        mean_textline_start, mean_textline_end = (None, None)
    else:
        mean_textline_start, mean_textline_end = mean_textline_confs
        
    num_top_ppns_word = top_ppns_word if top_ppns_word is not None else 50
    num_bottom_ppns_word = bottom_ppns_word if bottom_ppns_word is not None else 50
    
    num_top_ppns_textline = top_ppns_textline if top_ppns_textline is not None else 50
    num_bottom_ppns_textline = bottom_ppns_textline if bottom_ppns_textline is not None else 50
    
    num_best_mean_word_confs = best_mean_word_confs if best_mean_word_confs is not None else 50
    num_worst_mean_word_confs = worst_mean_word_confs if worst_mean_word_confs is not None else 50
    
    num_best_mean_textline_confs = best_mean_textline_confs if best_mean_textline_confs is not None else 50
    num_worst_mean_textline_confs = worst_mean_textline_confs if worst_mean_textline_confs is not None else 50
    
    num_best_mean_word_confs_unique = best_mean_word_confs_unique if best_mean_word_confs_unique is not None else 50
    num_worst_mean_word_confs_unique = worst_mean_word_confs_unique if worst_mean_word_confs_unique is not None else 50
    
    num_best_mean_textline_confs_unique = best_mean_textline_confs_unique if best_mean_textline_confs_unique is not None else 50
    num_worst_mean_textline_confs_unique = worst_mean_textline_confs_unique if worst_mean_textline_confs_unique is not None else 50
        
    plot_everything(csv_files=csv_files, mods_info_csv=mods_info_csv, search_genre=search_genre,
                    plot_file=plot_file, year_start=year_start, year_end=year_end,
                    use_top_ppns_word=(top_ppns_word is not None), use_bottom_ppns_word=(bottom_ppns_word is not None), num_top_ppns_word=num_top_ppns_word, num_bottom_ppns_word=num_bottom_ppns_word,
                    use_top_ppns_textline=(top_ppns_textline is not None), use_bottom_ppns_textline=(bottom_ppns_textline is not None), num_top_ppns_textline=num_top_ppns_textline, num_bottom_ppns_textline=num_bottom_ppns_textline,
                    mean_word_start=mean_word_start, mean_word_end=mean_word_end, mean_textline_start=mean_textline_start, mean_textline_end=mean_textline_end, 
                    show_genre_evaluation=show_genre_evaluation, output=output, show_dates_evaluation=show_dates_evaluation, show_results=show_results,
                    use_best_mean_word_confs_unique=(best_mean_word_confs_unique is not None), use_worst_mean_word_confs_unique=(worst_mean_word_confs_unique is not None), 
                    num_best_mean_word_confs_unique=num_best_mean_word_confs_unique, num_worst_mean_word_confs_unique=num_worst_mean_word_confs_unique,
                    use_best_mean_textline_confs_unique=(best_mean_textline_confs_unique is not None), use_worst_mean_textline_confs_unique=(worst_mean_textline_confs_unique is not None), 
                    num_best_mean_textline_confs_unique=num_best_mean_textline_confs_unique, num_worst_mean_textline_confs_unique=num_worst_mean_textline_confs_unique,
                    use_best_mean_word_confs=(best_mean_word_confs is not None), use_worst_mean_word_confs=(worst_mean_word_confs is not None), 
                    num_best_mean_word_confs=num_best_mean_word_confs, num_worst_mean_word_confs=num_worst_mean_word_confs,
                    use_best_mean_textline_confs=(best_mean_textline_confs is not None), use_worst_mean_textline_confs=(worst_mean_textline_confs is not None), 
                    num_best_mean_textline_confs=num_best_mean_textline_confs, num_worst_mean_textline_confs=num_worst_mean_textline_confs)

@cli.command('convert-mods-info')
@click.argument('MODS_INFO_SQLITE')
@click.argument('MODS_INFO_CSV')
def convert_mods_info(mods_info_sqlite, mods_info_csv):
    """
    Convert mods_info.parquet.sqlite3 to CSV and remove all non-zero indexed names.
    """
    con = sqlite3.connect(mods_info_sqlite)
    mods_info_df = pd.read_sql("SELECT * FROM mods_info", con, index_col="recordInfo_recordIdentifier")
    columns_to_drop = []
    for c in mods_info_df.columns:
        if c.startswith("name") and not c.startswith("name0"):
            columns_to_drop.append(c)
    mods_info_df.drop(columns=columns_to_drop, inplace=True)
    mods_info_df.to_csv(mods_info_csv, index=False)
    
@cli.command('create-metadata')
@click.argument('PPN_LIST')
@click.argument('METADATA_CSV')
def merge_mods_info(ppn_list, metadata_csv):
    """
    Create a lighter version of the METADATA_FILE (e.g., metadata.csv) based on a list of PPNs (e.g., ppns_pipeline_batch_01_2024.txt) and a MODS_INFO_FILE (e.g., mods_info_df_2024-09-06.csv).
    """
    ppn_list_df = pd.read_csv(ppn_list, header=None, names=['PPN'], dtype=str)

    # Initialize new columns in ppn_list_df
    ppn_list_df['genre-aad'] = pd.NA
    ppn_list_df['originInfo-publication0_dateIssued'] = pd.NA

    # Iterate through each row of ppn_list_df
    for index, row in ppn_list_df.iterrows():
        ppn_value = row['PPN']
        # Check if the ppn_value exists in mods_info_df 
        if ppn_value in mods_info_df['PPN'].values:
            # Get the corresponding row in mods_info_df
            matched_row = mods_info_df[mods_info_df['PPN'] == ppn_value]
            # Update the new columns in ppn_list_df
            ppn_list_df.at[index, 'genre-aad'] = matched_row['genre-aad'].values[0]
            ppn_list_df.at[index, 'originInfo-publication0_dateIssued'] = matched_row['originInfo-publication0_dateIssued'].values[0]
            
    ppn_list_df.to_csv(metadata_csv, index=False)
    
@cli.command('merge-mods-info')
@click.argument('PPN_LIST')
@click.argument('MODS_INFO_CSV')
def merge_mods_info(ppn_list, mods_info_csv):
    """
    Merge a list of PPNs (e.g., PPN.list.2024-09-06) with a MODS_INFO_FILE (e.g., mods_info_df_2024-09-06.csv) to create a lighter version of the MODS_INFO_FILE (e.g., merged_mods_info_df_2025-03-07.csv).
    """
    with open(ppn_list, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    ppn_list_df = pd.DataFrame(lines, columns=['PPN'])
    filtered_mods_info_df = mods_info_df[mods_info_df['recordInfo_recordIdentifier'].isin(ppn_list_df['PPN'])]
    
    merged_mods_info_df = pd.DataFrame()
    merged_mods_info_df['PPN'] = ppn_list_df['PPN']
    merged_mods_info_df = merged_mods_info_df.merge(filtered_mods_info_df[['recordInfo_recordIdentifier', 'genre-aad', 'originInfo-publication0_dateIssued']],
                            left_on='PPN', right_on='recordInfo_recordIdentifier', how='left')

    merged_mods_info_df.drop(columns='recordInfo_recordIdentifier', inplace=True)
    merged_mods_info_df["originInfo-publication0_dateIssued"] = pd.to_numeric(merged_mods_info_df["originInfo-publication0_dateIssued"], errors="coerce")
    merged_mods_info_df.dropna(subset=["originInfo-publication0_dateIssued"], inplace=True)
    merged_mods_info_df["originInfo-publication0_dateIssued"] = merged_mods_info_df["originInfo-publication0_dateIssued"].astype(int)
    merged_mods_info_df = merged_mods_info_df.reset_index(drop=True)
    print("\nMerged mods_info_df: \n", merged_mods_info_df.head())
    merged_mods_info_df.to_csv(mods_info_csv, index=False)

@cli.command('ppn2kitodo')
@click.argument('PPN', nargs=-1)
def ppn2kitodo_cli(ppn):
    """
    Translate PPN into Kitodo ID
    """
    ppn_handler = PpnHandler(PpnHandlerConfig())
    print(json.dumps(ppn_handler.ppn2kitodo(ppn)))

@cli.command('ppn2pagexml')
@click.option('--format', default='csv', type=click.Choice(['csv', 'json']), help="Whether to output csv or json")
@click.option('--output', type=click.File('w'), default=sys.stdout, help='Print to this file')
@click.argument('PPN', nargs=-1)
def ppn2pagexml_cli(format, output, ppn):
    """
    Get a list of PAGE-XML files for PPN
    """
    ppn_handler = PpnHandler(PpnHandlerConfig())
    table = ppn_handler.ppn2pagexml(ppn)
    if format == 'csv':
        writer = csv.writer(output)
        writer.writerow(['ppn', 'pagexml'])
        for ppn, pagexml_list in table.items():
            for pagexml in pagexml_list:
                writer.writerow([ppn, pagexml])
    else:
        json.dump({k: [str(x) for x in v] for k, v in table.items()}, output)

@cli.command('ppn2mets')
@click.argument('PPN', nargs=-1)
def ppn2mets_cli(ppn):
    """
    Get METS file for PPN
    """
    ppn_handler = PpnHandler(PpnHandlerConfig())
    print(ppn_handler.ppn2mets(ppn))

@cli.command('ppn2confs')
@click.option('--format', default='csv', type=click.Choice(['csv', 'json']), help="Whether to output csv or json")
@click.option('--output', type=click.File('w'), default=sys.stdout, help='Print to this file')
@click.argument('PPN', nargs=-1)
def ppn2confs_cli(format, output, ppn):
    """
    Get the table of line and word confidences for PPN
    """
    ppn_handler = PpnHandler(PpnHandlerConfig())
    table = ppn_handler.ppn2confs(ppn)
    if format == 'json':
        json.dump(table, output)
    else:
        writer = csv.writer(output)
        writer.writerow(['ppn', 'file_id', 'version', 'textline_confs', 'word_confs'])
        for row in table:
            for i in [-1, -2]:
                row[i] = ' '.join(map(str, row[i]))
            writer.writerow(row)
