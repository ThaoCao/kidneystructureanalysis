#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:55:16 2025

@author: thaocao
Parse csv 
"""
import csv
from tabulate import tabulate
from datetime import datetime

def safe_float_convert(value):
    """
    Safely convert a string to float, handling common issues.
    """
    if value is None or value.strip() == '':
        return 0.0
    
    # Remove whitespace and replace comma with period
    cleaned_value = value.strip().replace(',', '.')
    
    try:
        return float(cleaned_value)
    except ValueError:
        print(f"Warning: Could not convert '{value}' to float. Using 0.0 instead.")
        return 0.0

def calculate_date_difference(date1, date2):
    """
    Calculate the difference in days between two dates in YYYY/MM/DD format.
    """
    try:
        d1 = datetime.strptime(date1, "%Y/%m/%d")
        d2 = datetime.strptime(date2, "%Y/%m/%d")
        return (d1 - d2).days
    except ValueError:
        print(f"Warning: Invalid date format. Expected YYYY/MM/DD. Got '{date1}' and '{date2}'. Using 0 days.")
        return 0

def process_csv(file_path, input1_col, input2_col, input3_col, time_input1_col, time_input2_col):
    results = []
    headers = ["Input3", "Result (Input1 - Input2)", "Time Difference (days)"]

    try:
        with open(file_path, 'r') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            
            for row in csv_reader:
                input1 = safe_float_convert(row[input1_col])
                input2 = safe_float_convert(row[input2_col])
                input3 = row[input3_col]
                
                result = input1 - input2
                
                # Calculate time difference
                time_diff = calculate_date_difference(row[time_input1_col], row[time_input2_col])
                
                results.append([input3, result, time_diff])
        
        print(tabulate(results, headers=headers, tablefmt="grid"))
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except KeyError as e:
        print(f"Error: Column {e} not found in the CSV file.")

file_path = '/project/mclark/Cell_patch_classification/Data/all_cohorts/LN_clinical_data.csv'
input1_col = 'Serum_Cr_most_recent'
input2_col = 'Serum_Cr_bx'
input3_col = 'Accession'
time_input1_col = 'Serum_Cr_most_recent_date'
time_input2_col = 'Bx_date'

process_csv(file_path, input1_col, input2_col, input3_col, time_input1_col, time_input2_col)




