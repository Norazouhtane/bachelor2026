from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

def parse_ess_codebook(html_file):
    """
    Extract non-response codes from ESS codebook HTML.
    
    Parameters:
        html_file (string): Path to ESS codebook HTML file.
    
    Returns:
        nonresponse_dict (dict): Dictionary of lists of variable names and non-response values.
    """
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    nonresponse_dict = {}
    
    for var_header in soup.find_all('h3', id=True):
        var_name = var_header.get('id')
        parent_div = var_header.find_parent('div')
        if not parent_div:
            continue
            
        data_table = parent_div.find('div', class_='data-table')
        if not data_table:
            continue
        
        tbody = data_table.find('tbody', class_='codelist')
        if not tbody:
            continue
        
        nonresponse_codes = []
        for row in tbody.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                value_text = cells[0].get_text(strip=True)
                category_text = cells[1].get_text(strip=True)
                
                if category_text.endswith('*'):
                    value = int(value_text)
                    nonresponse_codes.append(value)
        
        if nonresponse_codes:
            nonresponse_dict[var_name] = nonresponse_codes
    
    return nonresponse_dict


def clean_ess_data(df, codebook, nonresponse_dict=None):
    """
    Replace all non-response codes with NaN.
    
    Parameters:
        df : DataFrame with ESS data
        codebook (string): Path to ESS codebook HTML file
    
    Returns:
        df_clean (DataFrame): Cleaned DataFrame without non-response values
    UPDATE DOCSTRING
    """
    if nonresponse_dict is None:
        nonresponse_dict = parse_ess_codebook(codebook)
        print(f"{len(nonresponse_dict)} variables had non-response values\n")
    
    df_clean = df.copy()
    
    for col, code_val in nonresponse_dict.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(code_val, np.nan)
    
    return df_clean