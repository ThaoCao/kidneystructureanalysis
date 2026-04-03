#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:29:36 2024

@author: thaocao
Goal: perform statistical anlysis --- need to double check the right tasks to perform and the statistical correction
"""
from scipy import stats


# Calculate mean and standard deviation for each dataset
summary_stats = df.groupby('Dataset').agg({
    'Percentage Tubule': ['mean', 'std'],
    'Percentage Vessel': ['mean', 'std'],
    'Percentage Interstitium': ['mean', 'std']
})

# Flatten column names
summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]

# Perform Mann-Whitney U test for each pair of datasets
datasets = df['Dataset'].unique()
test_results = []

for i in range(len(datasets)):
    for j in range(i+1, len(datasets)):
        dataset1 = datasets[i]
        dataset2 = datasets[j]
        
        # Tubule percentage
        tubule_stat, tubule_p = stats.mannwhitneyu(
            df[df['Dataset'] == dataset1]['Percentage Tubule'],
            df[df['Dataset'] == dataset2]['Percentage Tubule']
        )
        
        # Vessel percentage
        vessel_stat, vessel_p = stats.mannwhitneyu(
            df[df['Dataset'] == dataset1]['Percentage Vessel'],
            df[df['Dataset'] == dataset2]['Percentage Vessel']
        )
        
        test_results.append({
            'Dataset1': dataset1,
            'Dataset2': dataset2,
            'Tubule_p_value': tubule_p,
            'Vessel_p_value': vessel_p
        })

# Convert test results to DataFrame
test_results_df = pd.DataFrame(test_results)

# Save summary statistics and test results to CSV
with pd.ExcelWriter('tissue_composition_analysis.xlsx') as writer:
    summary_stats.to_excel(writer, sheet_name='Summary Statistics')
    test_results_df.to_excel(writer, sheet_name='Mann-Whitney U Test', index=False)

print("Analysis results have been saved in 'tissue_composition_analysis.xlsx'")

# Print summary statistics
print("\nSummary Statistics:")
print(summary_stats)

# Print Mann-Whitney U test results
print("\nMann-Whitney U Test Results:")
print(test_results_df)