import pandas as pd
import csv
import Levenshtein
import time

started_at = time.monotonic()
# Read files
left_df = pd.read_csv("../data/left_data.tsv", sep="\t", quoting=csv.QUOTE_NONE)
right_df = pd.read_csv("../data/right_data.tsv", sep="\t", quoting=csv.QUOTE_NONE)

print("="*80)
print(f'The left dataframe has {left_df.shape[0]:,} rows, the right dataframe has {right_df.shape[0]:,} rows.')
print(f'Total number of ratios to be computed: {left_df.shape[0]*right_df.shape[0]:,}')
threshold = 0.80
# Compare all left strings with right strings
# Get levenshtein ratio (similarity)
# Obtain a final dataframe with similar strings
similar_pairs = []
for str_left in left_df['string_left']:
    for str_right in right_df['string_right']:
        ratio = Levenshtein.ratio(str_left, str_right)
        if ratio > threshold:
            similar_pairs.append([
                ratio,
                str_left,
                str_right
            ])

# Convert to dataframe
result_df = pd.DataFrame(similar_pairs, columns=['l_ratio', 'str_left', 'str_right'])

elapsed_time = time.monotonic() - started_at
print(f"\nSequential Processing completed in {elapsed_time:.2f} seconds")
print(f"Found {len(result_df):,} similar pairs")
print("\nTop matches:")
print(result_df.sort_values('l_ratio', ascending=False).head(10))
