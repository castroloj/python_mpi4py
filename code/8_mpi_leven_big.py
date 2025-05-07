from mpi4py import MPI
import pandas as pd
import Levenshtein
import numpy as np
import time
import csv
import sys

def main():
    started_at = time.monotonic()
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only rank 0 loads the data
    if rank == 0:
        try:
            N = int(sys.argv[1])
        except (IndexError, ValueError) as e:
            N = 10

        if N < 0:
            N = 10
        print("Loading data...")
        left_df = pd.read_csv("../data/left_data.tsv", sep="\t", quoting=csv.QUOTE_NONE)
        right_df = pd.read_csv("../data/right_data.tsv", sep="\t", quoting=csv.QUOTE_NONE)
        left_df = left_df.loc[left_df.index.repeat(N)].reset_index(drop=True)
        right_df = right_df.loc[right_df.index.repeat(N)].reset_index(drop=True)
        left_strings = left_df['string_left'].values
        right_strings = right_df['string_right'].values
        left_splits = np.array_split(left_strings, size)
        print(f"Data loaded: {len(left_strings):,} vs {len(right_strings):,} strings")
        print(f'Total number of ratios to be computed: {left_df.shape[0]*right_df.shape[0]:,}')
    else:
        left_splits = None
        left_strings, right_strings = None, None
    
    # Scatter data to all nodes
    left_strings = comm.scatter(left_splits, root=0)
    print(f"Rank {rank} got {len(left_strings)} left strings.")
    # Broadcast data to all nodes
    right_strings = comm.bcast(right_strings, root=0)
    
    # Compute partial results
    threshold = 0.8
    local_results = []
    for str_left in left_strings:
        for str_right in right_strings:
            ratio = Levenshtein.ratio(str_left, str_right)
            if ratio > threshold:
                local_results.append([
                    ratio,
                    str_left,
                    str_right
                ])
   
    # Gather all results to root
    all_results = comm.gather(local_results, root=0)
    
    # Process and display results
    if rank == 0:
        final = [item for sublist in all_results for item in sublist]
        
        # Convert to dataframe
        result_df = pd.DataFrame(final, columns=['l_ratio', 'str_left', 'str_right'])
        elapsed_time = time.monotonic() - started_at
        print(f"\nMultiProcessing completed in {elapsed_time:.2f} seconds with {size} mpi processes.")
        print(f"Found {len(result_df):,} similar pairs")
        print("\nTop matches:")
        print(result_df.sort_values('l_ratio', ascending=False).head(10))
    
if __name__ == "__main__":
    main()
    
    MPI.Finalize()