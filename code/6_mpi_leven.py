from mpi4py import MPI
import pandas as pd
import Levenshtein
import numpy as np
import time
import csv

def main():
    started_at = time.monotonic()
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only rank 0 loads the data
    if rank == 0:
        print("Loading data...")
        left_df = pd.read_csv("../data/left_data.tsv", sep="\t", quoting=csv.QUOTE_NONE)
        right_df = pd.read_csv("../data/right_data.tsv", sep="\t", quoting=csv.QUOTE_NONE)
        left_strings = left_df['string_left'].values
        right_strings = right_df['string_right'].values
        print(f"Data loaded: {len(left_strings):,} vs {len(right_strings):,} strings")
        print(f'Total number of ratios to be computed: {left_df.shape[0]*right_df.shape[0]:,}')
    else:
        left_strings, right_strings = None, None
    
    # Broadcast data to all nodes
    left_strings = comm.bcast(left_strings, root=0)
    right_strings = comm.bcast(right_strings, root=0)
    
    # Split work across processes
    chunk_size = len(left_strings) // size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank != size - 1 else len(left_strings)
    
    # Compute partial results
    threshold = 0.8
    local_results = []
    
    for i in range(start, end):
        for j in range(len(right_strings)):
            ratio = Levenshtein.ratio(left_strings[i], right_strings[j])
            if ratio > threshold:
                local_results.append((ratio, left_strings[i], right_strings[j]))
    
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