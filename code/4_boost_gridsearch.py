import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from mpi4py import MPI
import itertools
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.pipeline import Pipeline

# Record the start time of the simulation
started_at = time.monotonic()
# Initialize MPI
# Get the MPI communicator for parallel processing
comm = MPI.COMM_WORLD
# Get the total number of processes (how many tasks will run in parallel)
total_procs = comm.Get_size()
# Get the rank (ID) of the current process (starts from 0)
rank = comm.Get_rank()
# Get the hostname of the machine running the current process
hostname = MPI.Get_processor_name()

# ====== Read and get data ready for model training ====== #
df = pd.read_csv("../data/cc_data.csv") 
df = df.dropna()
train_data = df[df['is_train'] == True].drop(columns=['is_train'])

# Get features and target (X, y)
X = train_data.drop(columns=['Credit_Score'])
y = train_data['Credit_Score']
# Split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Encode labels from strings
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
# Pre-processing of data
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
one_hot_cols = ["Occupation", "Payment_of_Min_Amount"]
ordinal_cols = ["Credit_Mix", "Spending_Level", "Payment_Value"]

ordinal_categories = [
    ['Bad', 'Standard', 'Good'],  
    ['Low', 'High'],    
    ['Small', 'Medium', 'Large']
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_cols),  
        ('one_hot_enc', OneHotEncoder(handle_unknown='ignore'), one_hot_cols),  
        ('ordinal_enc', OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1), ordinal_cols)
    ]
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

# === Grid Search Setup ===
if rank == 0:
    print(f"Rank {rank} running on {hostname}")

    param_grid = {
        'n_estimators': [200, 400, 600],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 8],
        'subsample': [0.7, 1],
        'colsample_bytree': [0.8, 1],
        'reg_lambda': [1, 10, 50],
    }

    param_combinations = list(itertools.product(
        param_grid['n_estimators'],
        param_grid['learning_rate'],
        param_grid['max_depth'],
        param_grid['subsample'],
        param_grid['colsample_bytree'],
        param_grid['reg_lambda'],
    ))

    random.shuffle(param_combinations)
    print(f"Starting grid search with {len(param_combinations)} combinations...")
    chunks = np.array_split(param_combinations, total_procs)
else:
    chunks = None

# === Distribute Work ===
local_combinations = comm.scatter(chunks, root=0)

print(f"Rank {rank} received {len(local_combinations)} combinations.")

# === Training & Evaluation ===
model = xgb.XGBClassifier(random_state=42)

local_results = []
start_time = time.time()

for idx, combination in enumerate(local_combinations, 1):
    params = {
        'n_estimators': int(combination[0]),
        'learning_rate': combination[1],
        'max_depth': int(combination[2]),
        'subsample': combination[3],
        'colsample_bytree': combination[4],
        'reg_lambda': int(combination[5]),
        'random_state': 42,
    }

    model.set_params(**params)

    model.fit(X_train, y_train_encoded, verbose=False)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    elapsed = time.time() - start_time

    local_results.append([
        f"{rank}-{idx}",
        params['n_estimators'],
        params['learning_rate'],
        params['max_depth'],
        params['subsample'],
        params['colsample_bytree'],
        params['reg_lambda'],
        elapsed,
        accuracy
    ])

    print(f"Rank {rank}: {idx}/{len(local_combinations)} | Accuracy: {accuracy:.6f} | Elapsed: {elapsed:.2f}s")

# === Gather Results ===
results = comm.gather(local_results, root=0)

# === Save Final Results ===
if rank == 0:
    all_results = []
    for chunk in results:
        all_results.extend(chunk)

    results_df = pd.DataFrame(all_results, columns=[
        'combination', 'n_estimators', 'learning_rate', 'max_depth',
        'subsample', 'colsample_bytree', 'reg_lambda', 'elapsed', 'accuracy'
    ])

    log_file = "grid_search_classification_log.csv"
    results_df.to_csv(log_file, index=False)

    best_result = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"Best Accuracy: {best_result['accuracy']:.6f}")
    print("Best Hyperparameters:")
    print(best_result)

MPI.Finalize()
