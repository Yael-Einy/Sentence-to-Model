import importlib
import src.filtering_methods.filtering_methods_return_index as filter_module

importlib.reload(filter_module)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from src.dataset_with_indices_for_full_and_partial_data import Index_Dataset
from src.dataset import encode_categorical_features
from src.filtering_methods.filtering_methods_return_index import (
    RandomSampleFilter,
    FilterEachIterXgboostPathSampleFinal,
    ActiveLearningStartFromCoreSet_based_on_reccommended_sampling_sizes
)
import os
import json
import pandas as pd
import numpy as np

def modify_datasets(input_base_folder, output_base_folder, target_cols_dict, p=0.5, q=0.0):
    """
    p = % of columns to remove
    q = % of values to remove from remaining columns
    """
    print(f"Modifying datasets with p={p}, q={q}")
    if not (0 <= p <= 1) or not (0 <= q <= 1):
        raise ValueError("p and q must be between 0 and 1.")

    # Ensure the input base folder exists
    if not os.path.exists(input_base_folder):
        raise FileNotFoundError(f"The input folder {input_base_folder} does not exist.")

    # Ensure the output base folder exists
    os.makedirs(output_base_folder, exist_ok=True)

    for dataset_name in os.listdir(input_base_folder):
        dataset_folder = os.path.join(input_base_folder, dataset_name)
        
        # Skip if not a directory
        if not os.path.isdir(dataset_folder):
            continue
            
        data_file = os.path.join(dataset_folder, "data.csv")

        if not os.path.isfile(data_file):
            print(f"Skipping {dataset_name}: data.csv not found.")
            continue

        df = pd.read_csv(data_file)

        if dataset_name not in target_cols_dict:
            print(f"Warning: {dataset_name} not in target_cols_dict, skipping.")
            continue
            
        target_col = target_cols_dict[dataset_name]

        if target_col not in df.columns:
            print(f"Skipping {dataset_name}: target column '{target_col}' not found.")
            continue

        # Step 1: Remove columns (excluding the target column)
        cols_to_consider = [col for col in df.columns if col != target_col]
        num_cols_to_remove = int(p * len(cols_to_consider))
        
        np.random.seed(42)  # For reproducibility
        cols_to_remove = np.random.choice(cols_to_consider, num_cols_to_remove, replace=False)
        df_reduced = df.drop(columns=cols_to_remove)

        # Step 2: Remove values from the remaining dataset (excluding target column)
        remaining_cols = [col for col in df_reduced.columns if col != target_col]
        num_values_to_remove = int(q * len(remaining_cols) * len(df))

        for _ in range(num_values_to_remove):
            col = np.random.choice(remaining_cols)
            row = np.random.randint(0, len(df_reduced))
            df_reduced.at[row, col] = np.nan  # Set to NaN

        # Create output dataset folder
        output_dataset_folder = os.path.join(output_base_folder, dataset_name)
        os.makedirs(output_dataset_folder, exist_ok=True)

        # Save original and modified datasets
        df.to_csv(os.path.join(output_dataset_folder, "full.csv"), index=False)
        df_reduced.to_csv(os.path.join(output_dataset_folder, "partial.csv"), index=False)

        # Save metadata with target column information
        metadata = {"target_col": target_col}
        with open(os.path.join(output_dataset_folder, "metadata.json"), "w") as meta_file:
            json.dump(metadata, meta_file)

        print(f"Processed {dataset_name}: {num_cols_to_remove} columns removed, {num_values_to_remove} values removed.")


def get_datasets_dict(modified_datasets_folder, target_cols_dict):
    """
    Reads datasets from the modified folder and retrieves their target column names from metadata.json.
    Ensures the folder exists before attempting to access its contents.
    """
    # Ensure the directory exists before listing its contents
    os.makedirs(modified_datasets_folder, exist_ok=True)

    dataset_dict = {}

    for dataset_name in os.listdir(modified_datasets_folder):
        if dataset_name not in target_cols_dict:
            print(f"Skipping {dataset_name}: not in target_cols_dict")
            continue
            
        dataset_folder = os.path.join(modified_datasets_folder, dataset_name)
        
        # Skip if not a directory
        if not os.path.isdir(dataset_folder):
            continue
            
        full_path = os.path.join(dataset_folder, "full.csv")
        partial_path = os.path.join(dataset_folder, "partial.csv")
        
        if not os.path.exists(full_path) or not os.path.exists(partial_path):
            print(f"Skipping {dataset_name}: full.csv or partial.csv not found.")
            continue
            
        dataset_dict[dataset_name] = {
            'full_path': full_path,
            'partial_path': partial_path,
            'target_col': target_cols_dict[dataset_name]
        }

    return dataset_dict


def evaluate_datasets(datasets, p, q, breaking_treshold, target_cols_dict, random_state=42):
    all_results = []

    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")

        # Read datasets
        try:
            df_full = pd.read_csv(dataset_info['full_path'])
            df_partial = pd.read_csv(dataset_info['partial_path'])
        except FileNotFoundError as e:
            print(f"Error: File not found for dataset '{dataset_name}'. {e}")
            continue
        except Exception as e:
            print(f"Error: Unable to read files for dataset '{dataset_name}'. {e}")
            continue

        # Validate or fallback to last column as target_col
        target_col = target_cols_dict.get(dataset_name, None)

        # Check again if the target column exists in both dataframes
        if target_col not in df_full.columns or target_col not in df_partial.columns:
            print(f"Warning: Target column '{target_col}' does not exist in one of the datasets.")
            target_col = df_full.columns[-1]
            print(f"Fallback: Setting the target column to the last column: '{target_col}'")
            
        if target_col not in df_full.columns or target_col not in df_partial.columns:
            print(f"Available columns in full dataset: {df_full.columns.tolist()}")
            print(f"Available columns in partial dataset: {df_partial.columns.tolist()}")
            continue

        # Encode categorical features and split datasets
        df_full = encode_categorical_features(df_full)
        df_partial = encode_categorical_features(df_partial)

        try:
            X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
                df_full.drop(target_col, axis=1),
                df_full[target_col],
                test_size=0.2, 
                shuffle=False,
                random_state=random_state)
        except KeyError as e:
            print(f"Error during train/test split for dataset '{dataset_name}': {e}")
            continue

        dataset_full = Index_Dataset(df_full, X_train_full, y_train_full, X_test_full, y_test_full, target_col, f1_score)

        try:
            X_train_partial, X_test_partial, y_train_partial, y_test_partial = train_test_split(
                df_partial.drop(target_col, axis=1),
                df_partial[target_col],
                test_size=0.2, 
                shuffle=False,
                random_state=random_state)
        except KeyError as e:
            print(f"Error during train/test split for partial dataset '{dataset_name}': {e}")
            continue

        dataset_partial = Index_Dataset(df_partial, X_train_partial, y_train_partial, X_test_partial, y_test_partial, target_col, f1_score)

        total_train_size = len(dataset_full.X_train)
        n = len(dataset_partial.X_train)
        
        print(f"Full dataset training size: {total_train_size}")
        print(f"Partial dataset training size: {n}")
        
        core_set_sizes = [n//20, n//10, n//5, n//2, int(0.75*n), n]
        modeling_attitudes_to_test = [1]
        
        for attitude in modeling_attitudes_to_test:
            for size in core_set_sizes:
                print(f"\n--- Testing with core set size: {size} ---")
                
                initial_set_size = int(0.25*size)
                actual_size = min(size, total_train_size)
                model = xgb.XGBClassifier(eval_metric='logloss', random_state=random_state)
                
                # ===============================
                #   ExtendTAB (Free size)
                # ===============================
                print("Running ExtendTAB method...")
                core_tab_hard_Samples = FilterEachIterXgboostPathSampleFinal(
                    'test_each_iter',
                    trees_to_stop=30,
                    threshold=5,
                    prediction_model=model
                )
                
                try:
                    core_set_sampler = core_tab_hard_Samples.sample_indices(
                        dataset_partial,
                        p=initial_set_size / len(dataset_full.X_train)
                    )
                except Exception as e:
                    print(f"Error in ExtendTAB sampling: {e}")
                    core_set_sampler = None

                if core_set_sampler is None or len(core_set_sampler) == 0:
                    print("ExtendTAB returned no samples, using fallback stratified sampling")
                    sss = StratifiedShuffleSplit(n_splits=1,
                                                 train_size=initial_set_size,
                                                 random_state=55)
                    train_idx, _ = next(sss.split(dataset_partial.X_train,
                                                  dataset_partial.y_train))
                    core_set_sampler = dataset_partial.X_train.index[train_idx].tolist()
                else:
                    core_set_sampler = list(core_set_sampler)

                # Clean the core set sampler
                core_set_sampler = [int(i) for i in core_set_sampler
                                    if i is not None and not (isinstance(i, float) and np.isnan(i))]

                core_set_sampler = pd.Index(core_set_sampler)
                print(f"Core set size: {len(core_set_sampler)}")

                # Initialize the ActiveLearning filter with the core set
                filter_instance = ActiveLearningStartFromCoreSet_based_on_reccommended_sampling_sizes(
                    name='ExtendTAB',
                    number_of_examples=actual_size,
                    model=xgb.XGBClassifier(eval_metric='logloss', random_state=random_state),
                    core_set_sampler=core_set_sampler,
                    batch_size=int(actual_size // 20)
                )

                try:
                    # Test the method
                    filter_instance.test_indices_filter_method(dataset_partial, dataset_full, print_results=True)
                    chosen_size = filter_instance.size
                    
                    all_results.append({
                        'Dataset': dataset_name,
                        'Method': 'ExtendTAB Free size',
                        'Core Set Size': chosen_size,
                        'F1 Score': filter_instance.scores[0] if filter_instance.scores else None,
                    })
                    print(f"ExtendTAB F1 Score: {filter_instance.scores[0] if filter_instance.scores else 'N/A'}")
                except Exception as e:
                    print(f"Error testing ExtendTAB method: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                # ===============================
                #   Random Sampling
                # ===============================
                print("\nRunning Random Sampling method...")
                try:
                    filter_instance_random = RandomSampleFilter(
                        name='RandomSampleFilter',
                        p=chosen_size/n, 
                        model=xgb.XGBClassifier(eval_metric='logloss', random_state=random_state)
                    )
                    filter_instance_random.test_indices_filter_method(dataset_partial, dataset_full, print_results=True)
                    
                    all_results.append({
                        'Dataset': dataset_name,
                        'Method': 'Random Sampling',
                        'Core Set Size': chosen_size,
                        'F1 Score': filter_instance_random.scores[0] if filter_instance_random.scores else None,
                    })
                    print(f"Random Sampling F1 Score: {filter_instance_random.scores[0] if filter_instance_random.scores else 'N/A'}")
                except Exception as e:
                    print(f"Error testing Random Sampling method: {e}")
                    import traceback
                    traceback.print_exc()

    # Convert results to a DataFrame and save
    if all_results:
        results_data = pd.DataFrame(all_results)
        output_filename = f'ExtendTAB_free_size_all_datasets_breaking_treshold{breaking_treshold}_{p}p_{q}q.csv'
        results_data.to_csv(output_filename, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_filename}")
        print(f"{'='*60}")
        return results_data
    else:
        print("No results to save.")
        return pd.DataFrame()


# Example Execution
if __name__ == "__main__":
    input_folder = r"C:\Users\Ezra\PycharmProjects\POCA\data"
    output_folder = r"C:\Users\Ezra\PycharmProjects\POCA\all_datasets_modified_data"

    # Ensure the directory exists before modifying or reading datasets
    os.makedirs(output_folder, exist_ok=True)

    # ALL DATASETS - Add target columns for all your datasets here
    target_cols_dict = {
        "diabetes": "Diabetes_binary",
        "adult": "y", 
        "banking": "y", 
        "cardio": 'cardio',
        "housing": 'y', 
        "magic": "y", 
        "bank_fraud": "fraud_bool", 
        "credit_card": "Class",
        "creditcard": "Class"  # Alternative name
    }
    
    # Test all three configurations
    configurations = [(0.5, 0.0), (0.0, 0.5), (0.25, 0.25)]
    breaking_treshold = 0.005
    
    for p, q in configurations:
        print(f"\n{'='*60}")
        print(f"Starting evaluation with p={p}, q={q}")
        print(f"ALL DATASETS MODE")
        print(f"{'='*60}")
        
        modify_datasets(input_folder, output_folder, target_cols_dict, p, q)
        datasets_dict = get_datasets_dict(output_folder, target_cols_dict)
        
        if not datasets_dict:
            print("No datasets found to process!")
        else:
            print(f"Found {len(datasets_dict)} datasets to process: {list(datasets_dict.keys())}")
            results = evaluate_datasets(datasets_dict, p, q, breaking_treshold, target_cols_dict)
            print(f"\nFinal Results for p={p}, q={q}:")
            print(results)
            print("\n")
