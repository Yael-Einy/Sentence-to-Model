
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from src.dataset_with_indices_for_full_and_partial_data import Index_Dataset
from src.dataset import encode_categorical_features
from src.filtering_methods.filtering_methods_return_index import (
    RandomSampleFilter,
    FilterEachIterXgboostPathSampleFinal,
    ActiveLearningUncertaintyFilter,
    ActiveLearningStartFromCoreSet,
ActiveLearningStartFromCoreSet_based_on_reccommended_sampling_sizes,
ActiveLearningStartFromCoreSet_with_class_balance
)

import os
import json
import pandas as pd
import numpy as np

def modify_datasets(input_base_folder, output_base_folder, target_cols_dict, p=0.5, q=0.0):
    #p = % of columns to remove
    #q = % of values to remove from remaining columns
    print(p, q)
    if not (0 <= p <= 1) or not (0 <= q <= 1):
        raise ValueError("p and q must be between 0 and 1.")

    # Ensure the input base folder exists
    if not os.path.exists(input_base_folder):
        raise FileNotFoundError(f"The input folder {input_base_folder} does not exist.")

    # Ensure the output base folder exists
    os.makedirs(output_base_folder, exist_ok=True)

    for dataset_name in os.listdir(input_base_folder):
        dataset_folder = os.path.join(input_base_folder, dataset_name)
        data_file = os.path.join(dataset_folder, "data.csv")

        if not os.path.isfile(data_file):
            print(f"Skipping {dataset_name}: data.csv not found.")
            continue

        df = pd.read_csv(data_file)

        if not target_cols_dict[dataset_name]:
            print("Error on target_cols_dict", dataset_name)
        target_col = target_cols_dict[dataset_name]

        if target_col not in df.columns:
            print(f"Skipping {dataset_name}: target column '{target_cols_dict}' not found.")
            continue

        # Step 1: Remove columns (excluding the target column)
        cols_to_consider = [col for col in df.columns if col != target_col]
        num_cols_to_remove = int(p * len(cols_to_consider))
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




def get_datasets_dict(modified_datasets_folder):
    """
    Reads datasets from the modified folder and retrieves their target column names from metadata.json.
    Ensures the folder exists before attempting to access its contents.
    """
    # Ensure the directory exists before listing its contents
    os.makedirs(modified_datasets_folder, exist_ok=True)

    dataset_dict = {}

    for dataset_name in os.listdir(modified_datasets_folder):
        dataset_folder = os.path.join(modified_datasets_folder, dataset_name)
        full_path = os.path.join(dataset_folder, "full.csv")
        partial_path = os.path.join(dataset_folder, "partial.csv")
        dataset_dict[dataset_name] = {
            'full_path': full_path,
            'partial_path': partial_path,
            'target_col': target_cols_dict[dataset_name]
        }

    return dataset_dict



def evaluate_datasets(datasets, p, q, core_set_sizes=[1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000],
                      random_state=42):
    all_results = []

    #for dataset_name, dataset_info in reversed(list(datasets.items())): #TODO: Un-reverse it
    for dataset_name, dataset_info in datasets.items():
        print(f"Processing dataset: {dataset_name}")

        # Read datasets
        try:
            df_full = pd.read_csv(dataset_info['full_path'])
            df_partial = pd.read_csv(dataset_info['partial_path'])
        except FileNotFoundError as e:
            print(f"Error: File not found for dataset '{dataset_name}'. {e}")
            continue  # Skip this dataset
        except Exception as e:
            print(f"Error: Unable to read files for dataset '{dataset_name}'. {e}")
            continue  # Skip this dataset

        # Validate or fallback to last column as target_col
        target_col = target_cols_dict.get(dataset_name, None)

        # Check again if the target column exists in both dataframes
        if target_col not in df_full.columns or target_col not in df_partial.columns:
            print(f"Warning: Target column '{target_col}' does not exist in one of the datasets.")
            # Fallback to the last column in df_full
            target_col = df_full.columns[-1]
            print(f"Fallback: Setting the target column to the last column: '{target_col}'")
        if target_col not in df_full.columns or target_col not in df_partial.columns:
                print(f"Available columns in full dataset: {df_full.columns.tolist()}")
                print(f"Available columns in partial dataset: {df_partial.columns.tolist()}")
                continue  # Skip this dataset

        # Encode categorical features and split datasets
        df_full = encode_categorical_features(df_full)
        df_partial = encode_categorical_features(df_partial)

        try:
            X_train_full, X_test_full, y_train, y_test = train_test_split(
                df_full.drop(target_col, axis=1),
                df_full[target_col],
                test_size=0.2, shuffle=False,
                random_state=random_state)
        except KeyError as e:
            print(f"Error during train/test split for dataset '{dataset_name}': {e}")
            continue  # Skip this dataset

        dataset_full = Index_Dataset(df_full, X_train_full, y_train, X_test_full, y_test, target_col, f1_score)

        try:
            X_train_partial, X_test_partial, y_train, y_test = train_test_split(
                df_partial.drop(target_col, axis=1),
                df_partial[target_col],
                test_size=0.2, shuffle=False,
                random_state=random_state)
        except KeyError as e:
            print(f"Error during train/test split for partial dataset '{dataset_name}': {e}")
            continue  # Skip this dataset

        dataset_partial = Index_Dataset(df_partial, X_train_partial, y_train, X_test_partial, y_test, target_col,
                                        f1_score)

        total_train_size = len(dataset_full.X_train)

        n=len(dataset_partial.X_train)
        core_set_sizes=[n//20, n//10, n//5, n//2, int(0.75*n)]
        modeling_attitudes_to_test =[1]
        for attitude in modeling_attitudes_to_test:
            for size in core_set_sizes:
                actual_size = min(size, total_train_size)
                model = xgb.XGBClassifier(eval_metric='logloss', random_state=random_state)

                # Random Sampling Filter
                filter_instance = RandomSampleFilter(name='RandomSampleFilter',
                                                     p=actual_size/n, model=model)
                print((actual_size/len(dataset_partial.X_train)))
                filter_instance.test_indices_filter_method(dataset_partial, dataset_full, print_results=True)
                all_results.append({
                    'Dataset': dataset_name,
                    'Method': 'Random Sampling',
                    'Core Set Size': actual_size,
                    'F1 Score': filter_instance.scores,
                    #'Trial': filter_instance.trials_number,
                    #'Data Used (%)': filter_instance.new_size_percents * 100
                })


                # CoreTab Filter
                filter_instance = FilterEachIterXgboostPathSampleFinal('test_each_iter', trees_to_stop=30, threshold=5,
                                                                       examples_to_keep=actual_size, prediction_model=model)
                filter_instance.test_indices_filter_method(dataset_partial, dataset_full, print_results=True)
                all_results.append({
                    'Dataset': dataset_name,
                    'Method': 'CoreTab',
                    'Core Set Size': actual_size,
                    'F1 Score': filter_instance.scores,
                    #'Trial': filter_instance.trials_number,
                    #'Data Used (%)': filter_instance.new_size_percents * 100
                })

                initial_set_size = int(actual_size*0.7)

                # Active Learning Filter
                filter_instance = ActiveLearningUncertaintyFilter('Active Learning', number_of_examples=actual_size,
                                                                  model=model, start_size=initial_set_size,
                                                                  batch_size=actual_size // 20)
                filter_instance.test_indices_filter_method(dataset_partial, dataset_full, print_results=True)
                all_results.append({
                    'Dataset': dataset_name,
                    'Method': 'Active Learning',
                    'Core Set Size': actual_size,
                    'F1 Score': filter_instance.scores,
                    #'Trial': filter_instance.trials_number,
                    #'Data Used (%)': filter_instance.new_size_percents * 100
                })

                # ExtendTAB Filter(Active Learning Start From Core Set)
                core_tab = FilterEachIterXgboostPathSampleFinal(
                    'test_each_iter',
                    trees_to_stop=30,
                    threshold=5,
                    examples_to_keep=initial_set_size,
                    prediction_model=model
                )
                core_set_sampler = core_tab.sample_indices(
                    dataset_partial,
                    p=initial_set_size / len(dataset_full.df)
                )

                filter_instance = ActiveLearningStartFromCoreSet(
                    name='ExtendTAB',
                    number_of_examples=actual_size,
                    model=model,
                    start_size=initial_set_size,
                    core_set_sampler=core_set_sampler,
                    batch_size=int(actual_size // 20)
                )

                # The critical fix: actually run the filter method to populate filter_instance.scores
                filter_instance.test_indices_filter_method(dataset_partial, dataset_full, print_results=True)

                all_results.append({
                    'Dataset': dataset_name,
                    'Method': 'Extentab',
                    'Core Set Size': actual_size,
                    'F1 Score': filter_instance.scores,
                    # 'Trial': filter_instance.trials_number,
                    # 'Data Used (%)': filter_instance.new_size_percents * 100
                })

                # Balanced ExtendTAB Filter(Active Learning Start From Core Set)
                filter_instance = ActiveLearningStartFromCoreSet_with_class_balance(
                    name='Balanced ExtendTAB',
                    number_of_examples=actual_size,
                    model=model,
                    start_size=initial_set_size,
                    core_set_sampler=core_set_sampler,
                    batch_size=int(actual_size // 20)
                )

                # The critical fix: actually run the filter method to populate filter_instance.scores
                filter_instance.test_indices_filter_method(dataset_partial, dataset_full, print_results=True)

                all_results.append({
                    'Dataset': dataset_name,
                    'Method': 'Balanced Extentab',
                    'Core Set Size': actual_size,
                    'F1 Score': filter_instance.scores,
                    # 'Trial': filter_instance.trials_number,
                    # 'Data Used (%)': filter_instance.new_size_percents * 100
                })

            # ===============================
            #   ExtendTAB (Free size)
            # ===============================
            core_tab_hard_Samples = FilterEachIterXgboostPathSampleFinal(
                'test_each_iter',
                trees_to_stop=30,
                threshold=5,
                prediction_model=model
            )
            core_set_sampler = core_tab_hard_Samples.sample_indices(
                dataset_partial,
                p=initial_set_size / len(dataset_full.X_train)
            )

            filter_instance = ActiveLearningStartFromCoreSet_based_on_reccommended_sampling_sizes(
                name='ExtendTAB',
                number_of_examples=actual_size,
                model=model,
                core_set_sampler=core_set_sampler,
                batch_size=int(actual_size // 20)
            )

            # Again, the critical fix: call the method
            filter_instance.test_indices_filter_method(dataset_partial, dataset_full, print_results=True)

            all_results.append({
                'Dataset': dataset_name,
                'Method': 'Extentab Free size',
                'Core Set Size': actual_size,
                'F1 Score': filter_instance.scores,
                # 'Trial': filter_instance.trials_number,
                # 'Data Used (%)': filter_instance.new_size_percents * 100
            })

        # End of for loops, etc.
            # ------------------------------------------------------------------
            # return or save all_results (your normal code)
            # ---------------------------------------

        # Convert results to a DataFrame and save
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(
            f'filtering_results_based_on_dataset_size_100b_55m_c33_{p}of_cols__removed_{q}_of_values_coreset{initial_set_size/n}_1trials.csv',
            index=False)

    return results_df


# Example Execution
input_folder = r"C:\Users\Ezra\PycharmProjects\POCA\data"  # Original dataset location
output_folder = r"C:\Users\Ezra\PycharmProjects\POCA\modified_data"

# Ensure the directory exists before modifying or reading datasets
os.makedirs(output_folder, exist_ok=True)

target_cols_dict = {"diabetes":"Diabetes_binary","adult":"y", "banking":"y", "cardio":'cardio',"housing":'y', "magic":"y", "bank_fraud":"fraud_bool", "credit_card": "Class"}
for p, q in [(0.5,0.0), (0.0, 0.5), (0.25, 0.25)]:
    modify_datasets(input_folder, output_folder, target_cols_dict,p, q)
    datasets_dict = get_datasets_dict(output_folder)

    # Run evaluation
    results = results = evaluate_datasets(datasets_dict, p, q)

    print(results)
