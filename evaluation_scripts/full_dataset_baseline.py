import importlib
import src.filtering_methods.filtering_methods_return_index as filter_module

importlib.reload(filter_module)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.dataset_with_indices_for_full_and_partial_data import Index_Dataset
from src.dataset import encode_categorical_features
from src.filtering_methods.filtering_methods_return_index import RandomSampleFilter
import os
import pandas as pd


def get_datasets_dict(input_base_folder, target_cols_dict):
    """
    Reads datasets from the input folder.
    """
    dataset_dict = {}

    for dataset_name in os.listdir(input_base_folder):
        if dataset_name not in target_cols_dict:
            print(f"Skipping {dataset_name}: not in target_cols_dict")
            continue

        dataset_folder = os.path.join(input_base_folder, dataset_name)

        # Skip if not a directory
        if not os.path.isdir(dataset_folder):
            continue

        data_file = os.path.join(dataset_folder, "data.csv")

        if not os.path.exists(data_file):
            print(f"Skipping {dataset_name}: data.csv not found.")
            continue

        dataset_dict[dataset_name] = {
            'data_path': data_file,
            'target_col': target_cols_dict[dataset_name]
        }

    return dataset_dict


def train_full_datasets(datasets, target_cols_dict, random_state=42):
    """
    Train models on FULL datasets (no filtering, no core set selection).
    Uses entire training set for modeling.
    """
    all_results = []
    failed_datasets = []
    successful_datasets = []

    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'=' * 60}")

        try:
            # Read dataset
            try:
                df = pd.read_csv(dataset_info['data_path'])
            except FileNotFoundError as e:
                print(f"❌ Error: File not found - {e}")
                failed_datasets.append((dataset_name, "File not found"))
                continue
            except Exception as e:
                print(f"❌ Error: Unable to read file - {e}")
                failed_datasets.append((dataset_name, f"Read error: {str(e)[:50]}"))
                continue

            target_col = target_cols_dict.get(dataset_name, None)

            # Check if target column exists
            if target_col not in df.columns:
                print(f"❌ Warning: Target column '{target_col}' not found in dataset")
                failed_datasets.append((dataset_name, "Target column missing"))
                continue

            print(f"Dataset shape: {df.shape}")
            print(f"Target column: {target_col}")

            # Encode categorical features
            df = encode_categorical_features(df)

            # Train/test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    df.drop(target_col, axis=1),
                    df[target_col],
                    test_size=0.2,
                    shuffle=False,
                    random_state=random_state)
            except KeyError as e:
                print(f"❌ Error during train/test split: {e}")
                failed_datasets.append((dataset_name, f"Split error: {e}"))
                continue
            except Exception as e:
                print(f"❌ Unexpected error during split: {e}")
                failed_datasets.append((dataset_name, f"Split error: {str(e)[:50]}"))
                continue

            dataset = Index_Dataset(df, X_train, y_train, X_test, y_test, target_col, f1_score)

            total_train_size = len(dataset.X_train)
            print(f"Training size: {total_train_size}, Test size: {len(dataset.X_test)}")

            # ===============================
            #   Full Dataset Training (Random Sample with p=1.0)
            # ===============================
            print("Training on FULL dataset (100% random sample)...")

            model = xgb.XGBClassifier(eval_metric='logloss', random_state=random_state, verbosity=0)

            # Use RandomSampleFilter with p=1.0 to select 100% of the training set
            random_sampler = RandomSampleFilter(
                name='Full_Dataset_Random',
                p=1.0,
                model=model
            )

            # Test using the entire dataset (suppress detailed output)
            try:
                random_sampler.test_indices_filter_method(dataset, dataset, print_results=False)
            except KeyError as ke:
                # Handle pandas index errors without printing massive error messages
                error_str = str(ke)
                if "not in index" in error_str:
                    print(f"❌ Error: Index mismatch in dataset")
                    failed_datasets.append((dataset_name, "Index mismatch"))
                else:
                    print(f"❌ Error: KeyError - {error_str[:100]}")
                    failed_datasets.append((dataset_name, f"KeyError: {error_str[:30]}"))
                continue
            except Exception as train_ex:
                print(f"❌ Error during training: {str(train_ex)[:100]}")
                failed_datasets.append((dataset_name, f"Training error: {str(train_ex)[:30]}"))
                continue

            if random_sampler.scores and len(random_sampler.scores) > 0:
                f1_score_value = random_sampler.scores[0]
                all_results.append({
                    'Dataset': dataset_name,
                    'Training Set Size': total_train_size,
                    'Test Set Size': len(dataset.X_test),
                    'F1 Score': f1_score_value,
                    'Data Used (%)': 100.0
                })

                print(f"✅ F1 Score: {f1_score_value:.4f}")
                successful_datasets.append(dataset_name)
            else:
                print(f"❌ Error: No F1 score computed")
                failed_datasets.append((dataset_name, "No F1 score"))
                continue

        except Exception as e:
            print(f"❌ Unexpected error processing dataset: {e}")
            failed_datasets.append((dataset_name, f"Unexpected: {str(e)[:50]}"))
            continue

    # Print summary
    print(f"\n{'=' * 60}")
    print("EXECUTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"✅ Successfully processed: {len(successful_datasets)} datasets")
    if successful_datasets:
        print(f"   {', '.join(successful_datasets)}")
    print(f"❌ Failed/Skipped: {len(failed_datasets)} datasets")
    if failed_datasets:
        for ds_name, reason in failed_datasets:
            print(f"   - {ds_name}: {reason}")
    print(f"{'=' * 60}\n")

    # Convert results to a DataFrame and save
    if all_results:
        results_data = pd.DataFrame(all_results)
        output_filename = 'full_datasets_baseline_results.csv'
        results_data.to_csv(output_filename, index=False)
        print(f"{'=' * 60}")
        print(f"Results saved to: {output_filename}")
        print(f"{'=' * 60}")
        return results_data
    else:
        print("⚠️ Warning: No results to save.")
        return pd.DataFrame()


# Example Execution
if __name__ == "__main__":
    input_folder = r"C:\Users\Ezra\PycharmProjects\POCA\data"

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

    print(f"\n{'=' * 60}")
    print(f"FULL DATASET BASELINE - Training on 100% of data")
    print(f"{'=' * 60}")

    datasets_dict = get_datasets_dict(input_folder, target_cols_dict)

    if not datasets_dict:
        print("No datasets found to process!")
    else:
        print(f"Found {len(datasets_dict)} datasets to process: {list(datasets_dict.keys())}")
        results = train_full_datasets(datasets_dict, target_cols_dict)
        print("\nFinal Results:")
        print(results)
