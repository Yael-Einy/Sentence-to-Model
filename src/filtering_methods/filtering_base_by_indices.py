from src.dataset import Dataset
from src.dataset_with_indices_for_full_and_partial_data import Index_Dataset
from sklearn.base import clone

from abc import ABC
from dataclasses import dataclass

from sklearn.model_selection import KFold
import time
import xgboost as xgb
import pandas as pd
from dataclasses import dataclass
import numpy as np


@dataclass
class FilteringResults:
    modeling_atitude: str
    score: float #score of full dataset evaluation
    filtered_score: float #score of partial dataset evaluation
    run_time: int #the expirement number
    new_size_percent: float


@dataclass
class FilteringIndices:
    indices: pd.Index  # pd.Index will store the index of filtered rows
    modeling_atitude: str
    score: float #score of full dataset evaluation
    filtered_score: float #score of partial dataset evaluation
    run_time: int #the expirement number
    new_size_percent: float

class FilteringExperiment(ABC):

    def __init__(self, name):

        self.name = name

        self.scores = []
        self.scores_filtered = []
        self.run_times = []
        self.new_size_percents = []

        self.trials_number = None

        self.save_each_iter = None
        self.iter_results = None

        self.trials_number = None
        self.modeling_attitude = None

    def sample_func(self, dataset: Dataset, p) -> FilteringResults:
        pass

    def sample_indices(self, dataset: Index_Dataset, p) -> FilteringIndices:
        pass

    def reset_attributes(self) -> None:
        pass

    def test_filter_method(self,
                           dataset: Dataset,
                           trials_number=10,
                           print_results=False,
                           save_each_iter=True):

        self.trials_number = trials_number
        self.save_each_iter = save_each_iter

        for _ in range(self.trials_number):
            results = self.sample_func(dataset, save_each_iter)
            if print_results:
                print(results)
            self.scores.append(results.score)
            self.run_times.append(results.run_time)
            self.scores_filtered.append(results.filtered_score)
            self.new_size_percents.append(results.new_size_percent)

    def test_method_cv(self,
                       dataset: Dataset,
                       trials_number=3,
                       cv=4,
                       print_results=False,
                       save_each_iter=True,
                       random_seed=100):

        self.trials_number = trials_number
        self.save_each_iter = save_each_iter

        for repeat in range(trials_number):
            # Initialize a list to store fold scores for this repeat
            fold_scores = []

            # Create a k-fold cross-validation object for this repeat
            kf = KFold(n_splits=cv, shuffle=True, random_state=random_seed + repeat)

            # Inner loop for folds
            for train_index, test_index in kf.split(dataset.df):
                self.reset_attributes()
                X, y = dataset.df.drop(dataset.target_col, axis=1), dataset.df[dataset.target_col]
                X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                trial_dataset = Dataset(df=dataset.df, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                        target_col=dataset.target_col, metric=dataset.metric)
                results = self.sample_func(trial_dataset, save_each_iter)
                if print_results:
                    print(results)
                self.scores.append(results.score)
                self.run_times.append(results.run_time)
                self.scores_filtered.append(results.filtered_score)
                self.new_size_percents.append(results.new_size_percent)

    def test_indices_filter_method(self,
                                   dataset_partial: Index_Dataset,
                                   dataset_full: Index_Dataset,
                                   trials_number=1,
                                   print_results=False,
                                   save_each_iter=False):
        self.trials_number = trials_number
        self.save_each_iter = save_each_iter

        for repeat in range(trials_number):
            # Step 1: Sample indices for the subset of the dataset
            core_set_indices = self.sample_indices(dataset_partial, save_each_iter)
            core_set = dataset_full.df.loc[core_set_indices]

            # Step 2: Separate features (X) and labels (y) for the filtered core set
            self.X_train_filtered = core_set.drop(dataset_full.target_col, axis=1)
            self.y_train_filtered = dataset_full.y_train[self.X_train_filtered.index]

            # Step 3: Validate presence of both classes in y_train_filtered
            unique_classes = np.unique(self.y_train_filtered)
            all_classes = np.unique(dataset_full.df[dataset_full.target_col])

            missing_classes = set(all_classes) - set(unique_classes)
            if missing_classes:
                for missing_class in missing_classes:
                    missing_samples = dataset_full.df[dataset_full.df[dataset_full.target_col] == missing_class].sample(
                        n=min(15, len(dataset_full.df[dataset_full.target_col] == missing_class)),
                        random_state=100 + repeat
                    )
                    self.X_train_filtered = pd.concat(
                        [self.X_train_filtered, missing_samples.drop(dataset_full.target_col, axis=1)]
                    )
                    self.y_train_filtered = pd.concat(
                        [self.y_train_filtered, missing_samples[dataset_full.target_col]]
                    )

            # Ensure consistent feature ordering
            feature_columns = self.X_train_filtered.columns.tolist()

            # Reorder features in both training and test data
            self.X_train_filtered = self.X_train_filtered[sorted(feature_columns)]
            X_test_ordered = dataset_full.X_test[sorted(feature_columns)]

            # Step 4: Train the prediction model
            if self.prediction_model is None:
                self.prediction_model = xgb.XGBClassifier()

            self.prediction_model.fit(self.X_train_filtered, self.y_train_filtered)

            # Step 5: Evaluate predictions using properly ordered features
            preds = self.prediction_model.predict(X_test_ordered)

            results = FilteringResults(
                modeling_atitude="sample by partial dataset, train and test by the full dataset",
                score=dataset_full.metric(dataset_full.y_test, preds),
                run_time=repeat,
                filtered_score=dataset_full.metric(dataset_full.y_test, preds),
                new_size_percent=len(self.X_train_filtered) / len(dataset_full.X_train))

            if print_results:
                print(results)

            # Store results
            self.scores.append(results.score)
            self.run_times.append(results.run_time)
            self.scores_filtered.append(results.filtered_score)
            self.new_size_percents.append(results.new_size_percent)
            self.size = len(self.X_train_filtered)


    def choose_indices_filter_method_for_wild_data(self,
                                   dataset_partial: Index_Dataset,
                                   #dataset_full: Index_Dataset,
                                   #trials_number=1,
                                   print_results=False,
                                   save_each_iter=False):

        # Step 1: Sample indices for the subset of the dataset
        core_set_indices = self.sample_indices(dataset_partial, save_each_iter)
        #core_set = dataset_partial.df.loc[core_set_indices]
        return core_set_indices


    def test_filter_method_for_wild_data(self,
                                        dataset_full: Index_Dataset,
                                        trials_number=5,
                                        core_set_indices=None,
                                        print_results=False,
                                        save_each_iter=False):

        self.trials_number = trials_number
        self.save_each_iter = save_each_iter

        for repeat in range(trials_number):
            # Step 1: Sample indices for the subset of the dataset
            core_set_indices = self.sample_indices(dataset_full, save_each_iter)
            core_set = dataset_full.df.loc[core_set_indices]

            # Step 2: Separate features (X) and labels (y) for the filtered core set
            self.X_train_filtered = core_set.drop(dataset_full.target_col, axis=1)
            self.y_train_filtered = dataset_full.y_train[self.X_train_filtered.index]

            # Step 3: Validate presence of both classes in y_train_filtered
            unique_classes = np.unique(self.y_train_filtered)
            all_classes = np.unique(dataset_full.df[dataset_full.target_col])

            missing_classes = set(all_classes) - set(unique_classes)
            if missing_classes:
                for missing_class in missing_classes:
                    missing_samples = dataset_full.df[dataset_full.df[dataset_full.target_col] == missing_class].sample(
                        n=min(15, len(dataset_full.df[dataset_full.target_col] == missing_class)),
                        random_state=100 + repeat
                    )
                    self.X_train_filtered = pd.concat(
                        [self.X_train_filtered, missing_samples.drop(dataset_full.target_col, axis=1)]
                    )
                    self.y_train_filtered = pd.concat(
                        [self.y_train_filtered, missing_samples[dataset_full.target_col]]
                    )

            # Ensure consistent feature ordering
            feature_columns = self.X_train_filtered.columns.tolist()

            # Reorder features in both training and test data
            self.X_train_filtered = self.X_train_filtered[sorted(feature_columns)]
            X_test_ordered = dataset_full.X_test[sorted(feature_columns)]

            # Step 4: Train the prediction model
            if self.prediction_model is None:
                self.prediction_model = xgb.XGBClassifier()

            self.prediction_model.fit(self.X_train_filtered, self.y_train_filtered)

            # Step 5: Evaluate predictions using properly ordered features
            preds = self.prediction_model.predict(X_test_ordered)

            results = FilteringResults(
                modeling_atitude="sample by partial dataset, train and test by the full dataset",
                score=dataset_full.metric(dataset_full.y_test, preds),
                run_time=repeat,
                filtered_score=dataset_full.metric(dataset_full.y_test, preds),
                new_size_percent=len(self.X_train_filtered) / len(dataset_full.X_train))

            if print_results:
                print(results)

            # Store results
            self.scores.append(results.score)
            self.run_times.append(results.run_time)
            self.scores_filtered.append(results.filtered_score)
            self.new_size_percents.append(results.new_size_percent)

    def test_indices_filter_method_cv(self,
                                      dataset_partial: Index_Dataset,
                                      dataset_full: Index_Dataset,
                                      trials_number=5,
                                      cv=4,
                                      print_results=False,
                                      save_each_iter=False,
                                      random_seed=100):
        """
        Perform cross-validation using a partial dataset (indices-based) to obtain a core set,
        handle missing classes by adding samples for missing classes,
        and then train and evaluate on the filtered data.

        :param dataset_partial: The partial dataset with indices-based access: to choose core-set samples.
        :param dataset_full: The full dataset from which training and test subsets are taken after being selected.
        :param trials_number: Number of repeated cross-validation runs.
        :param cv: Number of folds in K-Fold cross-validation.
        :param print_results: Whether to print each fold's result.
        :param save_each_iter: Whether to save intermediate results for each iteration.
        :param random_seed: Random seed for reproducibility.
        """
        self.trials_number = trials_number
        self.save_each_iter = save_each_iter

        for repeat in range(trials_number):
            kf = KFold(n_splits=cv, shuffle=True, random_state=random_seed + repeat)

            for train_index, test_index in kf.split(dataset_full.df):
                # Reset collected metrics before each fold
                self.reset_attributes()

                # Prepare the partial dataset for this fold
                X_partial = dataset_partial.df.drop(dataset_partial.target_col, axis=1)
                y_partial = dataset_partial.df[dataset_partial.target_col]

                X_train_partial = X_partial.loc[train_index, :]
                y_train_partial = y_partial[train_index]
                X_test_partial = X_partial.loc[test_index, :]
                y_test_partial = y_partial[test_index]

                partial_dataset_fold = Index_Dataset(
                    df=pd.concat([X_train_partial, y_train_partial], axis=1),
                    X_train=X_train_partial,
                    y_train=y_train_partial,
                    X_test=X_test_partial,
                    y_test=y_test_partial,
                    target_col=dataset_partial.target_col,
                    metric=dataset_partial.metric
                )

                # Prepare the full dataset for this fold
                X_full_fold = dataset_full.df.drop(dataset_full.target_col, axis=1)
                y_full_fold = dataset_full.df[dataset_full.target_col]

                X_train_full_fold = X_full_fold.loc[train_index, :]
                y_train_full_fold = y_full_fold[train_index]
                X_test_full_fold = X_full_fold.loc[test_index, :]
                y_test_full_fold = y_full_fold[test_index]

                full_dataset_fold = Index_Dataset(
                    df=pd.concat([X_train_full_fold, y_train_full_fold], axis=1),
                    X_train=X_train_full_fold,
                    y_train=y_train_full_fold,
                    X_test=X_test_full_fold,
                    y_test=y_test_full_fold,
                    target_col=dataset_full.target_col,
                    metric=dataset_full.metric
                )

                # Obtain the indices to form the core set from the partial training data
                core_set_indices = self.sample_indices(partial_dataset_fold, save_each_iter)

                # Build the core set from the full training data
                core_set = full_dataset_fold.df.loc[core_set_indices]
                self.X_train_filtered = core_set.drop(full_dataset_fold.target_col, axis=1)
                self.y_train_filtered = full_dataset_fold.y_train[self.X_train_filtered.index]

                # Validate the presence of all classes in the filtered dataset
                unique_classes = np.unique(self.y_train_filtered)
                all_classes = np.unique(dataset_full.df[dataset_full.target_col])

                missing_classes = set(all_classes) - set(unique_classes)
                if missing_classes:
                    print(f"Warning: Missing classes detected in filtered dataset: {missing_classes}")
                    for missing_class in missing_classes:
                        # Sample ~3 rows from the full dataset for the missing class
                        missing_samples = dataset_full.df[
                            dataset_full.df[dataset_full.target_col] == missing_class
                            ].sample(
                            n=min(15, len(dataset_full.df[dataset_full.target_col] == missing_class)),
                            random_state=42 + repeat
                        )
                        # Add the sampled rows to the filtered dataset
                        self.X_train_filtered = pd.concat(
                            [self.X_train_filtered, missing_samples.drop(dataset_full.target_col, axis=1)]
                        )
                        self.y_train_filtered = pd.concat(
                            [self.y_train_filtered, missing_samples[dataset_full.target_col]]
                        )
                    print(f"Added samples for missing classes: {missing_classes}")

                # Train or fit model on the filtered training data
                if self.prediction_model is not None:
                    self.prediction_model.fit(self.X_train_filtered, self.y_train_filtered)
                else:
                    self.prediction_model = xgb.XGBClassifier()
                    self.prediction_model.fit(self.X_train_filtered, self.y_train_filtered)

                # Evaluate on the test portion of the full dataset for this fold
                preds = self.prediction_model.predict(full_dataset_fold.X_test)

                results = FilteringResults(
                    modeling_atitude="sample by parial dataset, train and test by the full dataset",
                    score=full_dataset_fold.metric(full_dataset_fold.y_test, preds),
                    run_time=repeat,  # Time spent obtaining the core set
                    filtered_score=full_dataset_fold.metric(full_dataset_fold.y_test, preds),
                    new_size_percent=len(self.X_train_filtered) / len(full_dataset_fold.X_train)
                )
                if print_results:
                    print(results)
                # Store results
                # self.modeling_attitude.append(results.modeling_atitude)
                self.scores.append(results.score)
                self.run_times.append(results.run_time)
                self.scores_filtered.append(results.filtered_score)
                self.new_size_percents.append(results.new_size_percent)

    def test_indices_filter_method_cv(self,
                                      dataset_partial: Index_Dataset,
                                      dataset_full: Index_Dataset,
                                      trials_number=5,
                                      cv=4,
                                      print_results=False,
                                      save_each_iter=False,
                                      random_seed=100):
        """
        Perform cross-validation using a partial dataset (indices-based) to obtain a core set,
        handle missing classes by adding samples for missing classes,
        and then train and evaluate on the filtered data.

        :param dataset_partial: The partial dataset with indices-based access: to choose core-set samples.
        :param dataset_full: The full dataset from which training and test subsets are taken after being selected.
        :param trials_number: Number of repeated cross-validation runs.
        :param cv: Number of folds in K-Fold cross-validation.
        :param print_results: Whether to print each fold's result.
        :param save_each_iter: Whether to save intermediate results for each iteration.
        :param random_seed: Random seed for reproducibility.
        """
        self.trials_number = trials_number
        self.save_each_iter = save_each_iter

        for repeat in range(trials_number):
            kf = KFold(n_splits=cv, shuffle=True, random_state=random_seed + repeat)

            for train_index, test_index in kf.split(dataset_full.df):
                # Reset collected metrics before each fold
                self.reset_attributes()

                # Prepare the partial dataset for this fold
                X_partial = dataset_partial.df.drop(dataset_partial.target_col, axis=1)
                y_partial = dataset_partial.df[dataset_partial.target_col]

                X_train_partial = X_partial.loc[train_index, :]
                y_train_partial = y_partial[train_index]
                X_test_partial = X_partial.loc[test_index, :]
                y_test_partial = y_partial[test_index]

                partial_dataset_fold = Index_Dataset(
                    df=pd.concat([X_train_partial, y_train_partial], axis=1),
                    X_train=X_train_partial,
                    y_train=y_train_partial,
                    X_test=X_test_partial,
                    y_test=y_test_partial,
                    target_col=dataset_partial.target_col,
                    metric=dataset_partial.metric
                )

                # Prepare the full dataset for this fold
                X_full_fold = dataset_full.df.drop(dataset_full.target_col, axis=1)
                y_full_fold = dataset_full.df[dataset_full.target_col]

                X_train_full_fold = X_full_fold.loc[train_index, :]
                y_train_full_fold = y_full_fold[train_index]
                X_test_full_fold = X_full_fold.loc[test_index, :]
                y_test_full_fold = y_full_fold[test_index]

                full_dataset_fold = Index_Dataset(
                    df=pd.concat([X_train_full_fold, y_train_full_fold], axis=1),
                    X_train=X_train_full_fold,
                    y_train=y_train_full_fold,
                    X_test=X_test_full_fold,
                    y_test=y_test_full_fold,
                    target_col=dataset_full.target_col,
                    metric=dataset_full.metric
                )

                # Obtain the indices to form the core set from the partial training data
                core_set_indices = self.sample_indices(partial_dataset_fold, save_each_iter)

                # Build the core set from the full training data
                core_set = full_dataset_fold.df.loc[core_set_indices]
                self.X_train_filtered = core_set.drop(full_dataset_fold.target_col, axis=1)
                self.y_train_filtered = full_dataset_fold.y_train[self.X_train_filtered.index]

                # Validate the presence of all classes in the filtered dataset
                unique_classes = np.unique(self.y_train_filtered)
                all_classes = np.unique(dataset_full.df[dataset_full.target_col])

                missing_classes = set(all_classes) - set(unique_classes)
                if missing_classes:
                    print(f"Warning: Missing classes detected in filtered dataset: {missing_classes}")
                    for missing_class in missing_classes:
                        # Sample ~3 rows from the full dataset for the missing class
                        missing_samples = dataset_full.df[
                            dataset_full.df[dataset_full.target_col] == missing_class
                            ].sample(
                            n=min(15, len(dataset_full.df[dataset_full.target_col] == missing_class)),
                            random_state=42 + repeat
                        )
                        # Add the sampled rows to the filtered dataset
                        self.X_train_filtered = pd.concat(
                            [self.X_train_filtered, missing_samples.drop(dataset_full.target_col, axis=1)]
                        )
                        self.y_train_filtered = pd.concat(
                            [self.y_train_filtered, missing_samples[dataset_full.target_col]]
                        )
                    print(f"Added samples for missing classes: {missing_classes}")

                # Train or fit model on the filtered training data
                if self.prediction_model is not None:
                    self.prediction_model.fit(self.X_train_filtered, self.y_train_filtered)
                else:
                    self.prediction_model = xgb.XGBClassifier()
                    self.prediction_model.fit(self.X_train_filtered, self.y_train_filtered)

                # Evaluate on the test portion of the full dataset for this fold
                preds = self.prediction_model.predict(full_dataset_fold.X_test)

                results = FilteringResults(
                    modeling_atitude="sample by parial dataset, train and test by the full dataset",
                    score=full_dataset_fold.metric(full_dataset_fold.y_test, preds),
                    run_time=repeat,  # Time spent obtaining the core set
                    filtered_score=full_dataset_fold.metric(full_dataset_fold.y_test, preds),
                    new_size_percent=len(self.X_train_filtered) / len(full_dataset_fold.X_train)
                )
                if print_results:
                    print(results)
                # Store results
                #self.modeling_attitude.append(results.modeling_atitude)
                self.scores.append(results.score)
                self.run_times.append(results.run_time)
                self.scores_filtered.append(results.filtered_score)
                self.new_size_percents.append(results.new_size_percent)



    def test_indices_method_ensemble_classify_better_model(self,
                                                           dataset_partial: Index_Dataset,
                                                           dataset_full: Index_Dataset,
                                                           trials_number=1,
                                                           print_results=False,
                                                           save_each_iter=False):
        self.trials_number = trials_number
        self.save_each_iter = save_each_iter

        results_list = []

        for trial in range(trials_number):
            # Sample core set indices from the partial dataset
            core_set_indices = self.sample_indices(dataset_partial, save_each_iter)

            # Get the core set from the full dataset using those indices
            core_set = dataset_full.df.loc[core_set_indices]

            # From the core set, extract X and y
            core_set_X = core_set.drop(columns=[dataset_full.target_col])
            core_set_y = core_set[dataset_full.target_col]

            # Model 1: Train on the entire partial training data
            X_train_partial = dataset_partial.X_train.copy()
            y_train_partial = dataset_partial.y_train.copy()

            # Model 2: Train on the core set from the full data
            X_train_core = core_set_X.copy()
            y_train_core = core_set_y.copy()

            # Identify the full set of features from both training datasets
            combined_features = set(X_train_partial.columns) | set(X_train_core.columns)

            # Add missing features to both training sets
            for col in combined_features:
                if col not in X_train_partial.columns:
                    X_train_partial[col] = 0
                if col not in X_train_core.columns:
                    X_train_core[col] = 0

            # Reorder columns to ensure they are in the same order
            sorted_features = sorted(combined_features)
            X_train_partial = X_train_partial.loc[:, sorted_features]
            X_train_core = X_train_core.loc[:, sorted_features]

            # Align the test set features to match the training data
            dataset_full.add_missing_features(sorted_features)
            X_test_full = dataset_full.X_test.loc[:, sorted_features]
            y_test_full = dataset_full.y_test

            # Initialize prediction models
            if self.prediction_model is None:
                self.prediction_model = xgb.XGBClassifier()

            # Clone the prediction model to train separately
            from sklearn.base import clone
            model_partial = clone(self.prediction_model)
            model_core = clone(self.prediction_model)

            # Measure run time
            start_time = time.time()

            # Train Model 1 on the partial dataset
            model_partial.fit(X_train_partial, y_train_partial)

            # Train Model 2 on the core set
            model_core.fit(X_train_core, y_train_core)

            # Now, on the core set, get predicted probabilities from both models
            proba_partial_core = model_partial.predict_proba(X_train_core)
            proba_core_core = model_core.predict_proba(X_train_core)

            # Build features for the additional model
            # Use predicted probabilities from both models as features
            classes = model_partial.classes_
            proba_partial_core_df = pd.DataFrame(
                proba_partial_core,
                columns=[f'partial_proba_{cls}' for cls in classes],
                index=X_train_core.index
            )
            proba_core_core_df = pd.DataFrame(
                proba_core_core,
                columns=[f'core_proba_{cls}' for cls in classes],
                index=X_train_core.index
            )

            # Concatenate the features
            features_additional_model = pd.concat([proba_partial_core_df, proba_core_core_df], axis=1)

            # Labels are the true labels
            y_additional_model = y_train_core

            # Train the additional model (meta-model)
            from sklearn.linear_model import LogisticRegression
            additional_model = LogisticRegression(max_iter=1000)

            additional_model.fit(features_additional_model, y_additional_model)

            run_time = time.time() - start_time

            # On the test set, get predicted probabilities from both models
            proba_partial_test = model_partial.predict_proba(X_test_full)
            proba_core_test = model_core.predict_proba(X_test_full)

            # Build features for the additional model on the test set
            proba_partial_test_df = pd.DataFrame(
                proba_partial_test,
                columns=[f'partial_proba_{cls}' for cls in classes],
                index=X_test_full.index
            )
            proba_core_test_df = pd.DataFrame(
                proba_core_test,
                columns=[f'core_proba_{cls}' for cls in classes],
                index=X_test_full.index
            )

            features_additional_test = pd.concat([proba_partial_test_df, proba_core_test_df], axis=1)

            # Use the additional model to predict the labels
            final_predictions = additional_model.predict(features_additional_test)

            # Evaluate the predictions
            score = dataset_full.metric(y_test_full, final_predictions)

            results = FilteringResults(
                modeling_atitude="sample by parial dataset, train a model for full and a model for partial, get test prediction by the model that is classified as better for this sample.",
                score=score,
                run_time=trial,  # Time spent obtaining the core set
                filtered_score=score,
                new_size_percent=len(X_train_core) / len(dataset_full.X_train))

            if print_results:
                print(results)

            # Collect results into self attributes
            self.modeling_attitude.append(self.modeling_attitude)
            self.scores.append(score)
            self.run_times.append(run_time)
            self.scores_filtered.append(score)
            self.new_size_percents.append(len(X_train_core) / len(dataset_full.X_train))

