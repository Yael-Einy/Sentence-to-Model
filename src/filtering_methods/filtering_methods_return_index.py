import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb
import random
import math
from collections import defaultdict
from dataclasses import dataclass
import ahocorasick
from statsmodels.stats.proportion import proportion_confint
from sklearn.tree import DecisionTreeClassifier
from typing import Sequence
from src.dataset_with_indices_for_full_and_partial_data import Index_Dataset
from src.dataset import Dataset

from src.filtering_methods.filtering_base_by_indices import FilteringExperiment, FilteringResults


class NoneFilter(FilteringExperiment):
    def __init__(self, name, model):
        super().__init__(name)
        self.model = model
        self.results = []
        self.modeling_attitude = []

    def sample_func(self, dataset: Dataset, p):
        s = time.time()
        self.model.fit(dataset.X_train, dataset.y_train)
        run_time = time.time() - s
        score = dataset.metric(dataset.y_test, self.model.predict(dataset.X_test))
        results = FilteringResults(score=score,
                                   run_time=run_time,
                                   filtered_score=score,
                                   new_size_percent=1)
        return dataset.index

    def sample_indices(self, dataset: Index_Dataset, p):
        return dataset.index


class RandomSampleFilter(FilteringExperiment):
    def __init__(self, name, p, model, prediction_model=xgb.XGBClassifier()):
        super().__init__(name)
        self.p = p
        self.model = model
        self.prediction_model = prediction_model
        self.modeling_attitude = []

    def sample_indices(self, dataset: Index_Dataset, p):
        size = int(len(dataset.X_train)*p)
        print(size, self.p)
        s = time.time()
        # Temporarily add target_col to X_train
        dataset.X_train[dataset.target_col] = dataset.y_train

        # Sample at least one instance from each class before performing fractional sampling
        X_min_samples = dataset.X_train.groupby(dataset.target_col).sample(n=3, random_state=50)

        # Perform fractional sampling (self.p) for the remaining samples
        X_real_train = dataset.X_train.groupby(dataset.target_col).sample(frac=self.p, random_state=50)

        # Combine the guaranteed samples with fractionally sampled data,
        # ensuring no duplicates
        X_real_train = pd.concat([X_min_samples, X_real_train]).drop_duplicates()

        # Retrieve the corresponding y values
        y_real_train = dataset.y_train[X_real_train.index]

        # Drop the temporary column
        dataset.X_train.drop(columns=dataset.target_col, inplace=True)
        X_real_train.drop(columns=dataset.target_col, inplace=True)
        # self.model.fit(X_real_train, y_real_train)
        # run_time = time.time() - s
        # score = dataset.metric(dataset.y_test, self.model.predict(dataset.X_test))
        # results = FilteringResults(score=score,
        #                            run_time=run_time,
        #                            filtered_score=score,
        #                            new_size_percent=len(X_real_train) / len(dataset.X_train))
        print('Random Sampler:', len(X_real_train.index[:size]), len(X_real_train.index))
        return X_real_train.index

    def sample_func(self, dataset: Dataset, p):
        s = time.time()
        dataset.X_train[dataset.target_col] = dataset.y_train
        X_real_train = dataset.X_train.groupby(dataset.target_col).sample(frac=self.p)
        y_real_train = dataset.y_train[X_real_train.index]

        dataset.X_train.drop(columns=dataset.target_col, inplace=True)
        X_real_train.drop(columns=dataset.target_col, inplace=True)

        self.model.fit(X_real_train, y_real_train)
        run_time = time.time() - s
        score = dataset.metric(dataset.y_test, self.model.predict(dataset.X_test))
        results = FilteringResults(score=score,
                                   run_time=run_time,
                                   filtered_score=score,
                                   new_size_percent=len(X_real_train) / len(dataset.X_train))
        return results



class RandomClassBalancedFilter(FilteringExperiment):
    def __init__(self, name, p, model, prediction_model=xgb.XGBClassifier()):
        super().__init__(name)
        self.p = p
        self.model = model
        self.prediction_model = prediction_model
        self.modeling_attitude = []


    def sample_indices(self, dataset_partial, save_each_iter=False):
        """
        Ensures balanced sampling across both classes.

        :param dataset_partial: The partial dataset containing X_train and y_train.
        :param save_each_iter: (optional) Save sampled indices for each iteration.
        :return:
            indices: The sampled indices ensuring balanced representation across classes.
        """
        size = int(len(dataset_partial.X_train)*self.p)
        y_labels = dataset_partial.y_train
        X_train_indices = np.array(dataset_partial.X_train.index)

        # Group indices by class
        class_0_indices = X_train_indices[y_labels == 0]
        class_1_indices = X_train_indices[y_labels == 1]

        total_core_set_size = int(self.p * len(X_train_indices))
        num_class_0 = int(total_core_set_size / 2)
        num_class_1 = total_core_set_size - num_class_0

        # Adjust if one class doesn't have enough samples
        if len(class_0_indices) < num_class_0:
            num_class_0 = len(class_0_indices)
            num_class_1 = total_core_set_size - num_class_0
        if len(class_1_indices) < num_class_1:
            num_class_1 = len(class_1_indices)
            num_class_0 = total_core_set_size - num_class_1

        # Sample indices from both classes
        sampled_class_0 = np.random.choice(class_0_indices, num_class_0, replace=False)
        sampled_class_1 = np.random.choice(class_1_indices, num_class_1, replace=False)

        # Combine sampled indices
        sampled_indices = np.concatenate([sampled_class_0, sampled_class_1])
        np.random.shuffle(sampled_indices)

        if save_each_iter:
            self.last_sampled_indices = sampled_indices

        print('Random Balanced Sampler:', len(sampled_indices))

        return sampled_indices [:size]


@dataclass
class FilteredGroup:
    key: str
    label: int
    size: int
    group: list


class FilterEachIterXgboostPathSampleFinal(FilteringExperiment):

    def __init__(self, name,
                 trees_number=100, sample_percent=0, examples_to_keep=1000, prediction_model=None, trees_to_stop=None,
                 params={'objective': 'binary:logistic'},
                 index_name='index', threshold=40, n_jobs=24):
        super().__init__(name)
        self.model = None
        self.prediction_model = prediction_model if prediction_model is not None else None
        self.trees_number = trees_number
        self.sample_percent = sample_percent
        self.params = params
        self.params.update({'n_jobs': n_jobs})
        self.trees_to_stop = trees_to_stop
        self.index_name = index_name
        self.X_leaves = None
        self.groups = None
        self.hom_groups = dict()
        self.hom_groups_candidates = []
        self.examples_to_keep = examples_to_keep
        self.threshold = threshold
        self.results = []
        self.A = None
        self.X_train_filtered = None
        self.y_train_filtered = None
        self.original_sizes = None
        self.modeling_attitude = []

    def reset_attributes(self) -> None:
        self.model = None
        self.X_leaves = None
        self.groups = None
        self.hom_groups = dict()
        self.hom_groups_candidates = []
        self.results = []
        self.A = None
        self.X_train_filtered = None
        self.y_train_filtered = None

    def get_dmatrix(self, X, y=None):
        if self.params.get('enable_categorical', False):
            if y is not None:
                dmatrix = xgb.DMatrix(X, label=y, enable_categorical=True)
            else:
                dmatrix = xgb.DMatrix(X, enable_categorical=True)
        else:
            if y is not None:
                dmatrix = xgb.DMatrix(X, label=y)
            else:
                dmatrix = xgb.DMatrix(X)
        return dmatrix

    def filter_groups(self, dataset: Dataset, i):
        new_groups = defaultdict(list)

        if self.groups is None:
            groups_first_leaf = self.X_leaves.reset_index().loc[:, ['index', 'leaf_0', dataset.target_col]].groupby(
                'leaf_0')
            new_groups = {str(leaf): list(zip(list(group['index'].values), list(group[dataset.target_col].values))) for
                          leaf, group in groups_first_leaf}
        else:
            index_to_leaf = self.X_leaves[f'leaf_{i}'].to_dict()
            for key, group in self.groups.items():
                if len(group) <= self.threshold:
                    continue
                for item in group:
                    new_groups[f'{key}_{index_to_leaf[item[0]]}'].append(item)
        indexes_to_drop = []
        groups_to_drop = []

        for key, group in new_groups.items():
            labels_sum = sum(item[1] for item in group)
            group_length = len(group)
            if group_length <= self.threshold:
                continue
            elif labels_sum == group_length or labels_sum == 0:
                self.hom_groups_candidates.append(FilteredGroup(key=key,
                                                                label=group[0][1],
                                                                size=group_length,
                                                                group=group))
                indexes_to_drop = indexes_to_drop + [item[0] for item in group]
                groups_to_drop.append(key)

        for key in groups_to_drop:
            new_groups.pop(key)

        self.groups = new_groups
        self.X_leaves = self.X_leaves[~self.X_leaves.index.isin(indexes_to_drop)]

        return indexes_to_drop

    def choose_groups(self, dataset: Dataset):
        sorted_candidates = sorted(self.hom_groups_candidates, key=lambda g: g.size, reverse=True)

        label_counter = defaultdict(int)
        label_amount = dict(dataset.y_train.value_counts())
        indexes_to_filter = []
        for group in sorted_candidates:
            label_counter[group.label] += group.size
            if label_amount[group.label] - label_counter[group.label] < self.examples_to_keep:
                continue
            else:
                self.hom_groups[group.key] = group.label
                candidate_indexes = [item[0] for item in group.group]
                new_indexes_to_filter = random.sample([item[0] for item in group.group],
                                                      k=math.floor((1 - self.sample_percent) * len(candidate_indexes)))
                indexes_to_filter = indexes_to_filter + new_indexes_to_filter

        return indexes_to_filter

    def get_guarantees(self, X_test, y_test, confidence=0.8):
        # pred_leaves = self.model.predict(xgb.DMatrix(data=(X_test)), pred_leaf=True)
        pred_leaves = self.model.predict(self.get_dmatrix(X_test), pred_leaf=True)
        groups_mistakes_dict = {key: 0 for key in self.hom_groups.keys()}
        groups_candidates_dict = {g.key: g for g in self.hom_groups_candidates}
        groups_dict = {key: groups_candidates_dict[key] for key in self.hom_groups.keys()}
        for pred_lst, label in zip(pred_leaves, y_test):
            joined = '_'.join([str(s) for s in pred_lst])
            key = self._is_in_homogeneous_group(joined)
            if key is not None and label != groups_dict[key].label:
                groups_mistakes_dict[key] += 1

        groups_lst = [(groups_dict[key], groups_mistakes_dict[key]) for key in groups_dict.keys()]
        groups_0 = [(g[1], g[0].size, g[0].label) for g in groups_lst if g[0].label == 0]
        groups_1 = [(g[1], g[0].size, g[0].label) for g in groups_lst if g[0].label == 1]
        train_size_1 = self.original_sizes[1]
        train_size_0 = self.original_sizes[0]
        if groups_0:
            mistakes_0_df = pd.DataFrame(data=groups_0,
                                         columns=['group_mistakes', 'group_size', 'label'])
            mistakes_0_df['total_mistakes'] = mistakes_0_df['group_mistakes'].cumsum()
            mistakes_0_df['total_size'] = mistakes_0_df['group_size'].cumsum()
            mistakes_0_df['percent_remained'] = (train_size_0 - mistakes_0_df['total_size']) / train_size_0
            mistakes_0_df['delta_recall'] = mistakes_0_df['total_mistakes'].apply(
                lambda x: proportion_confint(count=x, nobs=len(y_test[y_test == 1]),
                                             alpha=1 - confidence, method='wilson')[1])
        else:
            mistakes_0_df = None

        if groups_1:
            mistakes_1_df = pd.DataFrame(data=groups_1,
                                         columns=['group_mistakes', 'group_size', 'label'])
            mistakes_1_df['total_mistakes'] = mistakes_1_df['group_mistakes'].cumsum()
            mistakes_1_df['total_size'] = mistakes_1_df['group_size'].cumsum()
            mistakes_1_df['percent_remained'] = (train_size_1 - mistakes_1_df['total_size']) / train_size_1
            pn_ratio = train_size_1 / train_size_0
            mistakes_1_df['delta_precision'] = mistakes_1_df['total_mistakes'].apply(
                lambda x: 1 / (1 + pn_ratio / proportion_confint(count=x, nobs=len(y_test[y_test == 0]),
                                                                 alpha=1 - confidence, method='wilson'))[1])
        else:
            mistakes_1_df = None

        return mistakes_0_df, mistakes_1_df

    def _is_in_homogeneous_group(self, joined):
        occurrences = [i for i in range(len(joined)) if joined.startswith('_', i)]
        hom_comb = [joined[:i] for i in occurrences if self.A.exists(joined[:i])]
        if hom_comb:
            return hom_comb[0]
        return None

    def _predict_by_leafs(self, joined, pred_value):
        occurrences = [i for i in range(len(joined)) if joined.startswith('_', i)]
        target_values = [self.hom_groups[joined[:i]] for i in occurrences if self.A.exists(joined[:i])]
        if target_values:
            return target_values[0]
        else:
            return round(pred_value)

    def predict(self, X_test):
        pred_leaves = self.model.predict(self.get_dmatrix(X_test), pred_leaf=True)
        pred_leaves_df = pd.DataFrame(pred_leaves, index=X_test.index,
                                      columns=[f'leaf_{i}' for i in range(len(self.model.get_dump()))])

        if self.prediction_model is None:
            preds = self.model.predict(xgb.DMatrix(data=X_test))
        else:
            preds = self.prediction_model.predict(X_test)
        pred_leaves_df.loc[:, 'pred_value'] = preds

        pred_leaves_df.loc[:, 'joined'] = pred_leaves_df.apply(
            lambda row: '_'.join([str(row[c]) for c in pred_leaves_df.columns if c.startswith('leaf')]), axis=1)
        predictions = pred_leaves_df.apply(lambda row: self._predict_by_leafs(row['joined'], row['pred_value']), axis=1)
        return predictions

    def get_predictions_df(self, X_test):
        pred_leaves = self.model.predict(xgb.DMatrix(data=(X_test)), pred_leaf=True)
        pred_leaves_df = pd.DataFrame(pred_leaves, index=X_test.index,
                                      columns=[f'leaf_{i}' for i in range(len(self.model.get_dump()))])

        if self.prediction_model is None:
            preds = self.model.predict(xgb.DMatrix(data=X_test))
        else:
            preds = self.prediction_model.predict(X_test)
        pred_leaves_df.loc[:, 'pred_value'] = preds

        pred_leaves_df.loc[:, 'joined'] = pred_leaves_df.apply(
            lambda row: '_'.join([str(row[c]) for c in pred_leaves_df.columns if c.startswith('leaf')]), axis=1)
        pred_leaves_df.loc[:, 'pred_by_groups'] = pred_leaves_df.apply(
            lambda row: self._is_in_homogeneous_group(row['joined']),
            axis=1)
        pred_leaves_df.loc[:, 'final_prediction'] = pred_leaves_df.apply(
            lambda row: self._predict_by_leafs(row['joined'], row['pred_value']), axis=1)
        return pred_leaves_df

    def filter_df_and_dmatrix(self, dmatrix: xgb.DMatrix, data: pd.DataFrame, indexes, index_name='index'):
        new_data = data.reset_index()
        filtered_data = new_data[new_data[index_name].isin(indexes)]
        slice_indexes = list(filtered_data.index)
        filtered_dmatrix = dmatrix.slice(slice_indexes)
        return filtered_data.set_index(index_name), filtered_dmatrix

    def sample_func(self, dataset: Dataset, p):
        s = time.time()
        X_real_train = dataset.X_train.sort_index()
        y_real_train = dataset.y_train.sort_index()
        self.original_sizes = dict(y_real_train.value_counts())
        self.X_leaves = pd.DataFrame(y_real_train)
        self.hom_groups = dict()
        self.groups = None

        iter_dtrain = self.get_dmatrix(X_real_train, y_real_train)
        self.model = xgb.train(self.params, num_boost_round=self.trees_to_stop, dtrain=iter_dtrain)
        pred_leaves = self.model[: self.trees_to_stop].predict(iter_dtrain, pred_leaf=True)

        self.X_leaves = self.X_leaves.assign(**{f'leaf_{i}': pred_leaves[:, i] for i in range(self.trees_to_stop)})
        for i in range(self.trees_to_stop):
            self.filter_groups(dataset, i)

        indexes_to_filter = self.choose_groups(dataset)
        indexes_to_keep = sorted(set(X_real_train.index).difference(set(indexes_to_filter)))
        self.X_train_filtered, iter_dtrain = self.filter_df_and_dmatrix(iter_dtrain, X_real_train, indexes_to_keep)
        self.y_train_filtered = y_real_train[self.X_train_filtered.index]

        if self.prediction_model is None:
            if self.trees_number - self.trees_to_stop - 1 > 0:
                self.model = xgb.train(self.params, num_boost_round=self.trees_number - self.trees_to_stop,
                                       dtrain=iter_dtrain, xgb_model=self.model)
        else:
            self.prediction_model.fit(self.X_train_filtered, self.y_train_filtered)

        self.A = ahocorasick.Automaton()
        for key, target in self.hom_groups.items():
            self.A.add_word(key, (target, key))

        run_time = time.time() - s

        if self.prediction_model is None:
            preds = [round(p) for p in self.model.predict(xgb.DMatrix(data=dataset.X_test))]
        else:
            preds = self.prediction_model.predict(dataset.X_test)

        results = FilteringResults(score=dataset.metric(dataset.y_test, preds),
                                   run_time=run_time,
                                   filtered_score=dataset.metric(dataset.y_test, self.predict(dataset.X_test)),
                                   new_size_percent=len(self.X_train_filtered) / len(dataset.X_train))
        return results

    def sample_indices(self, dataset: Index_Dataset, p):
        s = time.time()
        X_real_train = dataset.X_train.sort_index()
        y_real_train = dataset.y_train.sort_index()
        self.original_sizes = dict(y_real_train.value_counts())
        self.X_leaves = pd.DataFrame(y_real_train)
        self.hom_groups = dict()
        self.groups = None

        iter_dtrain = self.get_dmatrix(X_real_train, y_real_train)
        self.model = xgb.train(self.params, num_boost_round=self.trees_to_stop, dtrain=iter_dtrain)
        pred_leaves = self.model[: self.trees_to_stop].predict(iter_dtrain, pred_leaf=True)

        self.X_leaves = self.X_leaves.assign(**{f'leaf_{i}': pred_leaves[:, i] for i in range(self.trees_to_stop)})
        for i in range(self.trees_to_stop):
            self.filter_groups(dataset, i) #TODO: make sure that the func works OK with index dataset

        indexes_to_filter = self.choose_groups(dataset) #TODO: make sure that the func works OK with index dataset
        indexes_to_keep = sorted(set(X_real_train.index).difference(set(indexes_to_filter)))
        self.X_train_filtered, iter_dtrain = self.filter_df_and_dmatrix(iter_dtrain, X_real_train, indexes_to_keep)
        self.y_train_filtered = y_real_train[self.X_train_filtered.index]

        # if self.prediction_model is None:
        #     if self.trees_number - self.trees_to_stop - 1 > 0:
        #         self.model = xgb.train(self.params, num_boost_round=self.trees_number - self.trees_to_stop,
        #                                dtrain=iter_dtrain, xgb_model=self.model)
        # else:
        #     self.prediction_model.fit(self.X_train_filtered, self.y_train_filtered)
        #
        # self.A = ahocorasick.Automaton()
        # for key, target in self.hom_groups.items():
        #     self.A.add_word(key, (target, key))
        #
        # run_time = time.time() - s
        #
        # if self.prediction_model is None:
        #     preds = [round(p) for p in self.model.predict(xgb.DMatrix(data=dataset.X_test))]
        # else:
        #     preds = self.prediction_model.predict(dataset.X_test)
        #
        # results = FilteringResults(score=dataset.metric(dataset.y_test, preds),
        #                            run_time=run_time,
        #                            filtered_score=dataset.metric(dataset.y_test, self.predict(dataset.X_test)),
        #                            new_size_percent=len(self.X_train_filtered) / len(dataset.X_train))
        return self.X_train_filtered.index[:self.examples_to_keep]


class FilterDTSampleFinal(FilteringExperiment):

    def __init__(self, name,
                 trees_number=100, sample_percent=0, examples_to_keep=1000, prediction_model=None, trees_to_stop=None,
                 params=None,
                 index_name='index', threshold=40, n_jobs=24):
        super().__init__(name)
        self.model = None
        self.prediction_model = prediction_model if prediction_model is not None else None
        self.trees_number = trees_number
        self.sample_percent = sample_percent
        self.params = params
        self.trees_to_stop = trees_to_stop
        self.index_name = index_name
        self.X_leaves = None
        self.groups = None
        self.hom_groups = dict()
        self.hom_groups_candidates = []
        self.examples_to_keep = examples_to_keep
        self.threshold = threshold
        self.results = []
        self.A = None
        self.X_train_filtered = None
        self.y_train_filtered = None
        self.original_sizes = None
        self.modeling_attitude = []

    def reset_attributes(self) -> None:
        self.model = None
        self.X_leaves = None
        self.groups = None
        self.hom_groups = dict()
        self.hom_groups_candidates = []
        self.results = []
        self.A = None
        self.X_train_filtered = None
        self.y_train_filtered = None

    def choose_groups(self, dataset: Dataset):
        sorted_candidates = sorted(self.hom_groups_candidates, key=lambda g: g.size, reverse=True)

        label_counter = defaultdict(int)
        label_amount = dict(dataset.y_train.value_counts())
        indexes_to_filter = []
        for group in sorted_candidates:
            label_counter[group.label] += group.size
            if label_amount[group.label] - label_counter[group.label] < self.examples_to_keep:
                continue
            else:
                self.hom_groups[group.key] = group.label
                candidate_indexes = [item for item in group.group]
                new_indexes_to_filter = random.sample([item for item in group.group],
                                                      k=math.floor((1 - self.sample_percent) * len(candidate_indexes)))
                indexes_to_filter = indexes_to_filter + new_indexes_to_filter

        return indexes_to_filter

    def get_guarantees(self, X_test, y_test, confidence=0.8):
        pred_leaves = self.model.apply(X_test)
        groups_mistakes_dict = {key: 0 for key in self.hom_groups.keys()}
        groups_candidates_dict = {g.key: g for g in self.hom_groups_candidates}
        groups_dict = {key: groups_candidates_dict[key] for key in self.hom_groups.keys()}
        for pred_leaf, label in zip(pred_leaves, y_test):
            key = self._is_in_homogeneous_group(pred_leaf)
            if key is not None and label != groups_dict[key].label:
                groups_mistakes_dict[key] += 1

        groups_lst = [(groups_dict[key], groups_mistakes_dict[key]) for key in groups_dict.keys()]
        groups_0 = [(g[1], g[0].size, g[0].label) for g in groups_lst if g[0].label == 0]
        groups_1 = [(g[1], g[0].size, g[0].label) for g in groups_lst if g[0].label == 1]
        train_size_1 = self.original_sizes[1]
        train_size_0 = self.original_sizes[0]
        if groups_0:
            mistakes_0_df = pd.DataFrame(data=groups_0,
                                         columns=['group_mistakes', 'group_size', 'label'])
            mistakes_0_df['total_mistakes'] = mistakes_0_df['group_mistakes'].cumsum()
            mistakes_0_df['total_size'] = mistakes_0_df['group_size'].cumsum()
            mistakes_0_df['percent_remained'] = (train_size_0 - mistakes_0_df['total_size']) / train_size_0
            mistakes_0_df['delta_recall'] = mistakes_0_df['total_mistakes'].apply(
                lambda x: proportion_confint(count=x, nobs=len(y_test[y_test == 1]),
                                             alpha=1 - confidence, method='wilson')[1])
        else:
            mistakes_0_df = None

        if groups_1:
            mistakes_1_df = pd.DataFrame(data=groups_1,
                                         columns=['group_mistakes', 'group_size', 'label'])
            mistakes_1_df['total_mistakes'] = mistakes_1_df['group_mistakes'].cumsum()
            mistakes_1_df['total_size'] = mistakes_1_df['group_size'].cumsum()
            mistakes_1_df['percent_remained'] = (train_size_1 - mistakes_1_df['total_size']) / train_size_1
            pn_ratio = train_size_1 / train_size_0
            mistakes_1_df['delta_precision'] = mistakes_1_df['total_mistakes'].apply(
                lambda x: 1 / (1 + pn_ratio / proportion_confint(count=x, nobs=len(y_test[y_test == 0]),
                                                                 alpha=1 - confidence, method='wilson'))[1])
        else:
            mistakes_1_df = None

        return mistakes_0_df, mistakes_1_df

    def _is_in_homogeneous_group(self, leaf_id):
        target_value = self.hom_groups.get(leaf_id, None)
        if target_value is None:
            return None
        return leaf_id

    def _predict_by_leafs(self, leaf_id, pred_value):
        target_value = self.hom_groups.get(leaf_id, None)
        if target_value is None:
            return round(pred_value)
        else:
            return target_value

    def predict(self, X_test):
        pred_leaves = self.model.apply(X_test)
        pred_leaves_df = pd.DataFrame(pred_leaves, index=X_test.index, columns=['leaf_id'])

        if self.prediction_model is None:
            preds = self.model.predict(X_test)
        else:
            preds = self.prediction_model.predict(X_test)
        pred_leaves_df.loc[:, 'pred_value'] = preds

        predictions = pred_leaves_df.apply(lambda row: self._predict_by_leafs(row['leaf_id'], row['pred_value']),
                                           axis=1)
        return predictions

    def get_predictions_df(self, X_test):
        pred_leaves = self.model.predict(xgb.DMatrix(data=(X_test)), pred_leaf=True)
        pred_leaves_df = pd.DataFrame(pred_leaves, index=X_test.index,
                                      columns=[f'leaf_{i}' for i in range(len(self.model.get_dump()))])

        if self.prediction_model is None:
            preds = self.model.predict(xgb.DMatrix(data=X_test))
        else:
            preds = self.prediction_model.predict(X_test)
        pred_leaves_df.loc[:, 'pred_value'] = preds

        pred_leaves_df.loc[:, 'joined'] = pred_leaves_df.apply(
            lambda row: '_'.join([str(row[c]) for c in pred_leaves_df.columns if c.startswith('leaf')]), axis=1)
        pred_leaves_df.loc[:, 'pred_by_groups'] = pred_leaves_df.apply(
            lambda row: self._is_in_homogeneous_group(row['joined']),
            axis=1)
        pred_leaves_df.loc[:, 'final_prediction'] = pred_leaves_df.apply(
            lambda row: self._predict_by_leafs(row['joined'], row['pred_value']), axis=1)
        return pred_leaves_df

    def filter_df_and_dmatrix(self, dmatrix: xgb.DMatrix, data: pd.DataFrame, indexes, index_name='index'):
        new_data = data.reset_index()
        filtered_data = new_data[new_data[index_name].isin(indexes)]
        slice_indexes = list(filtered_data.index)
        filtered_dmatrix = dmatrix.slice(slice_indexes)
        return filtered_data.set_index(index_name), filtered_dmatrix

    def sample_func(self, dataset: Dataset, p):
        s = time.time()
        X_real_train = dataset.X_train.sort_index()
        y_real_train = dataset.y_train.sort_index()
        self.original_sizes = dict(y_real_train.value_counts())
        self.X_leaves = pd.DataFrame(y_real_train)
        self.hom_groups = dict()
        self.groups = None
        if not self.params:
            self.model = DecisionTreeClassifier()
        else:
            self.model = DecisionTreeClassifier(**self.params)

        self.model.fit(X_real_train, y_real_train)
        pred_leaves = self.model.apply(X_real_train)
        self.X_leaves.loc[:, 'leaf_id'] = pred_leaves

        groups = self.X_leaves.groupby(['leaf_id']).agg(['sum', 'count'])[dataset.target_col].sort_values('count',
                                                                                                          ascending=False).reset_index()
        groups.loc[:, 'target_col'] = groups.apply(
            lambda row: int(row['sum'] / row['count']) if row['sum'] == 0 or row['sum'] == row['count'] else None,
            axis=1)
        groups.dropna(inplace=True)
        self.hom_groups_candidates = [FilteredGroup(key=row['leaf_id'], label=row['target_col'], size=row['count'],
                                                    group=self.X_leaves[
                                                        self.X_leaves['leaf_id'] == row['leaf_id']].index.tolist()) for
                                      index, row in groups.iterrows()]
        indexes_to_filter = self.choose_groups(dataset)
        indexes_to_keep = sorted(set(X_real_train.index).difference(set(indexes_to_filter)))
        self.X_train_filtered = X_real_train.loc[indexes_to_keep, :]
        self.y_train_filtered = y_real_train[self.X_train_filtered.index]

        if self.prediction_model is not None:
            self.prediction_model.fit(self.X_train_filtered, self.y_train_filtered)

        run_time = time.time() - s

        if self.prediction_model is None:
            preds = self.model.predict(dataset.X_test)
        else:
            preds = self.prediction_model.predict(dataset.X_test)

        results = FilteringResults(score=dataset.metric(dataset.y_test, preds),
                                   run_time=run_time,
                                   filtered_score=dataset.metric(dataset.y_test, self.predict(dataset.X_test)),
                                   new_size_percent=len(self.X_train_filtered) / len(dataset.X_train))
        return results

    def sample_indices(self, dataset: Dataset, p):
        s = time.time()
        X_real_train = dataset.X_train.sort_index()
        y_real_train = dataset.y_train.sort_index()
        self.original_sizes = dict(y_real_train.value_counts())
        self.X_leaves = pd.DataFrame(y_real_train)
        self.hom_groups = dict()
        self.groups = None
        if not self.params:
            self.model = DecisionTreeClassifier()
        else:
            self.model = DecisionTreeClassifier(**self.params)

        self.model.fit(X_real_train, y_real_train)
        pred_leaves = self.model.apply(X_real_train)
        self.X_leaves.loc[:, 'leaf_id'] = pred_leaves

        groups = self.X_leaves.groupby(['leaf_id']).agg(['sum', 'count'])[dataset.target_col].sort_values('count',
                                                                                                          ascending=False).reset_index()
        groups.loc[:, 'target_col'] = groups.apply(
            lambda row: int(row['sum'] / row['count']) if row['sum'] == 0 or row['sum'] == row['count'] else None,
            axis=1)
        groups.dropna(inplace=True)
        self.hom_groups_candidates = [FilteredGroup(key=row['leaf_id'], label=row['target_col'], size=row['count'],
                                                    group=self.X_leaves[
                                                        self.X_leaves['leaf_id'] == row['leaf_id']].index.tolist()) for
                                      index, row in groups.iterrows()]
        indexes_to_filter = self.choose_groups(dataset)
        indexes_to_keep = sorted(set(X_real_train.index).difference(set(indexes_to_filter)))
        self.X_train_filtered = X_real_train.loc[indexes_to_keep, :]
        self.y_train_filtered = y_real_train[self.X_train_filtered.index]

        # if self.prediction_model is not None:
        #     self.prediction_model.fit(self.X_train_filtered, self.y_train_filtered)
        #
        # run_time = time.time() - s
        #
        # if self.prediction_model is None:
        #     preds = self.model.predict(dataset.X_test)
        # else:
        #     preds = self.prediction_model.predict(dataset.X_test)
        #
        # results = FilteringResults(score=dataset.metric(dataset.y_test, preds),
        #                            run_time=run_time,
        #                            filtered_score=dataset.metric(dataset.y_test, self.predict(dataset.X_test)),
        #                            new_size_percent=len(self.X_train_filtered) / len(dataset.X_train))
        return self.X_train_filtered.index


class ActiveLearningUncertaintyFilter(FilteringExperiment):

    def __init__(self, name, number_of_examples, model, prediction_model=None,
                 temp_model=None, start_size=1_500, batch_size=1_000):
        super().__init__(name)
        self.number_of_examples = number_of_examples
        self.model = model
        self.start_size = start_size
        self.batch_size = batch_size
        self.temp_model = temp_model if temp_model else model
        self.prediction_model = prediction_model
        self.coresets = []
        self.tests = []
        self.modeling_attitude = []

    def sample_func(self, dataset: Dataset, p):
        s = time.time()

        X_real_train = dataset.X_train.sample(n=self.start_size)
        y_real_train = dataset.y_train[X_real_train.index]
        X_train_left = dataset.X_train[~dataset.X_train.index.isin(list(X_real_train.index))]

        while len(X_real_train) < self.number_of_examples:
            self.temp_model.fit(X_real_train, y_real_train)
            X_train_left = X_train_left.copy()
            X_train_left.loc[:, 'min_pred'] = np.min(self.temp_model.predict_proba(X_train_left[X_real_train.columns]),
                                                     axis=1)
            rows_addition = X_train_left.nlargest(self.batch_size, 'min_pred')

            X_real_train = pd.concat([X_real_train, rows_addition[X_real_train.columns]])
            y_real_train = dataset.y_train[X_real_train.index]
            X_train_left = X_train_left[~X_train_left.index.isin(list(rows_addition.index))]

        self.model.fit(X_real_train, y_real_train)

        run_time = time.time() - s
        self.coresets.append(X_real_train)
        self.tests.append(dataset.X_test)

        score = dataset.metric(dataset.y_test, self.model.predict(dataset.X_test))
        results = FilteringResults(score=score,
                                   run_time=run_time,
                                   filtered_score=score,
                                   new_size_percent=len(X_real_train) / len(dataset.X_train))
        return X_real_train.index

    def sample_indices(self, dataset: Dataset, p):
        size = int(len(dataset.X_train)*p)
        s = time.time()

        X_real_train = dataset.X_train.sample(n=self.start_size)
        y_real_train = dataset.y_train[X_real_train.index]
        X_train_left = dataset.X_train[~dataset.X_train.index.isin(list(X_real_train.index))]

        print("AL start size", len(X_real_train))
        while len(X_real_train) < self.number_of_examples:
            self.temp_model.fit(X_real_train, y_real_train)
            X_train_left = X_train_left.copy()
            X_train_left.loc[:, 'min_pred'] = np.min(self.temp_model.predict_proba(X_train_left[X_real_train.columns]),
                                                     axis=1)
            rows_addition = X_train_left.nlargest(self.batch_size, 'min_pred')

            X_real_train = pd.concat([X_real_train, rows_addition[X_real_train.columns]])
            y_real_train = dataset.y_train[X_real_train.index]
            X_train_left = X_train_left[~X_train_left.index.isin(list(rows_addition.index))]

        return X_real_train.index



class ActiveLearningStartFromCoreSet(FilteringExperiment):

    def __init__(self, name, number_of_examples, model, prediction_model=None,
                 temp_model=None, start_size=1000, batch_size=100, core_set_sampler=None):
        super().__init__(name)
        self.number_of_examples = number_of_examples  # Total number of examples to select
        self.model = model  # The model to use for training
        self.start_size = start_size  # Initial core set size
        self.batch_size = batch_size  # Number of samples to add in each iteration
        self.temp_model = temp_model if temp_model else clone(model)
        self.prediction_model = prediction_model if prediction_model else clone(model)
        self.coresets = []
        self.tests = []
        self.core_set_sampler = core_set_sampler  # Function to generate the core set
        self.modeling_attitude = []

    def sample_indices(self, dataset: Index_Dataset, p):
        s = time.time()

        # Step 1: Generate initial core set
        if self.core_set_sampler is not None:
            if callable(self.core_set_sampler):
                initial_indices = self.core_set_sampler(dataset, p = p)  # func
            else:
                initial_indices = list(self.core_set_sampler)  # list/array
            print('initial_indices:', initial_indices)
            print('ExtendTab:', 'len(initial_indices)', len(initial_indices), 'start_size', self.start_size)
            if len(initial_indices) > self.start_size:
                print("core set of the hard examples was chosen")
        else:
            # Default to stratified sampling
            sss = StratifiedShuffleSplit(n_splits=1, train_size=min(self.start_size, len(dataset.X_train)),
                                         random_state=55)
            for train_index, _ in sss.split(dataset.X_train, dataset.y_train):
                # Map positional indices to actual DataFrame indices
                initial_indices = dataset.X_train.index[train_index]

        # Validate that all sampled indices exist in X_train
        initial_indices = [idx for idx in initial_indices if idx in dataset.X_train.index]

        # Ensure compatibility of indices with X_train
        X_real_train = dataset.X_train.loc[initial_indices]
        y_real_train = dataset.y_train.loc[initial_indices]

        X_train_left = dataset.X_train.drop(index=initial_indices)

        # Step 2: Active‑learning loop
        while len(X_real_train) < self.number_of_examples and not X_train_left.empty:
            if len(X_real_train) >= self.number_of_examples:
                break
            print(f"AL phase – current size: {len(X_real_train)}")

            self.temp_model.fit(X_real_train.fillna(0), y_real_train)

            proba = self.temp_model.predict_proba(X_train_left.fillna(0))
            uncertainty = 1 - np.max(proba, axis=1)
            uncertainty_df = pd.DataFrame(
                {"uncertainty": uncertainty}, index=X_train_left.index
            )

            # pick at most what's left
            k = min(self.batch_size, len(uncertainty_df))
            if k == 0:
                break
            selected_indices = uncertainty_df.nlargest(k, "uncertainty").index

            # update sets
            X_new = X_train_left.loc[selected_indices]
            y_new = dataset.y_train.loc[selected_indices]
            X_real_train = pd.concat([X_real_train, X_new])
            y_real_train = pd.concat([y_real_train, y_new])
            X_train_left = X_train_left.drop(index=selected_indices)


        # Final model training on the full selected set
        print('validate that the training set size is the wished size',
              'training set size:', len(X_real_train),
              'wished core-set size:', self.number_of_examples)
#        self.model.fit(X_real_train.fillna(0), y_real_train)

#        run_time = time.time() - s
#        self.coresets.append(X_real_train)
#        self.tests.append(dataset.X_test)

 #       # Evaluate the model
 #       X_test_filled = dataset.X_test.fillna(0)
 #       score = dataset.metric(dataset.y_test, self.model.predict(X_test_filled))
 #       results = FilteringResults(
 #           score=score,
 #           run_time=run_time,
 #           filtered_score=score,
 #          new_size_percent=len(X_real_train) / len(dataset.X_train) * 100    )
        return X_real_train.index


class ActiveLearningStartFromCoreSet_based_on_reccommended_sampling_sizes(
        FilteringExperiment):
    """
    ExtendTab: active-learning sampler that starts from a user-supplied
    core set (or a random one) and grows it in balanced, uncertainty-driven
    batches until the validation score stops improving.

    Parameters
    ----------
    name : str
        Name of the experiment (used for logging / bookkeeping).
    number_of_examples : int
        Maximum training-set size we allow the algorithm to reach.
    model : sklearn-compatible estimator
        The model that will *ultimately* be trained (also cloned for
        uncertainty estimation).
    prediction_model, temp_model : estimators, optional
        Explicit models for scoring / uncertainty; if None, `clone(model)`
        is used.
    start_size : int
        Initial core-set size (random if `core_set_sampler` is None).
    batch_size : int
        How many samples to add at each AL iteration.
    core_set_sampler : Sequence[int] | pd.Index | None
        Pre-computed “hard” indices to seed the training set.
    improvement_threshold : float
        Minimum ∆score to count as “improvement”; otherwise counts toward
        early-stopping counter *m* in `sample_indices`.
    """

    # ------------------------- constructor -----------------------------

    def __init__(self,
                 name: str,
                 number_of_examples: int,
                 model,
                 prediction_model=None,
                 temp_model=None,
                 start_size: int = 1_000,
                 batch_size: int = 1_000,
                 core_set_sampler: Sequence[int] | pd.Index | None = None,
                 improvement_threshold: float = 0.02):
        super().__init__(name)

        # user parameters
        self.number_of_examples = number_of_examples
        self.model = model
        self.start_size = start_size
        self.batch_size = batch_size
        self.improvement_threshold = improvement_threshold

        # optional / derived parameters
        self.temp_model = temp_model if temp_model is not None else clone(model)
        self.prediction_model = (prediction_model
                                 if prediction_model is not None
                                 else clone(model))

        # core-set bookkeeping
        self.core_set_sampler = core_set_sampler

    # ----------------------- main public API ---------------------------

    def sample_indices(self,
                       dataset: Index_Dataset,
                       p: float,
                       m: int = 5) -> pd.Index:
        """
        Return a pandas Index of chosen training rows.

        Parameters
        ----------
        dataset : Index_Dataset
            Full dataset wrapper (gives access to X_train / y_train / etc.).
        p : float
            *Not* used inside (kept for API compatibility).
        m : int
            Early-stop after `m` consecutive non-improvements.

        Returns
        -------
        pd.Index
            Final selected indices (size may be < `number_of_examples`
            if early-stopped).
        """
        t0 = time.time()

        # ---------- 1. initial core set --------------------------------
        if self.core_set_sampler is not None:
            initial_indices = pd.Index(self.core_set_sampler)
            print(f"[ExtendTab] using supplied core set of "
                  f"{len(initial_indices)}/{self.start_size}")
        else:
            # stratified random seed
            sss = StratifiedShuffleSplit(n_splits=1,
                                         train_size=min(self.start_size,
                                                        len(dataset.X_train)),
                                         random_state=55)
            train_idx, _ = next(sss.split(dataset.X_train, dataset.y_train))
            initial_indices = dataset.X_train.index[train_idx]

        # keep only indices that actually exist
        initial_indices = initial_indices.intersection(dataset.X_train.index)
        self.initial_coreset_size = len(initial_indices)

        X_real_train = dataset.X_train.loc[initial_indices]
        y_real_train = dataset.y_train.loc[initial_indices]
        X_train_left = dataset.X_train.drop(index=initial_indices)

        prev_score = -float("inf")
        no_improve = 0

        # ---------- 2. active-learning loop ----------------------------
        while (len(X_real_train) < self.number_of_examples
               and len(X_train_left) > 0):

            print(f"[ExtendTab] AL-phase — train size {len(X_real_train)}")

            self.temp_model.fit(X_real_train.fillna(0), y_real_train)
            proba = self.temp_model.predict_proba(X_train_left.fillna(0))
            uncertainty = 1.0 - np.max(proba, axis=1)

            df_uncert = pd.DataFrame({
                "uncertainty": uncertainty,
                "y": dataset.y_train.loc[X_train_left.index].values
            }, index=X_train_left.index)

            # balanced selection: half from each class if possible
            half = self.batch_size // 2
            pick = self._balanced_top_k(df_uncert, half)

            X_new = X_train_left.loc[pick]
            y_new = dataset.y_train.loc[pick]

            X_real_train = pd.concat([X_real_train, X_new])
            y_real_train = pd.concat([y_real_train, y_new])
            X_train_left = X_train_left.drop(index=pick)

            # ---------- validation & early-stop ------------------------
            score = dataset.metric(dataset.y_test,
                                   self.temp_model.predict(dataset.X_test.fillna(0)))
            print(f"[ExtendTab] val-score {score:.4f}")

            if score - prev_score < self.improvement_threshold:
                no_improve += 1
                if no_improve >= m:
                    print(f"[ExtendTab] Stopped after {m} flat rounds.")
                    break
            else:
                no_improve = 0
            prev_score = score

        # ---------- 3. bookkeeping / return ---------------------------
        self.final_coreset_size = len(X_real_train)
        self.size = self.final_coreset_size
        self.coresets.append(X_real_train)
        self.tests.append(dataset.X_test)

        print(f"[ExtendTab] initial {self.initial_coreset_size} → final "
              f"{self.final_coreset_size} (time {time.time()-t0:.1f}s)")

        return X_real_train.index

    # --------------------- helper functions ---------------------------

    @staticmethod
    def _balanced_top_k(df_uncert: pd.DataFrame, half: int) -> np.ndarray:
        """Return up to `2*half` indices, half per class when available."""
        c0 = df_uncert[df_uncert["y"] == 0].nlargest(half, "uncertainty").index
        c1 = df_uncert[df_uncert["y"] == 1].nlargest(half, "uncertainty").index
        both = np.concatenate([c0, c1])
        if len(both) < 2 * half:      # not enough from one class → fill up
            rest = df_uncert.drop(index=both).nlargest(2 * half - len(both),
                                                       "uncertainty").index
            both = np.concatenate([both, rest])
        return both


class ActiveLearningStartFromCoreSet(FilteringExperiment):
    def __init__(self, name, number_of_examples, model, prediction_model=None,
                 temp_model=None, start_size=1000, batch_size=1000, core_set_sampler=None):
        super().__init__(name)
        self.number_of_examples = number_of_examples  # Total number of examples to select
        self.model = model  # The model to use for training
        self.start_size = start_size  # Initial core set size
        self.batch_size = batch_size  # Number of samples to add in each iteration
        self.temp_model = temp_model if temp_model else clone(model)
        self.prediction_model = prediction_model if prediction_model else clone(model)
        self.coresets = []
        self.tests = []
        self.core_set_sampler = core_set_sampler  # Function to generate the core set
        self.modeling_attitude = []

    def sample_indices(self, dataset: Index_Dataset, p):
        s = time.time()

        # Step 1: Generate initial core set (unchanged)
        if self.core_set_sampler is not None:
            initial_indices = self.core_set_sampler
            print('ExtendTab:', 'len(initial_indices)', len(initial_indices), 'start_size', self.start_size)
            if len(initial_indices) > self.start_size:
                print("core set of the hard examples was chosen")
        else:
            sss = StratifiedShuffleSplit(n_splits=1,
                                         train_size=min(self.start_size, len(dataset.X_train)),
                                         random_state=55)
            for train_index, _ in sss.split(dataset.X_train, dataset.y_train):
                initial_indices = dataset.X_train.index[train_index]

        print('start size', len(initial_indices))

        # Validate that all sampled indices exist in X_train (unchanged)
        initial_indices = [idx for idx in initial_indices if idx in dataset.X_train.index]

        # Ensure compatibility of indices with X_train (unchanged)
        X_real_train = dataset.X_train.loc[initial_indices]
        y_real_train = dataset.y_train.loc[initial_indices]

        X_train_left = dataset.X_train.drop(index=initial_indices)

        # Step 2: Active learning loop (MODIFIED)
        while len(X_real_train) < self.number_of_examples:
            print('ExtendTab: AL Phase', f'Current training set size: {len(X_real_train)}')

            # Train temp model on current training set (unchanged)
            self.temp_model.fit(X_real_train.fillna(0), y_real_train)

            # Predict probabilities on the remaining data (unchanged)
            proba = self.temp_model.predict_proba(X_train_left.fillna(0))

            # Compute uncertainty = 1 - max(prob)
            uncertainty = 1 - np.max(proba, axis=1)

            # -------------------------------------------------------------------
            # MODIFICATION: Instead of selecting the top `batch_size` by uncertainty
            # across *all* classes, we will take half from class 0 and half from class 1
            # (or as many as are left in that class) based on highest uncertainty.
            # -------------------------------------------------------------------
            # Build a small DataFrame with uncertainties + their known class label
            df_uncert = pd.DataFrame({
                'uncertainty': uncertainty,
                'y': dataset.y_train.loc[X_train_left.index].values
            }, index=X_train_left.index)

            # Split into class 0 and class 1
            class0_df = df_uncert[df_uncert['y'] == 0]
            class1_df = df_uncert[df_uncert['y'] == 1]

            # Determine how many samples per class in this AL batch
            c0_needed = self.batch_size // 2
            c1_needed = self.batch_size - c0_needed

            # Take the most uncertain from each class
            class0_top = class0_df.nlargest(min(c0_needed, len(class0_df)), 'uncertainty').index
            class1_top = class1_df.nlargest(min(c1_needed, len(class1_df)), 'uncertainty').index

            # Combine selected indices
            selected_indices = np.concatenate([class0_top, class1_top])
            # -------------------------------------------------------------------

            # Add selected samples to the training set (unchanged)
            X_new = X_train_left.loc[selected_indices]
            y_new = dataset.y_train.loc[selected_indices]
            X_real_train = pd.concat([X_real_train, X_new], axis=0)
            y_real_train = pd.concat([y_real_train, y_new], axis=0)

            # Remove selected samples from the remaining data (unchanged)
            X_train_left = X_train_left.drop(index=selected_indices)

        # Final model training on the full selected set (unchanged debug prints)
        print('validate that the training set size is the wished size',
              'training set size:', len(X_real_train),
              'wished core-set size:', self.number_of_examples)


        return X_real_train.index


class ActiveLearningStartFromCoreSet_with_class_balance(FilteringExperiment):
    def __init__(self, name, number_of_examples, model, prediction_model=None,
                 temp_model=None, start_size=1000, batch_size=1000, core_set_sampler=None):
        super().__init__(name)
        self.number_of_examples = number_of_examples  # Total number of examples to select
        self.model = model  # The model to use for training
        self.start_size = start_size  # Initial core set size
        self.batch_size = batch_size  # Number of samples to add in each iteration
        self.temp_model = temp_model if temp_model else clone(model)
        self.prediction_model = prediction_model if prediction_model else clone(model)
        self.coresets = []
        self.tests = []
        self.core_set_sampler = core_set_sampler  # Function to generate the core set
        self.modeling_attitude = []

    def sample_indices(self, dataset: Index_Dataset, p):
        import time
        s = time.time()

        # Step 1: Generate initial core set
        if self.core_set_sampler is not None:
            initial_indices = self.core_set_sampler
            print('ExtendTab:', 'len(initial_indices)', len(initial_indices), 'start_size', self.start_size)
            if len(initial_indices) > self.start_size:
                print("core set of the hard examples was chosen")
        else:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1,
                                         train_size=min(self.start_size, len(dataset.X_train)),
                                         random_state=42)
            for train_index, _ in sss.split(dataset.X_train, dataset.y_train):
                initial_indices = dataset.X_train.index[train_index]

        # Validate that all sampled indices exist in X_train
        initial_indices = [idx for idx in initial_indices if idx in dataset.X_train.index]

        # Initialize training and pool sets
        X_real_train = dataset.X_train.loc[initial_indices]
        y_real_train = dataset.y_train.loc[initial_indices]
        X_train_left = dataset.X_train.drop(index=initial_indices)

        # Step 2: Active learning loop
        while len(X_real_train) < self.number_of_examples:
            print('ExtendTab: AL Phase', f'Current training set size: {len(X_real_train)}')

            # Train temporary model
            self.temp_model.fit(X_real_train.fillna(0), y_real_train)

            # Predict probabilities and compute uncertainty
            proba = self.temp_model.predict_proba(X_train_left.fillna(0))
            uncertainty = 1 - np.max(proba, axis=1)

            # Create DataFrame with uncertainty and true class
            df_uncert = pd.DataFrame({
                'uncertainty': uncertainty,
                'y': dataset.y_train.loc[X_train_left.index].values
            }, index=X_train_left.index)

            # Split by class
            class0_df = df_uncert[df_uncert['y'] == 0]
            class1_df = df_uncert[df_uncert['y'] == 1]

            # Desired number of samples from each class
            c0_needed = self.batch_size // 2
            c1_needed = self.batch_size - c0_needed

            # Case 1: Enough samples for balanced sampling
            if len(class0_df) >= c0_needed and len(class1_df) >= c1_needed:
                class0_top = class0_df.nlargest(c0_needed, 'uncertainty').index
                class1_top = class1_df.nlargest(c1_needed, 'uncertainty').index
                selected_indices = np.concatenate([class0_top, class1_top])
            else:
                # Case 2: Not enough in one class → fallback to top-K uncertainty
                selected_indices = df_uncert.nlargest(self.batch_size, 'uncertainty').index

            # Add selected samples to training set
            X_new = X_train_left.loc[selected_indices]
            y_new = dataset.y_train.loc[selected_indices]
            X_real_train = pd.concat([X_real_train, X_new], axis=0)
            y_real_train = pd.concat([y_real_train, y_new], axis=0)

            # Remove selected samples from pool
            X_train_left = X_train_left.drop(index=selected_indices)

        print('validate that the training set size is the wished size',
              'training set size:', len(X_real_train),
              'wished core-set size:', self.number_of_examples)

        return X_real_train.index


from modAL.models import ActiveLearner

class ActiveLearningBaselines(FilteringExperiment):
    def __init__(self, name, number_of_examples, model, prediction_model=None,
                 temp_model=None, start_size=100, batch_size=1000, core_set_sampler=None, query_strategy=None):
        super().__init__(name)
        self.number_of_examples = number_of_examples  # Total number of examples to select
        self.model = model  # The model to use for training
        self.start_size = start_size  # Initial core set size
        self.batch_size = batch_size  # Number of samples to add in each iteration
        self.temp_model = temp_model if temp_model else clone(model)
        self.prediction_model = prediction_model if prediction_model else clone(model)
        self.coresets = []
        self.tests = []
        self.core_set_sampler = core_set_sampler  # Function to generate the core set
        self.modeling_attitude = []
        self.query_strategy = query_strategy  # Query strategy for active learning

    def sample_indices(self, dataset: Dataset, p):
        #size = int(len(dataset.X_train)*p)
        s = time.time()

        X_current_train = dataset.X_train.sample(n=self.start_size, random_state=42)
        y_current_train = dataset.y_train[X_current_train.index]
        X_pool = dataset.X_train[~dataset.X_train.index.isin(list(X_current_train.index))].copy()
        y_pool = dataset.y_train[X_pool.index].copy()

        self.temp_model.fit(X_current_train, y_current_train)
        while len(X_current_train) < self.number_of_examples:

            learner = ActiveLearner(
                estimator=self.temp_model,
                query_strategy=self.query_strategy,
                X_training=X_current_train.values, 
                y_training=y_current_train.values
            )
 
            query_index, query_instance = learner.query(X_pool.values)

            # Flatten query_index if it's a multidimensional array and convert to proper format
            if isinstance(query_index, np.ndarray):
                query_index = query_index.flatten()
            
            # Get the actual pandas indices of the queried samples
            queried_indices = X_pool.iloc[query_index].index
            
            # Add queried samples to training set
            X_current_train = pd.concat([X_current_train, X_pool.loc[queried_indices]])
            y_current_train = dataset.y_train[X_current_train.index]
            
            # Remove queried samples from pool
            X_pool = X_pool.drop(queried_indices)
            y_pool = y_pool.drop(queried_indices)

        return X_current_train.index



    def sample_indices_(self, dataset: Dataset, p):
        size = int(len(dataset.X_train)*p)

        X_current_train = dataset.X_train.sample(n=self.start_size)
        y_current_train = dataset.y_train[X_current_train.index]
        X_train_left = dataset.X_train[~dataset.X_train.index.isin(list(X_current_train.index))]

        while len(X_current_train) < self.number_of_examples:
            self.temp_model.fit(X_current_train, y_current_train)
            learner = ActiveLearner(estimator=self.temp_model, X_training=X_current_train, y_training=y_current_train)
            query_index, query_instance = learner.query(X_train_left)
            rows_addition = X_train_left[query_index]
            X_current_train = pd.concat([X_current_train, rows_addition[X_current_train.columns]])
            y_current_train = dataset.y_train[X_current_train.index]
            X_train_left = X_train_left[~X_train_left.index.isin(list(rows_addition.index))]

        return X_current_train.index

class ActiveLearningQueryByCommittee(ActiveLearningUncertaintyFilter):

    def sample_indices_knn(self, dataset: Dataset, p):
        # size = int(len(dataset.X_train)*p)
        s = time.time()

        X_current_train = dataset.X_train.sample(n=self.start_size)
        y_current_train = dataset.y_train[X_current_train.index]
        X_pool = dataset.X_train[~dataset.X_train.index.isin(list(X_current_train.index))]
        y_pool = dataset.y_train[X_pool.index]

        while len(X_current_train) < self.number_of_examples:
            self.temp_model.fit(X_current_train, y_current_train)
            learner = ActiveLearner(estimator=self.temp_model, X_training=X_current_train, y_training=y_current_train)
            query_index, query_instance = learner.query(X_pool)

            # Teach our ActiveLearner model the record it has requested.
            X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            learner.teach(X=X, y=y)

            # Remove the queried instance from the unlabeled pool.
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
            rows_addition = X_pool[query_index]

            X_current_train = pd.concat([X_current_train, rows_addition[X_current_train.columns]])
            y_current_train = dataset.y_train[X_current_train.index]

        return X_current_train.index

    def sample_indices(self, dataset: Dataset, p):
        size = int(len(dataset.X_train) * p)
        s = time.time()

        X_current_train = dataset.X_train.sample(n=self.start_size)
        y_current_train = dataset.y_train[X_current_train.index]
        X_train_left = dataset.X_train[~dataset.X_train.index.isin(list(X_current_train.index))]

        while len(X_current_train) < self.number_of_examples:
            self.temp_model.fit(X_current_train, y_current_train)
            X_train_left = X_train_left.copy()
            X_train_left.loc[:, 'min_pred'] = np.min(
                self.temp_model.predict_proba(X_train_left[X_current_train.columns]),
                axis=1)
            rows_addition = X_train_left.nlargest(self.batch_size, 'min_pred')

            X_current_train = pd.concat([X_current_train, rows_addition[X_current_train.columns]])
            y_current_train = dataset.y_train[X_current_train.index]
            X_train_left = X_train_left[~X_train_left.index.isin(list(rows_addition.index))]

        return X_current_train.index


