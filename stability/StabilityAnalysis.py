#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:49:07 2019

@author: chrisbartel
"""

import os
from mab_ml.stability.CompositionAnalysis import CompositionAnalysis
from mab_ml.stability.HullAnalysis import AnalyzeHull
from mab_ml.data.data import id_hull_json, spaces, hull_out
from utils.utils import read_json, write_json
import multiprocessing as multip
from time import time
from sklearn.metrics import confusion_matrix, r2_score
import numpy as np
import random


def _update_hullin_space(ml, mp_hullin, space):
    """
    replace MP data for chemical space with ML data
    
    Args:
        ml (dict) - {formula (str) : {'Ef' : ML-predicted formation energy per atom (float)}}
        mp_hullin (dict) - mlstabilitytest.data.hullin
        space (str) - chemical space to update (format is '_'.join(sorted([element (str) for element in chemical space])))
        
    Returns:
        input data for analysis of one chemical space updated with ML-predicted data
    """

    ml_space = mp_hullin[space]

    for id in ml_space:
        if (CompositionAnalysis(id).num_els_in_formula == 1) or (id not in ml):
            continue
        else:
            ml_space[id]['E'] = ml[id]

    return ml_space


def _assess_stability(hullin, spaces, id):
    """
    determine the stability of a given id by hull analysis
    
    Args:
        hullin (dict) - {space (str) : {formula (str) : {'E' : formation energy per atom (float),
                                                         'amts' : 
                                                             {element (str) : 
                                                                 fractional amount of element in formula (float)}}}}
        spaces (dict) - {formula (str) : smallest convex hull space that includes formula (str)}
        id (str) - formula to determine stability for
        
    Returns:
        {'stability' : True if on the hull, else False,
         'Ef' : formation energy per atom (float),
         'Ed' : decomposition energy per atom (float),
         'rxn' : decomposition reaction (str)}
    """
    return AnalyzeHull(hullin, spaces[id]).cmpd_hull_output_data(id)


def _get_smact_hull_space(id, mp_spaces):
    """
    Args:
        id (str) - id to retrieve phase space for
        mp_spaces (list) - list of phase spaces in MP (str, '_'.join(elements))
    
    Returns:
        relevant chemical space (str) to determine stability id
    """
    els = set(CompositionAnalysis(id).els)
    for s in mp_spaces:
        space_els = set(s.split('_'))
        if (len(els.intersection(space_els)) == 4) and (len(space_els) < 7):
            return s


def _update_smact_space(ml, mp_hullin, space):
    """
    Args:
        ml (dict) - {id (str) - formation energy per atom (float)}
        mp_hullin (dict) - hull input file for all of MP
        space (str) - chemical space to update
        
    Returns:
        replaces MP formation energy with ML formation energies in chemical space
    """
    ml_space = mp_hullin[space]
    for id in ml_space:
        if (CompositionAnalysis(id).num_els_in_formula == 1) or (id not in ml):
            continue
        else:
            ml_space[id]['E'] = ml[id]

    for id in ml:
        if set(CompositionAnalysis(id).els).issubset(set(space.split('_'))):
            ml_space[id] = {'E': ml[id],
                            'amts': {el: CompositionAnalysis(id).amt_of_el(el)
                                     for el in space.split('_')}}

    return ml_space


def _get_stable_ids(hullin, space):
    """
    Args:
        hullin (dict) - hull input data
        space (str) - chemical space
        
    Returns:
        list of all stable ids (str) in chemical space
    """
    return AnalyzeHull(hullin, space).stable_ids


class EdAnalysis(object):
    """
    Assess performance of direct ML predictions on decomposition energy
    """

    def __init__(self,
                 data_dir,
                 data_file,
                 threshold=0.1):

        """
        converts input data to convenient format
        
        Args:
            data_dir (os.PathLike) - place where input ML data lives and to generate output data
            data_file (str) - .json file with input ML data of form {formula (str) : formation energy per atom (float)}
        """

        start = time()
        print('\nChecking input data...')

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        input_data = read_json(os.path.join(data_dir, data_file))  # *ml_input.json
        # {"45628709": {"target": 0.1130647, "predict": 0.1133114},

        # mp = hull_out()
        # ids = list(mp.keys())
        #
        # if set(ids).intersection(set(list(input_data.keys()))) != set(ids):
        #     print('ML dataset does not include all MP ids!')
        #     print('Cannot perform analysis.')
        #     raise AssertionError
        # {"45628709": {"target": 0.1130647, "predict": 0.1133114},

        self.ids = list(input_data.keys())
        self.input_data = input_data
        self.data_dir = data_dir
        self.threshold = threshold

        end = time()
        print('Data looks good.')
        print('Time elapsed = %.0f s.' % (end - start))

    @property
    def ml_hullout(self):
        """
        Args:
            
        Returns:
            converts input data into standard hull output data using ML-predicted Ed (dict)
            {id (str) : {'Ef' : None (not inputted),
                               'Ed' : decomposition energy per atom (float),
                               'stability' : True if Ed <= 0 else False,
                               'rxn' : None (not determined)}}
        """
        print('1')
        ml_in = self.input_data
        ids = self.ids

        return {c: {'Ef': None,
                    'Ed': ml_in[c]["predict"],
                    'stability': True if ml_in[c]["predict"] <= self.threshold else False,
                    'rxn': None
                    } for c in ids}

    def results(self, remake=False, label='', stability_thresholds=[], space_group=''):
        """
        generates output file with summary data for rapid analysis
        
        Args:
            remake (bool) - repeat generation of file if True; else read file
            
        Returns:
            dictionary with results in convenient format for analysis
        """
        fjson = os.path.join(self.data_dir, label + '_ml_results.json')
        if not remake and os.path.exists(fjson):
            print('\nReading existing results file: %s' % fjson)
            return read_json(fjson)

        ml_hullout = self.ml_hullout

        print('\nCompiling results...')
        start = time()

        ids = self.ids
        mp_hullout = hull_out(self.threshold, space_group)
        mp_hullout = {id: mp_hullout[id] for id in ids}
        print(mp_hullout)
        print(ml_hullout)

        obj = StabilitySummary(mp_hullout, ml_hullout, stability_thresholds=stability_thresholds)

        results = {'stats': {'Ed': obj.stats_Ed},
                   'data': {'Ed': obj.Ed['pred'],
                            'ids': obj.ids,
                            'sg': obj.sg,
                            'formula': obj.formula,
                            'M': obj.M,
                            'A': obj.A
                            }}

        end = time()
        print('Writing to %s.' % fjson)
        print('Time elapsed = %.0f s.' % (end - start))

        return write_json(results, fjson)

    def results_summary(self, label, stability_thresholds, space_group):
        """
        Prints summary of performance
        """
        self.smact_summary(label)

        # results = self.results(True)
        results = self.results(True, label, stability_thresholds, space_group)

        start = time()
        print('\nSummarizing performance...')

        Ed_MAE = np.nan
        tp, fp, tn, fn = [results['stats']['Ed']['cl'][str(self.threshold)]['raw'][num] for num in
                          ['tp', 'fp', 'tn', 'fn']]
        prec, recall, acc, f1 = [results['stats']['Ed']['cl'][str(self.threshold)]['scores'][score] for score in
                                 ['precision', 'recall', 'accuracy', 'f1']]
        fpr = fp / (fp + tn)

        print('\nMAE on decomposition enthalpy = %.3f eV/atom\n' % Ed_MAE)

        print('\nClassifying stable or unstable:')
        print('Precision = %.3f' % prec)
        print('Recall = %.3f' % recall)
        print('Accuracy = %.3f' % acc)
        print('F1 = %.3f' % f1)
        print('FPR = %.3f' % fpr)

        print('\nConfusion matrix:')
        print('TP | FP\nFN | TN = \n%i | %i\n%i | %i' % (tp, fp, fn, tn))

        end = time()
        print('\nTime elapsed = %i s' % (end - start))

    def smact_results(self, remake=False, label=''):
        """
        Args:
            remake (bool) - repeat generation of file if True; else read file
        
        Returns:
            Analyzes smact results
        """
        fjson = os.path.join(self.data_dir, label + '_ml_smact_results.json')
        if not remake and os.path.exists(fjson):
            print('\nReading existing smact results from: %s.' % fjson)
            return read_json(fjson)

        start = time()

        ids = self.ids
        ml = self.input_data

        pred_stable = [c for c in ids if ml[c]["predict"] <= self.threshold]

        mp_hullout = id_hull_json()
        mp_stable = [c for c in mp_hullout
                     if c in ids if mp_hullout[c]["hull_distance"] <= self.threshold]

        results = {
            'ids': ids,
            'MP_stable': mp_stable,
                   'pred_stable': pred_stable}
        end = time()
        print('Time elapsed = %i s' % (end - start))
        # import pprint
        # pprint.pprint(results)

        return write_json(results, fjson)

    def smact_summary(self, label):
        """
        Args:
            
        Returns:
            prints summary of SMACT results
        """
        results = self.smact_results(True, label=label)
        ids, mp_stable, pred_stable = [results[k] for k in ['ids', 'MP_stable', 'pred_stable']]

        print('%i ids investigated' % len(ids))
        print('%i are stable in MP' % len(mp_stable))
        print('%i (%.2f) are predicted to be stable' % (len(pred_stable), len(pred_stable) / len(ids)))
        print('%i of those are stable in MP' % (len([c for c in pred_stable if c in mp_stable])))


def _make_binary_labels(data, thresh):
    """
    Args:
        data (list) - list of floats
        thresh (float) - value to partition on 
    
    Returns:
        1 if value <= thresh else 0 for value in list
    """
    return [1 if v <= thresh else 0 for v in data]


class StabilityStats(object):
    """
    Perform statistical analysis on stability results
    """

    def __init__(self, actual, pred,
                 percentiles=[1, 10, 25, 50, 75, 90, 99],
                 stability_thresholds=[0, 0.1, 0.15, 0.2]):
        """
        Args:
            actual (list) - list of actual values (float) for some property
            pred (list) - list of predicted values (float) for some property
            percentiles (list) - list of percentiles (int) to obtain
            stability_thresholds (list) - list of thresholds (float) on which to classify as stable (below threshold) or unstable (above threshold)
        
        Returns:
            checks that actual and predicted lists are same length
        """
        if len(actual) != len(pred):
            raise ValueError
        self.actual = actual
        self.pred = pred
        self.percentiles = percentiles
        self.stability_thresholds = stability_thresholds

    @property
    def errors(self):
        """
        list of actual minus predicted
        """
        a, p = self.actual, self.pred
        return [a[i] - p[i] for i in range(len(a))]

    @property
    def abs_errors(self):
        """
        list of absolute value of actual minus predicted
        """
        errors = self.errors
        return [abs(e) for e in errors]

    @property
    def sq_errors(self):
        """
        list of (actual minus predicted) squared
        """
        errors = self.errors
        return [e ** 2 for e in errors]

    @property
    def mean_error(self):
        """
        mean error
        """
        return np.mean(self.errors)

    @property
    def mean_abs_error(self):
        """
        mean absolute error
        """
        return np.mean(self.abs_errors)

    @property
    def root_mean_sq_error(self):
        """
        root mean squared error
        """
        return np.sqrt(np.mean(self.sq_errors))

    @property
    def median_error(self):
        """
        median error
        """
        return np.median(self.errors)

    @property
    def median_abs_error(self):
        """
        median absolute error
        """
        return np.median(self.abs_errors)

    @property
    def r2(self):
        """
        correlation coefficient squared
        """
        return r2_score(self.actual, self.pred)

    @property
    def per_errors(self):
        """
        percentile errors (dict) {percentile : e such that percentile % of errors are < e}
        """
        percentiles = self.percentiles
        errors = self.errors
        return {int(percentiles[i]): np.percentile(errors, percentiles)[i] for i in range(len(percentiles))}

    @property
    def per_abs_errors(self):
        """
        percentile absolute errors (dict) {percentile : |e| such that percentile % of |errors| are < |e|}
        """
        percentiles = self.percentiles
        errors = self.abs_errors
        return {int(percentiles[i]): np.percentile(errors, percentiles)[i] for i in range(len(percentiles))}

    @property
    def regression_stats(self):
        """
        summary of stats
        """
        return {'abs': {'mean': self.mean_abs_error,
                        'median': self.median_abs_error,
                        'per': self.per_abs_errors},
                'raw': {'mean': self.mean_error,
                        'median': self.median_error,
                        'per': self.per_errors},
                'rmse': self.root_mean_sq_error,
                'r2': self.r2}

    def confusion(self, thresh):
        """
        Args:
            thresh (float) - threshold for stability (eV/atom)
        
        Returns:
            confusion matrix as dictionary
        """
        actual = _make_binary_labels(self.actual, thresh)
        pred = _make_binary_labels(self.pred, thresh)
        cm = confusion_matrix(actual, pred).ravel()
        # print(cm)
        labels = ['tn', 'fp', 'fn', 'tp']
        return dict(zip(labels, [int(v) for v in cm]))

    def classification_scores(self, thresh):
        """
        Args:
            thresh (float) - threshold for stability (eV/atom)
        
        Returns:
            classification stats as dict
        """
        confusion = self.confusion(thresh)
        print(confusion)
        tn, fp, fn, tp = [confusion[stat] for stat in ['tn', 'fp', 'fn', 'tp']]
        if tp + fp == 0:
            prec = 0
        else:
            prec = tp / (tp + fp)
        if tp + fn == 0:
            rec = 0
        else:
            rec = tp / (tp + fn)
        if prec + rec == 0:
            f1 = 0
        else:
            f1 = 2 * (prec * rec) / (prec + rec)
        acc = (tp + tn) / (tp + tn + fp + fn)
        if fp + tn == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)
        return {'precision': prec,
                'recall': rec,
                'f1': f1,
                'accuracy': acc,
                'fpr': fpr}

    @property
    def classification_stats(self):
        """
        summary of classification stats
        """
        threshs = self.stability_thresholds
        return {str(thresh): {'raw': self.confusion(thresh),
                              'scores': self.classification_scores(thresh)} for thresh in threshs}


class StabilitySummary(object):
    """
    Summarize stability performance stats
    """

    def __init__(self,
                 mp,
                 ml,
        stability_thresholds = [0, 0.1, 0.15, 0.2]):
        """
        Args:
            mp (dict) - dictionary of MP hull output data
            ml (dict) - dictionary of ML hull output data
        
        Returns:
            mp, ml
        """

        self.mp = mp
        self.ml = ml
        self.stability_thresholds = stability_thresholds
    @property
    def Ef(self):
        """
        put actual and predicted formation energies in dict
        """
        mp = self.mp
        ml = self.ml
        ids = sorted(list(ml.keys()))
        return {'actual': [mp[id]['Ef'] for id in ids],
                'pred': [ml[id]['Ef'] for id in ids]}

    @property
    def stats_Ef(self):
        """
        get stats on predicting formation energy
        """
        Ef = self.Ef
        actual, pred = Ef['actual'], Ef['pred']
        return StabilityStats(actual, pred).regression_stats

    @property
    def Ed(self):
        """
        put actual and predicted decomposition energies in dict
        """
        mp = self.mp
        ml = self.ml
        ids = sorted(list(ml.keys()))
        return {'actual': [mp[id]['Ed'] for id in ids],
                'pred': [ml[id]['Ed'] for id in ids]}

    @property
    def stats_Ed(self):
        """
        get stats on predicting decomposition energy
        """
        Ed = self.Ed
        actual, pred = Ed['actual'], Ed['pred']
        reg = StabilityStats(actual, pred, stability_thresholds=self.stability_thresholds).regression_stats
        cl = StabilityStats(actual, pred, stability_thresholds=self.stability_thresholds).classification_stats
        return {'reg': reg,
                'cl': cl}

    @property
    def rxns(self):
        """
        get decomposition reactions
        """
        mp = self.mp
        ml = self.ml
        ids = sorted(list(ml.keys()))
        return {'actual': [mp[id]['rxn'] for id in ids],
                'pred': [ml[id]['rxn'] for id in ids]}

    @property
    def ids(self):
        """
        get ids considered
        """
        ml = self.ml
        return sorted(list(ml.keys()))

    @property
    def sg(self):
        """
        get space_group
        """
        mp = self.mp
        ml = self.ml
        ids = sorted(list(ml.keys()))
        return [mp[id]["space_group"] for id in ids]

    @property
    def M(self):
        """
        get space_group
        """
        mp = self.mp
        ml = self.ml
        ids = sorted(list(ml.keys()))
        return [mp[id]["M"] for id in ids]

    @property
    def A(self):
        """
        get space_group
        """
        mp = self.mp
        ml = self.ml
        ids = sorted(list(ml.keys()))
        return [mp[id]["A"] for id in ids]

    @property
    def formula(self):
        """
        get space_group
        """
        mp = self.mp
        ml = self.ml
        ids = sorted(list(ml.keys()))
        return [mp[id]["formula"] for id in ids]
