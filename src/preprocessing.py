import logging
import pandas as pd
import numpy as np
from nilearn.input_data import NiftiMasker

logger = logging.getLogger(__name__)

def load_behavioral_data(labels_file):
    """Loads and filters behavioral data (removes rest)."""
    behavioral = pd.read_csv(labels_file, sep=" ")
    conditions = behavioral['labels']
    runs = behavioral['chunks']
    
    task_mask = (conditions != 'rest')
    return conditions[task_mask], runs[task_mask], task_mask

def apply_masking(func_file, mask_file, task_mask):
    """Applies NiftiMasker to the functional data."""
    masker = NiftiMasker(mask_img=mask_file, standardize=True, detrend=True, 
                         memory="nilearn_cache", memory_level=1)
    fmri_masked = masker.fit_transform(func_file)
    return fmri_masked[task_mask]

def get_categories(conditions):
    """Returns sorted unique categories."""
    categories = np.sort(conditions.unique())
    return categories[categories != 'rest']

def calculate_category_patterns(fmri_data, conditions, run_mask, categories):
    """
    Calculates the mean activity pattern for each category 
    and subtracts the mean across all categories (Cocktail Blank).
    """
    data = fmri_data[run_mask]
    labels = conditions[run_mask]
    
    # Mean per category
    patterns = np.array([data[labels == c].mean(axis=0) for c in categories])
    
    # Normalize (Cocktail Blank)
    return patterns - patterns.mean(axis=0)

def split_runs_and_extract_patterns(fmri_masked, conditions, runs, categories):
    """Splits data into even/odd runs and extracts normalized patterns."""
    even_runs = (runs % 2 == 0)
    odd_runs = (runs % 2 != 0)
    
    patterns_even = calculate_category_patterns(fmri_masked, conditions, even_runs, categories)
    patterns_odd = calculate_category_patterns(fmri_masked, conditions, odd_runs, categories)
    
    return patterns_even, patterns_odd