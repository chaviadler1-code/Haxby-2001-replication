import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_correlation_matrix(patterns_even, patterns_odd):
    """Computes the Pearson correlation matrix between split-half patterns."""
    # We use the top-right quadrant of the full correlation matrix
    full_corr = np.corrcoef(patterns_even, patterns_odd)
    n_cats = patterns_even.shape[0]
    return full_corr[:n_cats, n_cats:]

def calculate_classification_accuracy(correlation_matrix):
    """Calculates accuracy based on diagonal elements (hits)."""
    n_cats = correlation_matrix.shape[0]
    predicted = np.argmax(correlation_matrix, axis=0)
    correct = sum(predicted[i] == i for i in range(n_cats))
    return (correct / n_cats) * 100

def perform_exclusion_analysis(patterns_even, patterns_odd, categories, original_matrix):
    """
    Iterates over categories, removes preferred voxels, and recalculates correlations.
    """
    voxel_preferences = np.argmax(patterns_even, axis=0)
    drops_list = []
    
    for cat_idx, cat_name in enumerate(categories):
        # 1. Identify Voxels to Keep (Non-maximal for this category)
        keep_voxels = (voxel_preferences != cat_idx)
        
        # 2. Extract Sub-patterns
        p_even_sub = patterns_even[:, keep_voxels]
        p_odd_sub = patterns_odd[:, keep_voxels]
        
        # 3. Re-calculate Correlation
        full_corr = np.corrcoef(p_even_sub, p_odd_sub)
        n_cats = len(categories)
        new_corr = full_corr[:n_cats, n_cats:][cat_idx, cat_idx]
        
        # 4. Calculate Drop
        orig_corr = original_matrix[cat_idx, cat_idx]
        drop_percent = ((orig_corr - new_corr) / orig_corr) * 100
        
        drops_list.append({
            'Category': cat_name,
            'Original Corr': orig_corr,
            'Excluded Corr': new_corr,
            'Drop': drop_percent
        })     
    return drops_list