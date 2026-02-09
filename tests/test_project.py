import numpy as np
import pandas as pd
import pytest
from src.preprocessing import calculate_category_patterns
from src.analysis import (compute_correlation_matrix, 
                          calculate_classification_accuracy, 
                          perform_exclusion_analysis)

# --- Test Preprocessing ---
def test_cocktail_blank_normalization():
    """
    Test that the 'Cocktail Blank' normalization works correctly.
    If we have a pattern, subtracting the mean should result in a zero-mean vector.
    """
    # Create dummy fMRI data: 2 voxels, 4 runs
    # Run 1: Cat A, Run 2: Cat B, Run 3: Cat A, Run 4: Cat B
    fmri_data = np.array([
        [10, 20],  # Run 1
        [30, 40],  # Run 2
        [10, 20],  # Run 3
        [30, 40]   # Run 4
    ])
    
    conditions = pd.Series(['A', 'B', 'A', 'B'])
    run_mask = pd.Series([True, True, True, True]) # Use all runs
    categories = np.array(['A', 'B'])
    
    # Calculate patterns
    patterns = calculate_category_patterns(fmri_data, conditions, run_mask, categories)
    
    # Check shape: Should be (2 categories, 2 voxels)
    assert patterns.shape == (2, 2)
    
    # The mean across categories for each voxel should be close to 0
    # (Because we subtract the mean)
    mean_across_cats = patterns.mean(axis=0)
    assert np.allclose(mean_across_cats, 0)

# --- Test Analysis ---
def test_correlation_matrix_logic():
    """
    Test that the correlation matrix is calculated correctly.
    """
    # Create two identical patterns (Perfect correlation of 1.0)
    patterns_even = np.array([[1, 2, 3], [4, 5, 6]])
    patterns_odd = np.array([[1, 2, 3], [4, 5, 6]])
    
    corr_matrix = compute_correlation_matrix(patterns_even, patterns_odd)
    
    # The diagonal (A vs A, B vs B) should be exactly 1.0
    assert np.allclose(np.diag(corr_matrix), 1.0)

def test_accuracy_calculation():
    """
    Test that accuracy is 100% when the diagonal has the highest values.
    """
    # 2 categories. Diagonal elements (0,0) and (1,1) are highest.
    mock_corr_matrix = np.array([
        [0.9, 0.1],
        [0.2, 0.8]
    ])
    
    accuracy = calculate_classification_accuracy(mock_corr_matrix)
    assert accuracy == 100.0

def test_accuracy_calculation_failure():
    """
    Test accuracy when predictions are wrong (0%).
    """
    # Diagonal elements are LOWER than off-diagonal
    mock_corr_matrix = np.array([
        [0.1, 0.9],
        [0.8, 0.2]
    ])
    
    accuracy = calculate_classification_accuracy(mock_corr_matrix)
    assert accuracy == 0.0

def test_exclusion_analysis_structure():
    """
    Test that exclusion analysis returns the correct data structure.
    """
    # Create dummy data
    cats = np.array(['face'])
    # 1 category, 2 voxels
    p_even = np.array([[10, 1]]) 
    p_odd = np.array([[10, 1]])
    
    # Dummy correlation matrix (perfect correlation)
    orig_matrix = np.array([[1.0]])
    
    results = perform_exclusion_analysis(p_even, p_odd, cats, orig_matrix)
    
    # Assertions
    assert isinstance(results, list)
    assert len(results) == 1
    assert 'Drop' in results[0]
    assert 'Original Corr' in results[0]