import logging
from src.data_loader import get_subject_paths
from src.preprocessing import (load_behavioral_data, apply_masking, 
                               get_categories, split_runs_and_extract_patterns)
from src.analysis import (compute_correlation_matrix, calculate_classification_accuracy, 
                          perform_exclusion_analysis)

logger = logging.getLogger(__name__)

def process_single_subject(subject_index, dataset):
    """
    Runs the full analysis pipeline for a single subject.
    Returns: correlation matrix, accuracy, exclusion drops, and categories.
    """
    # 1. Load & Prepare Data
    func, mask, labels = get_subject_paths(subject_index, dataset)
    conditions, runs, task_mask = load_behavioral_data(labels)
    categories = get_categories(conditions)
    
    # 2. Masking
    fmri_masked = apply_masking(func, mask, task_mask)
    
    # 3. Pattern Extraction
    p_even, p_odd = split_runs_and_extract_patterns(fmri_masked, conditions, runs, categories)
    
    # 4. Analysis
    corr_matrix = compute_correlation_matrix(p_even, p_odd)
    acc = calculate_classification_accuracy(corr_matrix)
    drops = perform_exclusion_analysis(p_even, p_odd, categories, corr_matrix)
    
    return corr_matrix, acc, drops, categories