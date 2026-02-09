import logging
from nilearn import datasets

logger = logging.getLogger(__name__)

def fetch_haxby_data(subjects=None):
    """
    Fetches the Haxby dataset for the specified subjects.
    """
    if subjects is None:
        subjects = [1, 2, 3, 4, 5, 6]
        
    logger.info(f"Fetching Haxby dataset for subjects: {subjects}")
    haxby_dataset = datasets.fetch_haxby(subjects=subjects)
    
    return haxby_dataset, subjects

def get_subject_paths(subject_index, dataset):
    """
    Returns file paths for a specific subject.
    """
    func_file = dataset.func[subject_index]
    mask_file = dataset.mask_vt[subject_index]
    labels_file = dataset.session_target[subject_index]
    return func_file, mask_file, labels_file