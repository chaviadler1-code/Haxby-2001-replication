import logging
import numpy as np
from src.data_loader import fetch_haxby_data
from src.pipeline import process_single_subject
from src.visualization import (plot_correlation_heatmap, plot_exclusion_bars, 
                               print_final_tables)

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Haxby Reproduction Project...")

    # 1. Data Import
    dataset, subjects = fetch_haxby_data()
    
    # Init storage
    all_matrices = []
    accuracies = []
    all_drops_data = []
    categories = None 

    # 2. Main Loop
    for i, sub_id in enumerate(subjects):
        logger.info(f"Processing Subject {sub_id}...")
        
        # Call the pipeline for this subject
        corr, acc, drops, cats = process_single_subject(i, dataset)
        
        # Collect results
        all_matrices.append(corr)
        accuracies.append(acc)
        all_drops_data.extend(drops)
        categories = cats 

    # 3. Visualization & Output
    logger.info("Generating Final Results...")
    
    grand_avg_matrix = np.mean(all_matrices, axis=0)
    plot_correlation_heatmap(grand_avg_matrix, categories)
    plot_exclusion_bars(all_drops_data)
    print_final_tables(accuracies, all_drops_data, subjects)
    
    logger.info("Project completed successfully.")

if __name__ == "__main__":
    main()