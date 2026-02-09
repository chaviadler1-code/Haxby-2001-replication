# Final Project – Haxby 2001 Replication
**Authors:** Chavi Adler ID 324252113, Chen Cohen ID 207138389, Kislev Wainer ID 208662767

## Project Description

**Goal:**
The goal of this project is to reproduce key findings from Haxby et al. (2001), demonstrating that object categories are represented by distributed patterns of activity in the ventral temporal cortex rather than by isolated, category-specific regions.

**Main Objectives:**
- Load and analyze the Haxby fMRI dataset.
- Extract voxel-wise activation patterns for multiple object categories.
- Compare within-category and between-category pattern similarity using split-half correlations.
- Test whether category information remains after excluding maximally responsive voxels.

**Hypothesis:**
If object representations are distributed, then category-specific patterns should remain distinguishable even after removing voxels that respond most strongly to a given category.

**Assumptions:**
- We assume that the fMRI BOLD signal, after normalization (cocktail blank), accurately reflects the underlying neural population code for visual categories.
- The Ventral Temporal (VT) masks provided in the dataset correctly localize the relevant visual cortex areas.

---

## Folder Structure

This project is implemented as a modular Python package:

* `src/`: Contains the source code modules.
    * `data_loader.py`: Fetches the dataset.
    * `preprocessing.py`: Masking, splits (even/odd), and normalization (Cocktail Blank).
    * `analysis.py`: Calculates correlations and runs exclusion analysis.
    * `visualization.py`: Generates heatmaps and plots.
    * `pipeline.py`: Orchestrates the processing for a single subject.
* `tests/`: Contains unit tests to verify logic.
* `main.py`: The entry point script that runs the full pipeline.
* `pyproject.toml`: Project configuration and dependencies.

---

## Key Stages of the Analysis Pipeline

### 1. Data Import
- The Haxby fMRI dataset is loaded using `nilearn.datasets.fetch_haxby`.
- Six subjects are included in the analysis.
- Functional images, ventral temporal (VT) cortex masks, and category labels are loaded.

### 2. Preprocessing
- A ventral temporal cortex mask is applied to restrict the analysis to object-selective regions.
- Voxel time series are standardized and detrended using `NiftiMasker`.

### 3. Pattern Extraction
- Experimental runs are split into even and odd runs to create independent halves.
- For each object category, voxel-wise activation patterns are computed by averaging activity across all relevant time points, separately for even and odd runs.

### 4. Normalization (Cocktail Blank Removal)
- For each voxel, the mean activation across all categories is subtracted.
- This step emphasizes category-specific activation patterns.

### 5. Similarity Analysis
- Pearson correlation is computed between even-run and odd-run patterns.
- Within-category correlations are compared to between-category correlations.
- The resulting correlation matrix is visualized as a heatmap.

### 6. Exclusion Analysis
- Voxels are labeled according to the category that elicits their maximal response.
- For each category, voxels that respond maximally to that category are excluded.
- Pattern extraction and correlation analysis are repeated to test whether category information remains.

---

## Important Definitions and Parameters

- **Ventral Temporal (VT) Mask:** Restricts the analysis to ventral temporal cortex voxels.
- **Split-Half Analysis:** Even and odd runs are used to avoid circularity.
- **Cocktail Blank:** Subtraction of the mean activation across categories for each voxel.
- **Similarity Measure:** Pearson correlation coefficient.
- **Preferred Voxels:** Voxels showing maximal activation for a specific category.

---

## Data Description and Access

- **Dataset:** Haxby et al. (2001) fMRI dataset.
- **Source:** The data is fetched programmatically using `nilearn.datasets.fetch_haxby`.
- **Link to Dataset Documentation:** [Nilearn Haxby Dataset](https://nilearn.github.io/modules/generated/nilearn.datasets.fetch_haxby.html)
- **Content:** The dataset includes functional MRI scans (BOLD), anatomical images, category labels, and a ventral temporal cortex mask for 6 subjects.

---

## Results Summary

- Within-category correlations are higher than between-category correlations.
- The correlation matrix shows a clear diagonal structure, indicating category-specific activation patterns.
- After excluding maximally responsive voxels, category information remains above chance, supporting the hypothesis of distributed representations.

---

## Instructions for Running the Project

1. **Install dependencies:**
   Make sure you have the required packages installed:
   ```bash
   pip install numpy pandas matplotlib seaborn nilearn scikit-learn