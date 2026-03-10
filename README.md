# SGMAnalysis

A package for analyzing and visualizing SGM beamline data.

## Installation

```bash
pip install .
```

## Usage

```python
from sgmanalysis import MapScan, StackScan

# For a map scan
scan = MapScan('path/to/my_map_scan.h5')
scan.plot_overview(channel_roi=(80, 101))

# For a stack scan
stack = StackScan('path/to/my_stack_scan.h5')
stack.plot_summary(channel_roi=(80, 101), map_roi=[-1, 1, -1, 1])
```

### PCA & K-Means Clustering (Stack Scans)

Identify different chemical phases in your sample by clustering spectra across multiple energies. This implementation uses `scikit-learn` and includes robust data scaling and variance filtering.

```python
# 1. Perform the analysis
# You can use a single detector or a list of multiple detectors
results = stack.analyze_pca_kmeans(
    detector_names=['sdd1', 'sdd2', 'sdd3', 'sdd4'], 
    channel_roi=(80, 101), 
    n_clusters=4,
    n_components=5,    # Number of principal components to include
    normalize=True     # Apply StandardScaler for better sensitivity
)

# 2. Visualize and export results
# Use roll_shift to correct "jagged" edges in snake scans
# Use outfile to dump results to CSV files
stack.plot_pca_kmeans(
    results, 
    roll_shift=-2, 
    outfile="my_analysis_results"
)

# 3. Use the same clusters but plot the response for a specific detector
stack.plot_pca_kmeans(results, detector_names='sdd1', roll_shift=-2)
```

#### CSV Export Details
When using `outfile="my_analysis"`, two files are created:
- `my_analysis.csv`: Spatial data (x, y, cluster_id, and PCA component values for every pixel).
- `my_analysis_spectra.csv`: Spectral data (excitation energy and mean intensity for each identified cluster).
