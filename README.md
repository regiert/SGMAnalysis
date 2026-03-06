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
scan.plot_roi_map(channel_roi=(80, 101))

# For a stack scan
stack = StackScan('path/to/my_stack_scan.h5')
stack.plot_summary(channel_roi=(80, 101), map_roi=[-1, 1, -1, 1])
```
