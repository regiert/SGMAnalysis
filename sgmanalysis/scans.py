import h5py
import numpy as np
import os
import re
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class MapScan:
    """
    Represents a single map scan from the SGM beamline.
    
    This class loads and holds the data from a map scan, including metadata,
    coordinates, and paths to raw data files. It also provides methods
    for plotting the data.
    """
    def __init__(self, file_path):
        """
        Initializes a MapScan object by reading the data from an HDF5 file.

        Args:
            file_path (str): The path to the HDF5 file from a map scan.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        self.file_path = file_path
        self.directory = os.path.dirname(file_path)

        # --- Initialize attributes ---
        self.x = np.array([])
        self.y = np.array([])
        self.scan_name = "N/A"
        self.project = "N/A"
        self.energy = -1.0
        self.mcc_file = None
        self._mcc_data = None
        self.mcc_channel_names = []
        self.sdd_files = {}
        self.xeol_file = None
        self._xeol_data = None

        self._load_data()

    def _load_data(self):
        """Internal method to load all data from the HDF5 file and associated files."""
        with h5py.File(self.file_path, 'r') as f:
            # --- Extract Metadata ---
            if 'scan_metadata' in f:
                metadata_attrs = f['scan_metadata'].attrs
                self.scan_name = metadata_attrs.get('scan_name', 'N/A')
                self.project = metadata_attrs.get('project', 'N/A')
                
                energy = metadata_attrs.get('energy', None)
                if energy is None and 'initial_motor_positions' in f and 'all_beamline_motors_snapshot' in f['initial_motor_positions']:
                    motors_snapshot_attrs = f['initial_motor_positions/all_beamline_motors_snapshot'].attrs
                    energy = motors_snapshot_attrs.get('energy', -1.0)
                
                self.energy = float(energy) if energy is not None else -1.0
            else:
                print("Warning: /scan_metadata group not found in HDF5 file.", file=sys.stderr)

            # --- Extract Coordinates ---
            if 'hexapod_waves/x' in f and 'hexapod_waves/y' in f:
                self.x = f['hexapod_waves/x'][:]
                self.y = f['hexapod_waves/y'][:]
            else:
                print("Warning: Coordinate data (hexapod_waves/x or y) not found.", file=sys.stderr)

        # --- Find and prepare Raw Data Files ---
        self.mcc_file = self._find_first_file('mcc*.csv')
        self.xeol_file = self._find_first_file('xeol*.bin')

        sdd_out_files = glob.glob(os.path.join(self.directory, 'sdd*.out'))
        sdd_bin_files = glob.glob(os.path.join(self.directory, 'sdd*_*.bin'))
        for sdd_file_path in sdd_out_files + sdd_bin_files:
            match = re.match(r'(sdd\d+)', os.path.basename(sdd_file_path))
            if match:
                detector_name = match.group(1)
                self.sdd_files[detector_name] = sdd_file_path
    
    def _find_first_file(self, pattern):
        """Finds the first file in the directory matching a pattern."""
        files = glob.glob(os.path.join(self.directory, pattern))
        return files[0] if files else None

    @property
    def mcc_data(self):
        """Lazy-loads MCC data from the CSV file."""
        if self._mcc_data is None and self.mcc_file:
            with open(self.mcc_file, 'r') as mcc_f:
                header = mcc_f.readline().strip()
                if header.startswith('#'):
                    self.mcc_channel_names = [name.strip() for name in header[1:].split(',')]
            self._mcc_data = np.genfromtxt(self.mcc_file, delimiter=',', skip_header=1)
        return self._mcc_data

    @property
    def xeol_data(self):
        """Lazy-loads XEOL data from the binary file."""
        if self._xeol_data is None and self.xeol_file:
            self._xeol_data = np.fromfile(self.xeol_file, dtype=np.uint32)
        return self._xeol_data
        
    def get_sdd_data(self, detector_name):
        """
        Loads the 2D spectrum data for a specific SDD detector.

        Args:
            detector_name (str): The name of the detector (e.g., 'sdd1').

        Returns:
            np.ndarray: A 2D numpy array where each row is a spectrum. 
                        Returns None if the detector or file is not found.
        """
        sdd_filepath = self.sdd_files.get(detector_name)
        if not sdd_filepath or not os.path.exists(sdd_filepath):
            print(f"Error: Data file for detector {detector_name} not found.", file=sys.stderr)
            return None

        try:
            pixels_per_spectrum = 256
            data_1d = np.fromfile(sdd_filepath, dtype=np.uint32)
            
            if data_1d.size == 0:
                print(f"Warning: File for {detector_name} is empty.", file=sys.stderr)
                return None

            num_spectra = len(data_1d) // pixels_per_spectrum
            
            if num_spectra != self.x.size:
                print(f"Warning: Mismatch in number of spectra ({num_spectra}) and scan points ({self.x.size}) for {detector_name}. Truncating.", file=sys.stderr)
                num_spectra = min(num_spectra, self.x.size)

            clean_size = num_spectra * pixels_per_spectrum
            return data_1d[:clean_size].reshape((num_spectra, pixels_per_spectrum))

        except Exception as e:
            print(f"An error occurred while reading data for {detector_name}: {e}", file=sys.stderr)
            return None

    def __repr__(self):
        return (f"MapScan(scan_name='{self.scan_name}', project='{self.project}', "
                f"energy={self.energy:.2f} eV, detectors={list(self.sdd_files.keys())})")

    def plot_overview(self, channel_roi, roll_shift=0, as_scatter_plot: bool = False, map_roi=None, contrast=None, mcc_channels=None):
        """
        Loads SDD data for all available detectors from a map scan, and for each one,
        plots a map based on an ROI sum and a total summed spectrum. Vertical bars
        are added to the spectrum plot to highlight the channel ROI.

        Args:
            channel_roi (tuple): A tuple of two integers defining the start and end of the ROI for the map, e.g., (80, 101).
            roll_shift (int): The number of positions to shift the data for roll correction.
            as_scatter_plot (bool): If True, plots as a scatter plot; otherwise, plots as a heatmap.
            map_roi (list, optional): A list of four coordinates [x1, x2, y1, y2] to define a
                                      rectangular ROI for the summed spectrum. Defaults to None.
            contrast (list, optional): A list of two numbers [vmin, vmax] for the plot contrast.
            mcc_channels (list, optional): A list of MCC channel numbers to plot maps for.
        """
        if not self.sdd_files:
            print("Error: No SDD files found for this scan.", file=sys.stderr)
            return

        if self.x.size == 0 or self.y.size == 0:
            print("Error: Coordinate data not found.", file=sys.stderr)
            return

        num_detectors = len(self.sdd_files)
        
        num_mcc_plots = 0
        if self.mcc_data is not None and mcc_channels:
            num_mcc_plots = len(mcc_channels)

        num_xeol_plots = 1 if self.xeol_data is not None else 0

        if num_detectors == 0 and num_mcc_plots == 0 and num_xeol_plots == 0:
            print("No SDD, MCC, or XEOL detectors found to plot.", file=sys.stderr)
            return

        # --- Create Figure and Title ---
        total_rows = num_detectors + num_mcc_plots + num_xeol_plots
        fig, axes = plt.subplots(total_rows, 2, figsize=(12, 5 * total_rows), squeeze=False)
        
        title = f"Scan: {self.scan_name}  |  Project: {self.project}  |  Energy: {self.energy:.2f} eV"
        fig.suptitle(title, fontsize=14)

        # Sort detectors by name (e.g., sdd1, sdd2, ...)
        sorted_detectors = sorted(self.sdd_files.keys())

        for i, detector_name in enumerate(sorted_detectors):
            ax_map = axes[i, 0]
            ax_spec = axes[i, 1]
            
            spectra_2d = self.get_sdd_data(detector_name)
            if spectra_2d is None:
                ax_map.set_title(f"{detector_name} - Data Not Found")
                continue

            current_x, current_y = self.x[:spectra_2d.shape[0]], self.y[:spectra_2d.shape[0]]

            # --- Data for Map Plot (Left) ---
            intensity = np.sum(spectra_2d[:, channel_roi[0]:channel_roi[1]], axis=1)
            if roll_shift != 0:
                intensity = np.roll(intensity, shift=roll_shift)

            # --- Data for Spectrum Plot (Right) ---
            spectrum_title = f"{detector_name} - Total Spectrum"
            if map_roi:
                x1, x2 = sorted(map_roi[0:2])
                y1, y2 = sorted(map_roi[2:4])
                mask = (current_x >= x1) & (current_x <= x2) & (current_y >= y1) & (current_y <= y2)
                
                if np.any(mask):
                    total_spectrum = np.sum(spectra_2d[mask], axis=0)
                    spectrum_title = f"{detector_name} - Spectrum from Map ROI"
                else:
                    total_spectrum = np.sum(spectra_2d, axis=0)
            else:
                total_spectrum = np.sum(spectra_2d, axis=0)
            
            bins = np.arange(spectra_2d.shape[1])

            # --- Plotting ---
            plot_kwargs = {}
            if contrast and len(contrast) == 2:
                plot_kwargs['vmin'] = contrast[0]
                plot_kwargs['vmax'] = contrast[1]

            if as_scatter_plot:
                scatter = ax_map.scatter(current_x, current_y, c=intensity, cmap='viridis', marker='s', edgecolors='none', **plot_kwargs)
                fig.colorbar(scatter, ax=ax_map, label=f"Counts (ROI: {channel_roi[0]}-{channel_roi[1]})")
            else:
                tripcolor = ax_map.tripcolor(current_x, current_y, intensity, shading='gouraud', **plot_kwargs)
                fig.colorbar(tripcolor, ax=ax_map, label=f"Counts (ROI: {channel_roi[0]}-{channel_roi[1]})")
            
            if map_roi:
                x1, x2 = sorted(map_roi[0:2])
                y1, y2 = sorted(map_roi[2:4])
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='r', facecolor='none', linestyle='--')
                ax_map.add_patch(rect)

            ax_map.set_title(f"{detector_name} - ROI Map")
            ax_map.set_xlabel("Hexapod X")
            ax_map.set_ylabel("Hexapod Y")
            ax_map.set_aspect('equal', adjustable='box')

            ax_spec.plot(bins, total_spectrum)
            ax_spec.axvline(x=channel_roi[0], color='red', linestyle=':', linewidth=1.5, label=f'Channel ROI: {channel_roi[0]}-{channel_roi[1]}')
            ax_spec.axvline(x=channel_roi[1], color='red', linestyle=':', linewidth=1.5)
            ax_spec.legend(loc='upper right', fontsize='small')
            ax_spec.set_title(spectrum_title)
            ax_spec.set_xlabel("Bin Number")
            ax_spec.set_ylabel("Total Intensity")
            ax_spec.grid(True)

        # --- Plot MCC Data ---
        if num_mcc_plots > 0:
            for i, mcc_channel in enumerate(mcc_channels):
                ax_map = axes[num_detectors + i, 0]
                ax_spec = axes[num_detectors + i, 1]
                ax_spec.axis('off')

                try:
                    channel_index = self.mcc_channel_names.index(f'ch{mcc_channel}')
                except ValueError:
                    ax_map.set_title(f"MCC Channel {mcc_channel} - Not Found")
                    continue

                intensity = self.mcc_data[:, channel_index]
                if roll_shift != 0:
                    intensity = np.roll(intensity, shift=roll_shift)

                if as_scatter_plot:
                    scatter = ax_map.scatter(self.x, self.y, c=intensity, cmap='viridis', marker='s', edgecolors='none')
                    fig.colorbar(scatter, ax=ax_map, label=f"MCC Channel {mcc_channel} Value")
                else:
                    tripcolor = ax_map.tripcolor(self.x, self.y, intensity, shading='gouraud')
                    fig.colorbar(tripcolor, ax=ax_map, label=f"MCC Channel {mcc_channel} Value")

                ax_map.set_title(f"MCC Channel {mcc_channel} - Map")
                ax_map.set_xlabel("Hexapod X")
                ax_map.set_ylabel("Hexapod Y")
                ax_map.set_aspect('equal', adjustable='box')

        # --- Plot XEOL Data ---
        if num_xeol_plots > 0:
            ax_map = axes[num_detectors + num_mcc_plots, 0]
            ax_spec = axes[num_detectors + num_mcc_plots, 1]
            ax_map.axis('off')

            ax_spec.plot(self.xeol_data)
            ax_spec.set_title("XEOL Spectrum")
            ax_spec.set_xlabel("Bin Number")
            ax_spec.set_ylabel("Total Intensity")
            ax_spec.grid(True)

class StackScan:
    """
    Represents a stack scan from the SGM beamline.
    """
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        self.file_path = file_path
        self.stack_dir = os.path.dirname(file_path)

        self.energies = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        self.scan_name = "N/A"
        self.project = "N/A"
        self.mcc_files = {}
        self.mcc_data = {}
        self.mcc_channel_names = []
        self.sdd_files = {}
        self.xeol_files = {}
        self.xeol_data = {}
        
        self._load_data()

    def _load_data(self):
        with h5py.File(self.file_path, 'r', swmr=True) as f:
            if 'stack_metadata' in f:
                metadata_attrs = f['stack_metadata'].attrs
                self.scan_name = metadata_attrs.get('scan_name', 'N/A')
                self.project = metadata_attrs.get('project', 'N/A')
            else:
                print("Warning: /stack_metadata group not found in HDF5 file.", file=sys.stderr)
                return

            if 'map_data/energy' in f:
                self.energies = f['map_data/energy'][:]
            else:
                print("Warning: 'map_data/energy' not found in HDF5 file.", file=sys.stderr)
                return

            if 'hexapod_waves/x' in f and 'hexapod_waves/y' in f:
                self.x = f['hexapod_waves/x'][:]
                self.y = f['hexapod_waves/y'][:]
            else:
                print("Warning: Coordinate data (hexapod_waves/x or y) not found.", file=sys.stderr)

            got_mcc_header = False
            for energy in self.energies:
                energy_str = f"{energy:.2f}".replace('.', '_')
                expected_subdir_name = f"{self.scan_name}_{energy_str}eV"
                en_dir_path = os.path.join(self.stack_dir, expected_subdir_name)

                if not os.path.isdir(en_dir_path):
                    continue

                mcc_file_list = glob.glob(os.path.join(en_dir_path, 'mcc*.csv'))
                if mcc_file_list:
                    mcc_file_path = mcc_file_list[0]
                    self.mcc_files[energy] = mcc_file_path
                    if not got_mcc_header:
                        with open(mcc_file_path, 'r') as mcc_f:
                            header = mcc_f.readline().strip()
                            if header.startswith('#'):
                                self.mcc_channel_names = [name.strip() for name in header[1:].split(',')]
                        got_mcc_header = True
                    self.mcc_data[energy] = np.genfromtxt(mcc_file_path, delimiter=',', skip_header=1)

                sdd_files_list = glob.glob(os.path.join(en_dir_path, 'sdd*.out')) + glob.glob(os.path.join(en_dir_path, 'sdd*_*.bin'))
                for sdd_file_path in sdd_files_list:
                    match = re.match(r'(sdd\d+)', os.path.basename(sdd_file_path))
                    if match:
                        detector_name = match.group(1)
                        if detector_name not in self.sdd_files:
                            self.sdd_files[detector_name] = {}
                        self.sdd_files[detector_name][energy] = sdd_file_path

                xeol_files_list = glob.glob(os.path.join(en_dir_path, 'xeol*.bin'))
                if xeol_files_list:
                    xeol_file_path = xeol_files_list[0]
                    self.xeol_files[energy] = xeol_file_path
                    self.xeol_data[energy] = np.fromfile(xeol_file_path, dtype=np.uint32)
    def __repr__(self):
        return (f"StackScan(scan_name='{self.scan_name}', project='{self.project}', "
                f"energies={len(self.energies)}, detectors={list(self.sdd_files.keys())})")

    def get_sdd_data(self, detector_name, energy):
        sdd_filepath = self.sdd_files.get(detector_name, {}).get(energy)
        if not sdd_filepath or not os.path.exists(sdd_filepath):
            return None
        try:
            pixels_per_spectrum = 256
            data_1d = np.fromfile(sdd_filepath, dtype=np.uint32)
            if data_1d.size == 0:
                return None
            num_spectra = len(data_1d) // pixels_per_spectrum
            if num_spectra != self.x.size:
                num_spectra = min(num_spectra, self.x.size)
            clean_size = num_spectra * pixels_per_spectrum
            return data_1d[:clean_size].reshape((num_spectra, pixels_per_spectrum))
        except Exception as e:
            print(f"Error reading SDD data for {detector_name} at {energy} eV: {e}", file=sys.stderr)
            return None

    def plot_summary(self, channel_roi, map_roi, roll_shift=0, as_scatter_plot: bool = False, contrast=None, mcc_channels=None, sdd_detectors_to_plot=None):
        if not self.sdd_files:
            print("Error: No SDD files found.", file=sys.stderr)
            return

        all_energies = np.array(sorted(self.energies))
        all_detector_names = sorted(self.sdd_files.keys())
        
        detector_names = [d for d in sdd_detectors_to_plot if d in all_detector_names] if sdd_detectors_to_plot else all_detector_names
        if not detector_names:
            print(f"Warning: None of the specified detectors found.", file=sys.stderr)
            return
        
        summary_data = {det: [] for det in detector_names}
        mcc_summary_data = {ch: [] for ch in mcc_channels} if mcc_channels else {}
        summary_energies = []

        x1_map, x2_map = sorted(map_roi[0:2])
        y1_map, y2_map = sorted(map_roi[2:4])
        spatial_mask = (self.x >= x1_map) & (self.x <= x2_map) & (self.y >= y1_map) & (self.y <= y2_map)

        for energy in all_energies:
            intensity_found = False
            for det_name in detector_names:
                spectra_2d = self.get_sdd_data(det_name, energy)
                if spectra_2d is not None:
                    selected_spectra = spectra_2d[spatial_mask]
                    roi_intensity = np.sum(selected_spectra[:, channel_roi[0]:channel_roi[1]])
                    summary_data[det_name].append(roi_intensity)
                    intensity_found = True
                else:
                    summary_data[det_name].append(np.nan)
            
            if mcc_channels and self.mcc_data.get(energy) is not None:
                for mcc_channel in mcc_channels:
                    try:
                        channel_index = self.mcc_channel_names.index(f'ch{mcc_channel}')
                        mcc_summary_data[mcc_channel].append(np.mean(self.mcc_data[energy][spatial_mask, channel_index]))
                    except (ValueError, IndexError):
                        mcc_summary_data[mcc_channel].append(np.nan)
            
            if intensity_found:
                summary_energies.append(energy)
        
        threshold_energy = all_energies[len(all_energies) // 2] 

        averaged_maps = {det: {'before': {'sum': None, 'count': 0}, 'after': {'sum': None, 'count': 0}} for det in detector_names}
        for energy in all_energies:
            period = 'before' if energy < threshold_energy else 'after'
            for det_name in detector_names:
                spectra_2d = self.get_sdd_data(det_name, energy)
                if spectra_2d is not None:
                    if roll_shift != 0:
                        spectra_2d = np.roll(spectra_2d, shift=roll_shift, axis=0)
                    map_intensity = np.sum(spectra_2d[:, channel_roi[0]:channel_roi[1]], axis=1)
                    if averaged_maps[det_name][period]['sum'] is None:
                        averaged_maps[det_name][period]['sum'] = map_intensity.astype(np.float64)
                    else:
                        if averaged_maps[det_name][period]['sum'].shape == map_intensity.shape:
                            averaged_maps[det_name][period]['sum'] += map_intensity
                    averaged_maps[det_name][period]['count'] += 1
        
        final_maps = {det: {} for det in detector_names}
        for det_name in detector_names:
            for period in ['before', 'after']:
                if averaged_maps[det_name][period]['count'] > 0:
                    final_maps[det_name][period] = averaged_maps[det_name][period]['sum'] / averaged_maps[det_name][period]['count']
                else:
                    final_maps[det_name][period] = None

        num_detectors = len(detector_names)
        num_summary_plots = 1 + (1 if mcc_channels else 0) + (1 if self.xeol_data else 0)
        fig = plt.figure(figsize=(18, 5 * num_detectors + 5 * num_summary_plots))
        gs = gridspec.GridSpec(num_detectors + num_summary_plots, 3, figure=fig, height_ratios=[1] * num_detectors + [2] * num_summary_plots)
        
        fig.suptitle(f"Scan: {self.scan_name} | Project: {self.project}", fontsize=16)

        for i, det_name in enumerate(detector_names):
            ax_map_before = fig.add_subplot(gs[i, 0])
            ax_map_after = fig.add_subplot(gs[i, 1])
            ax_spec = fig.add_subplot(gs[i, 2])
            
            plot_kwargs = {'vmin': contrast[0], 'vmax': contrast[1]} if contrast else {}

            for ax, period in [(ax_map_before, 'before'), (ax_map_after, 'after')]:
                map_data = final_maps[det_name].get(period)
                if map_data is not None:
                    if as_scatter_plot:
                        ax.scatter(self.x, self.y, c=map_data, cmap='viridis', marker='s', edgecolors='none', **plot_kwargs)
                    else:
                        ax.tripcolor(self.x, self.y, map_data, shading='gouraud', **plot_kwargs)
                    ax.set_title(f"{det_name} - Avg {period.capitalize()} {threshold_energy:.2f} eV")
                else:
                    ax.set_title(f"{det_name} - No data {period} threshold")
                ax.set_xlabel("Hexapod X"); ax.set_ylabel("Hexapod Y")
                ax.set_aspect('equal', adjustable='box')

            representative_energy = min(summary_energies, key=lambda x: abs(x - threshold_energy)) if summary_energies else None
            if representative_energy:
                spectra_2d = self.get_sdd_data(det_name, representative_energy)
                if spectra_2d is not None:
                    spectrum_from_roi = np.sum(spectra_2d[spatial_mask], axis=0)
                    ax_spec.plot(np.arange(spectra_2d.shape[1]), spectrum_from_roi)
                    ax_spec.axvline(x=channel_roi[0], color='r', linestyle=':'); ax_spec.axvline(x=channel_roi[1], color='r', linestyle=':')
                    ax_spec.set_title(f"{det_name} - Spectrum at {representative_energy:.2f} eV")
                    ax_spec.grid(True)

        summary_plot_index = num_detectors
        if summary_energies:
            ax_summary = fig.add_subplot(gs[summary_plot_index, :])
            summary_plot_index += 1
            for det_name in detector_names:
                ax_summary.plot(summary_energies, summary_data[det_name], 'o-', label=det_name)
            ax_summary.axvline(x=threshold_energy, color='purple', linestyle='--', label=f'Threshold: {threshold_energy:.2f} eV')
            ax_summary.set_title("Energy Dependence of Intensity in Map ROI")
            ax_summary.set_xlabel("Energy (eV)"); ax_summary.set_ylabel("Total Intensity in ROIs")
            ax_summary.legend(); ax_summary.grid(True)

        if mcc_channels:
            ax_mcc_summary = fig.add_subplot(gs[summary_plot_index, :])
            summary_plot_index += 1
            for mcc_channel in mcc_channels:
                ax_mcc_summary.plot(summary_energies, mcc_summary_data[mcc_channel], 'o-', label=f'MCC ch{mcc_channel}')
            ax_mcc_summary.axvline(x=threshold_energy, color='purple', linestyle='--')
            ax_mcc_summary.set_title("MCC Channel Dependence on Energy in Map ROI")
            ax_mcc_summary.set_xlabel("Energy (eV)"); ax_mcc_summary.set_ylabel("Mean Value in ROI")
            ax_mcc_summary.legend(); ax_mcc_summary.grid(True)

        if self.xeol_data:
            xeol_energies = sorted(self.xeol_data.keys())
            if xeol_energies:
                xeol_stack = np.array([self.xeol_data[en] for en in xeol_energies])
                ax_xeol = fig.add_subplot(gs[summary_plot_index, :])
                im = ax_xeol.imshow(xeol_stack.T, aspect='auto', extent=[min(xeol_energies), max(xeol_energies), 0, xeol_stack.shape[1]], origin='lower', cmap='viridis')
                fig.colorbar(im, ax=ax_xeol, label="Intensity")
                ax_xeol.set_title("XEOL Spectrum vs. Excitation Energy")
                ax_xeol.set_xlabel("Energy (eV)"); ax_xeol.set_ylabel("XEOL Bin Number")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


