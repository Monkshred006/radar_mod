"""RadarLoader implementation preserving original NumPy dtype and format."""

import json
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from module_01_radar_input.validation import validate_file_exists, CorruptedFileError, validate_frame


class RadarLoader:
    """Loader for raw radar data files (.npy, .npz, .csv, .json).

    Preserves original numerical dtype and data structure (int, float, complex).
    Does NOT convert raw data to FP16 or normalize.
    """

    def __init__(self, npz_key: Optional[str] = None):
        """
        Args:
            npz_key: Key name to extract array from .npz archives. If None,
                defaults to the first array key found in the archive.
        """
        self.npz_key = npz_key

    def load_frame(self, filepath: Union[str, Path]) -> np.ndarray:
        """Load a single raw radar frame from disk.

        Args:
            filepath: Path to the radar data file.

        Returns:
            np.ndarray: Raw radar data array with original dtype preserved.
        """
        path = validate_file_exists(filepath)
        suffix = path.suffix.lower()

        try:
            if suffix == ".npy":
                data = np.load(path, allow_pickle=False)
            elif suffix == ".npz":
                with np.load(path, allow_pickle=False) as npz:
                    key = self.npz_key if (self.npz_key and self.npz_key in npz) else npz.files[0]
                    data = npz[key]
            elif suffix == ".csv":
                df = pd.read_csv(path)
                data = df.to_numpy()
            elif suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                if isinstance(content, dict) and "data" in content:
                    data = np.array(content["data"])
                elif isinstance(content, list):
                    data = np.array(content)
                else:
                    raise CorruptedFileError(f"Unsupported JSON format in {path}")
            else:
                raise CorruptedFileError(f"Unsupported file format extension: {suffix}")

        except Exception as e:
            if isinstance(e, (CorruptedFileError, FileNotFoundError)):
                raise e
            raise CorruptedFileError(f"Failed to load radar frame file '{path}': {str(e)}") from e

        # Validate loaded frame (checks NaN/Inf/empty)
        validate_frame(data)
        return data
