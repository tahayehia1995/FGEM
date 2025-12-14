"""File upload components for Streamlit."""

import streamlit as st
import json
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path


def upload_config_file() -> Optional[tuple[Dict[str, Any], Optional[str]]]:
    """Upload and parse configuration JSON file.
    
    Returns:
        Tuple of (configuration dictionary, temp_file_path) if file uploaded, None otherwise.
        The temp_file_path can be used for resolving relative paths in the config.
    """
    uploaded_file = st.file_uploader(
        "Upload Configuration File (JSON)",
        type=['json'],
        help="Upload a JSON configuration file to load settings"
    )
    
    if uploaded_file is not None:
        try:
            config = json.load(uploaded_file)
            
            # Save to temp location for path resolution
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "fgem_configs"
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / uploaded_file.name
            with open(temp_file, "w") as f:
                json.dump(config, f, indent=2)
            
            st.success("Configuration file loaded successfully!")
            return config, str(temp_file)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON file: {e}")
            return None
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    
    return None


def upload_data_file(file_type: str, file_extensions: list = ['csv']) -> Optional[str]:
    """Upload data file and save to temporary location.
    
    Args:
        file_type: Type of file (e.g., 'reservoir', 'market', 'weather').
        file_extensions: List of allowed file extensions.
        
    Returns:
        Path to saved file if uploaded, None otherwise.
    """
    uploaded_file = st.file_uploader(
        f"Upload {file_type.title()} Data File",
        type=file_extensions,
        help=f"Upload {file_type} data file"
    )
    
    if uploaded_file is not None:
        try:
            # Save to temporary location
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            
            file_path = temp_dir / f"{file_type}_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"{file_type.title()} file uploaded successfully!")
            return str(file_path)
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return None
    
    return None


def download_config_file(config: Dict[str, Any], filename: str = "config.json"):
    """Create download button for configuration file.
    
    Args:
        config: Configuration dictionary to download.
        filename: Name of the file to download.
    """
    json_str = json.dumps(config, indent=4)
    st.download_button(
        label="Download Configuration",
        data=json_str,
        file_name=filename,
        mime="application/json"
    )

