"""
Download utilities with resume support and progress tracking.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Callable
import hashlib
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_file_size(url: str, timeout: int = 30) -> Optional[int]:
    """
    Get remote file size via HEAD request.
    
    Args:
        url: Remote file URL
        timeout: Request timeout in seconds
        
    Returns:
        File size in bytes, or None if unavailable
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            return int(response.headers.get('content-length', 0))
    except Exception as e:
        logger.warning(f"Could not get file size for {url}: {e}")
    return None


def download_file(
    url: str,
    dest_path: Path,
    chunk_size: int = 8192,
    timeout: int = 30,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    show_progress: bool = True,
) -> bool:
    """
    Download file from URL with progress tracking.
    
    Args:
        url: Source URL
        dest_path: Destination file path
        chunk_size: Download chunk size in bytes
        timeout: Request timeout in seconds
        progress_callback: Optional callback(downloaded, total)
        show_progress: Whether to show tqdm progress bar
        
    Returns:
        True if download successful
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        progress_bar = None
        if show_progress and total_size > 0:
            progress_bar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=dest_path.name,
            )
        
        downloaded = 0
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if progress_bar:
                        progress_bar.update(len(chunk))
                    if progress_callback:
                        progress_callback(downloaded, total_size)
        
        if progress_bar:
            progress_bar.close()
        
        logger.info(f"Downloaded: {dest_path}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial file
        return False


def download_with_resume(
    url: str,
    dest_path: Path,
    chunk_size: int = 8192,
    timeout: int = 30,
    show_progress: bool = True,
) -> bool:
    """
    Download file with resume support for interrupted downloads.
    
    Args:
        url: Source URL
        dest_path: Destination file path  
        chunk_size: Download chunk size in bytes
        timeout: Request timeout in seconds
        show_progress: Whether to show progress bar
        
    Returns:
        True if download successful
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check for partial download
    existing_size = 0
    if dest_path.exists():
        existing_size = dest_path.stat().st_size
    
    # Get total file size
    total_size = get_file_size(url, timeout)
    
    # If file already complete, skip
    if total_size and existing_size == total_size:
        logger.info(f"File already downloaded: {dest_path}")
        return True
    
    # Prepare headers for resume
    headers = {}
    if existing_size > 0 and total_size:
        headers['Range'] = f'bytes={existing_size}-'
        logger.info(f"Resuming download from byte {existing_size}")
    
    try:
        response = requests.get(
            url, 
            stream=True, 
            timeout=timeout,
            headers=headers,
        )
        
        # Handle resume response (206) or fresh download (200)
        if response.status_code not in (200, 206):
            response.raise_for_status()
        
        # For fresh download, reset existing_size
        if response.status_code == 200:
            existing_size = 0
        
        content_length = int(response.headers.get('content-length', 0))
        final_size = existing_size + content_length
        
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                total=final_size,
                initial=existing_size,
                unit='B',
                unit_scale=True,
                desc=dest_path.name,
            )
        
        mode = 'ab' if existing_size > 0 else 'wb'
        with open(dest_path, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    if progress_bar:
                        progress_bar.update(len(chunk))
        
        if progress_bar:
            progress_bar.close()
        
        logger.info(f"Downloaded: {dest_path}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return False


def calculate_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path: Path to file
        chunk_size: Read chunk size
        
    Returns:
        MD5 hash string
    """
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def verify_download(file_path: Path, expected_md5: Optional[str] = None) -> bool:
    """
    Verify downloaded file integrity.
    
    Args:
        file_path: Path to downloaded file
        expected_md5: Expected MD5 hash (optional)
        
    Returns:
        True if file exists and hash matches (if provided)
    """
    if not file_path.exists():
        return False
    
    if file_path.stat().st_size == 0:
        return False
    
    if expected_md5:
        actual_md5 = calculate_md5(file_path)
        if actual_md5 != expected_md5:
            logger.error(f"MD5 mismatch for {file_path}: expected {expected_md5}, got {actual_md5}")
            return False
    
    return True
