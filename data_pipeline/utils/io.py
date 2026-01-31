"""
I/O utilities for JSON, JSONL, and image files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Iterator, Union, Optional
import zipfile
import tarfile

logger = logging.getLogger(__name__)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data as dictionary
    """
    file_path = Path(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(
    data: Dict[str, Any], 
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Destination path
        indent: JSON indentation level
        ensure_ascii: Whether to escape non-ASCII characters
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load JSONL file (one JSON object per line).
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of parsed JSON objects
    """
    file_path = Path(file_path)
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
    return data


def iter_jsonl(file_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """
    Iterate over JSONL file without loading all into memory.
    
    Args:
        file_path: Path to JSONL file
        
    Yields:
        Parsed JSON objects one at a time
    """
    file_path = Path(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def save_jsonl(
    data: List[Dict[str, Any]], 
    file_path: Union[str, Path],
    ensure_ascii: bool = False,
) -> int:
    """
    Save data to JSONL file.
    
    Args:
        data: List of data items to save
        file_path: Destination path
        ensure_ascii: Whether to escape non-ASCII characters
        
    Returns:
        Number of items written
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=ensure_ascii)
            f.write('\n')
            count += 1
    
    return count


def append_jsonl(
    items: Union[Dict[str, Any], List[Dict[str, Any]]],
    file_path: Union[str, Path],
    ensure_ascii: bool = False,
) -> int:
    """
    Append items to JSONL file.
    
    Args:
        items: Single item or list of items to append
        file_path: Destination path
        ensure_ascii: Whether to escape non-ASCII characters
        
    Returns:
        Number of items appended
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(items, dict):
        items = [items]
    
    count = 0
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in items:
            json.dump(item, f, ensure_ascii=ensure_ascii)
            f.write('\n')
            count += 1
    
    return count


def extract_archive(
    archive_path: Union[str, Path],
    dest_dir: Union[str, Path],
    remove_archive: bool = False,
) -> bool:
    """
    Extract ZIP or TAR archive.
    
    Args:
        archive_path: Path to archive file
        dest_dir: Destination directory
        remove_archive: Whether to delete archive after extraction
        
    Returns:
        True if extraction successful
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(dest_dir)
                
        elif archive_path.suffix in ('.tar', '.gz', '.tgz'):
            mode = 'r:gz' if archive_path.suffix in ('.gz', '.tgz') else 'r'
            with tarfile.open(archive_path, mode) as tf:
                tf.extractall(dest_dir)
        else:
            logger.error(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        logger.info(f"Extracted {archive_path} to {dest_dir}")
        
        if remove_archive:
            archive_path.unlink()
            logger.info(f"Removed archive: {archive_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        return False


def count_lines(file_path: Union[str, Path]) -> int:
    """
    Count lines in a file efficiently.
    
    Args:
        file_path: Path to file
        
    Returns:
        Number of lines
    """
    count = 0
    with open(file_path, 'rb') as f:
        for _ in f:
            count += 1
    return count
