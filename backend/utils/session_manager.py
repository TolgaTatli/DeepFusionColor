"""
Session Management for Streaming Analysis
==========================================
Manages temporary storage of images and metadata for SSE streaming.
"""

import uuid
import time
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)

# Session storage directory
SESSIONS_DIR = Path(__file__).parent.parent / 'temp_sessions'
SESSIONS_DIR.mkdir(exist_ok=True)

# In-memory session registry
SESSIONS: Dict[str, dict] = {}

# Session expiry time (seconds)
SESSION_MAX_AGE = 3600  # 1 hour


def generate_session_id() -> str:
    """
    Generate unique session ID.
    
    Returns:
        str: UUID-based session ID
    """
    return str(uuid.uuid4())


def store_session_data(session_id: str, ir_pil, vis_pil, fused_pil, 
                       metrics_dict: dict, method_name: str):
    """
    Store images and metadata for a session.
    
    Args:
        session_id: Unique session identifier
        ir_pil: PIL Image (infrared source)
        vis_pil: PIL Image (visible source)
        fused_pil: PIL Image (fusion result)
        metrics_dict: Calculated metrics
        method_name: Fusion method used
    """
    try:
        # Create session directory
        session_path = SESSIONS_DIR / session_id
        session_path.mkdir(exist_ok=True)
        
        # Save images
        ir_pil.save(session_path / 'ir.png')
        vis_pil.save(session_path / 'vis.png')
        fused_pil.save(session_path / 'fused.png')
        
        # Save metadata
        metadata = {
            'metrics': metrics_dict,
            'method': method_name,
            'timestamp': time.time()
        }
        with open(session_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        # Register in memory
        SESSIONS[session_id] = {
            'timestamp': time.time(),
            'path': str(session_path),
            'method': method_name
        }
        
        logger.info(f"Session {session_id} stored successfully")
        
    except Exception as e:
        logger.error(f"Failed to store session {session_id}: {e}")
        raise


def load_session_data(session_id: str) -> Optional[Tuple]:
    """
    Load session data for analysis.
    
    Args:
        session_id: Session identifier
    
    Returns:
        tuple: (ir_image_pil, vis_image_pil, fused_image_pil, metrics_dict, method_name)
        None if session not found or expired
    """
    try:
        if session_id not in SESSIONS:
            logger.warning(f"Session {session_id} not found")
            return None
        
        session_info = SESSIONS[session_id]
        session_path = Path(session_info['path'])
        
        # Check if session expired
        age = time.time() - session_info['timestamp']
        if age > SESSION_MAX_AGE:
            logger.warning(f"Session {session_id} expired (age: {age:.0f}s)")
            cleanup_session(session_id)
            return None
        
        # Check if directory exists
        if not session_path.exists():
            logger.warning(f"Session directory not found: {session_path}")
            return None
        
        # Load images
        from PIL import Image
        ir_image = Image.open(session_path / 'ir.png')
        vis_image = Image.open(session_path / 'vis.png')
        fused_image = Image.open(session_path / 'fused.png')
        
        # Load metadata
        with open(session_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        metrics_dict = metadata['metrics']
        method_name = metadata['method']
        
        logger.info(f"Session {session_id} loaded successfully")
        
        return ir_image, vis_image, fused_image, metrics_dict, method_name
        
    except Exception as e:
        logger.error(f"Failed to load session {session_id}: {e}")
        return None


def cleanup_session(session_id: str):
    """
    Remove session data from disk and memory.
    
    Args:
        session_id: Session to cleanup
    """
    try:
        if session_id in SESSIONS:
            session_path = Path(SESSIONS[session_id]['path'])
            
            # Remove directory
            if session_path.exists():
                shutil.rmtree(session_path)
            
            # Remove from memory
            del SESSIONS[session_id]
            
            logger.info(f"Session {session_id} cleaned up")
    
    except Exception as e:
        logger.error(f"Failed to cleanup session {session_id}: {e}")


def cleanup_old_sessions(max_age: int = SESSION_MAX_AGE):
    """
    Cleanup sessions older than max_age.
    
    Args:
        max_age: Maximum age in seconds
    
    Returns:
        int: Number of sessions cleaned up
    """
    current_time = time.time()
    to_cleanup = []
    
    for session_id, info in SESSIONS.items():
        age = current_time - info['timestamp']
        if age > max_age:
            to_cleanup.append(session_id)
    
    for session_id in to_cleanup:
        cleanup_session(session_id)
    
    if to_cleanup:
        logger.info(f"Cleaned up {len(to_cleanup)} expired sessions")
    
    return len(to_cleanup)


def get_session_stats() -> dict:
    """
    Get statistics about active sessions.
    
    Returns:
        dict: Session statistics
    """
    return {
        'active_sessions': len(SESSIONS),
        'oldest_session_age': min(
            [time.time() - s['timestamp'] for s in SESSIONS.values()],
            default=0
        ),
        'newest_session_age': max(
            [time.time() - s['timestamp'] for s in SESSIONS.values()],
            default=0
        ),
        'sessions_dir': str(SESSIONS_DIR)
    }
