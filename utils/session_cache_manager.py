"""
SessionCacheManager for DeconomiX GUI
Manages per-session DCXCache objects, persistence, and session operations.
All code and comments in English.
"""
import os
import threading
import pickle
import uuid
import datetime
from utils.DeconomixCache import DCXCache
from utils.DTD_config import DTDConfig
from utils.ADTD_config import ADTDConfig

class SessionCacheManager:
    def __init__(self, cache_dir="cache", archive_dir="cache/archive", debug=False):
        self.cache_dir = cache_dir
        self.archive_dir = archive_dir
        self.debug = debug
        self._lock = threading.Lock()
        self._sessions = {}  # session_id -> DCXCache
        self._meta = {}      # session_id -> metadata dict
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
        self._load_all_sessions()

    def _log(self, msg):
        if self.debug:
            print(f"[SessionCacheManager] {msg}")

    def _session_path(self, session_id):
        return os.path.join(self.cache_dir, f"{session_id}.pkl")

    def _archive_path(self, session_id):
        return os.path.join(self.archive_dir, f"{session_id}.pkl")

    def _load_all_sessions(self):
        for fname in os.listdir(self.cache_dir):
            if fname.endswith(".pkl"):
                session_id = fname[:-4]
                try:
                    with open(self._session_path(session_id), "rb") as f:
                        self._sessions[session_id] = pickle.load(f)
                    self._meta[session_id] = {"created": os.path.getctime(self._session_path(session_id))}
                    self._log(f"Loaded session {session_id}")
                except Exception as e:
                    self._log(f"Failed to load session {session_id}: {e}")

    def create_session(self, name=None):
        with self._lock:
            session_id = datetime.datetime.now().strftime("%Y%m%d#%H%M%S#") + str(uuid.uuid4())[:8]
            cache = DCXCache(None, DTDConfig(), ADTDConfig())
            self._sessions[session_id] = cache
            self._meta[session_id] = {
                "created": datetime.datetime.now().isoformat(),
                "name": name or session_id,
                "status": "active"
            }
            self.save_session(session_id)
            self._log(f"Created new session {session_id}")
            return session_id

    def get_session(self, session_id):
        with self._lock:
            return self._sessions.get(session_id)

    def save_session(self, session_id):
        with self._lock:
            cache = self._sessions.get(session_id)
            if cache is not None:
                with open(self._session_path(session_id), "wb") as f:
                    pickle.dump(cache, f)
                self._log(f"Saved session {session_id}")

    def delete_session(self, session_id):
        with self._lock:
            self._sessions.pop(session_id, None)
            self._meta.pop(session_id, None)
            try:
                os.remove(self._session_path(session_id))
                self._log(f"Deleted session {session_id}")
            except FileNotFoundError:
                pass

    def archive_session(self, session_id):
        with self._lock:
            if session_id in self._sessions:
                with open(self._archive_path(session_id), "wb") as f:
                    pickle.dump(self._sessions[session_id], f)
                self.delete_session(session_id)
                self._log(f"Archived session {session_id}")

    def restore_session(self, session_id):
        with self._lock:
            archive_path = self._archive_path(session_id)
            if os.path.exists(archive_path):
                with open(archive_path, "rb") as f:
                    cache = pickle.load(f)
                self._sessions[session_id] = cache
                self._meta[session_id] = {"created": os.path.getctime(archive_path), "status": "active"}
                os.remove(archive_path)
                self.save_session(session_id)
                self._log(f"Restored session {session_id}")

    def list_sessions(self):
        with self._lock:
            return [
                {"session_id": sid, **self._meta[sid]} for sid in self._sessions.keys()
            ]

    def list_archived_sessions(self):
        with self._lock:
            archived = []
            for fname in os.listdir(self.archive_dir):
                if fname.endswith(".pkl"):
                    session_id = fname[:-4]
                    archived.append({
                        "session_id": session_id,
                        "created": os.path.getctime(self._archive_path(session_id)),
                        "status": "archived"
                    })
            return archived

    def rename_session(self, session_id, new_name):
        with self._lock:
            if session_id in self._meta:
                self._meta[session_id]["name"] = new_name
                self._log(f"Renamed session {session_id} to {new_name}")

    def export_session(self, session_id, export_path):
        with self._lock:
            cache = self._sessions.get(session_id)
            if cache is not None:
                with open(export_path, "wb") as f:
                    pickle.dump(cache, f)
                self._log(f"Exported session {session_id} to {export_path}")

    def import_session(self, import_path, name=None):
        with self._lock:
            with open(import_path, "rb") as f:
                cache = pickle.load(f)
            session_id = datetime.datetime.now().strftime("%Y%m%d#%H%M%S#") + str(uuid.uuid4())[:8]
            self._sessions[session_id] = cache
            self._meta[session_id] = {
                "created": datetime.datetime.now().isoformat(),
                "name": name or session_id,
                "status": "active"
            }
            self.save_session(session_id)
            self._log(f"Imported session {session_id}")
            return session_id

    def cleanup_sessions(self, max_sessions=None, max_age_days=None):
        # Optional: implement auto-cleanup logic based on config
        pass

def get_session_cache(session_id):
    """
    Returns the session cache for the given session_id. If no cache exists yet, creates a new one automatically.
    """
    from utils.session_cache_manager import session_manager
    cache = session_manager.get_session(session_id)
    if cache is None:
        # Create new cache for this session_id
        from utils.DeconomixCache import DCXCache
        from utils.DTD_config import DTDConfig
        from utils.ADTD_config import ADTDConfig
        cache = DCXCache(None, DTDConfig(), ADTDConfig())
        session_manager._sessions[session_id] = cache
        session_manager._meta[session_id] = {
            "created": datetime.datetime.now().isoformat(),
            "name": session_id,
            "status": "active"
        }
        session_manager.save_session(session_id)
        session_manager._log(f"Auto-created session cache for {session_id}")
    return cache

# Global singleton for the app
session_manager = SessionCacheManager()
