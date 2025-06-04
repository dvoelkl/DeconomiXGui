"""
SessionCacheManager for DeconomiX GUI
Manages per-session DCXCache objects, persistence, and session operations.
"""

import os
import threading
import pickle
import uuid
import datetime
import json
import shutil
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

        self.max_sessions = -1
        self.session_max_age_days = -1

    def _log(self, msg):
        if self.debug:
            print(f"[SessionCacheManager] {msg}")

    def _meta_path(self, session_id, archived=False):
        if archived:
            return os.path.join(self.archive_dir, f"{session_id}.meta.json")
        return os.path.join(self.cache_dir, f"{session_id}.meta.json")

    def _session_path(self, session_id, archived=False):
        if archived:
            return os.path.join(self.archive_dir, f"{session_id}.pkl")
        return os.path.join(self.cache_dir, f"{session_id}.pkl")

    def _save_meta(self, session_id):
        meta = self._meta.get(session_id)
        if meta is not None:
            with open(self._meta_path(session_id), "w") as f:
                json.dump(meta, f)

    def _load_meta(self, session_id):
        try:
            with open(self._meta_path(session_id), "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _load_all_sessions(self):
        for fname in os.listdir(self.cache_dir):
            if fname.endswith(".pkl"):
                session_id = fname[:-4]
                try:
                    with open(self._session_path(session_id), "rb") as f:
                        self._sessions[session_id] = pickle.load(f)
                    meta = self._load_meta(session_id)
                    if meta:
                        self._meta[session_id] = meta
                    else:
                        self._meta[session_id] = {"created": os.path.getctime(self._session_path(session_id))}
                    self._log(f"Loaded session {session_id}")
                except Exception as e:
                    self._log(f"Failed to load session {session_id}: {e}")

    def create_session(self, name=None):
        session_id = datetime.datetime.now().strftime("%Y%m%d#%H%M%S#") + str(uuid.uuid4())[:8]
        cache = DCXCache(None, DTDConfig(), ADTDConfig())
        cache.session_name = name or "Session " + session_id[:8]
        self._sessions[session_id] = cache
        self._meta[session_id] = {
            "created": datetime.datetime.now().isoformat(),
            "name": name or "Session " + session_id[:8],
            "status": "active"
        }
        self.save_session(session_id)
        self._save_meta(session_id)
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
                self._save_meta(session_id)
                self._log(f"Saved session {session_id}")

    def delete_session(self, session_id):
        with self._lock:
            self._sessions.pop(session_id, None)
            self._meta.pop(session_id, None)
            # Try both locations
            for archived in [False, True]:
                try:
                    os.remove(self._session_path(session_id, archived=archived))
                except FileNotFoundError:
                    pass
                try:
                    os.remove(self._meta_path(session_id, archived=archived))
                except FileNotFoundError:
                    pass
            self._log(f"Deleted session {session_id}")

    def archive_session(self, session_id):
        with self._lock:
            # Move .pkl
            src_pkl = self._session_path(session_id)
            dst_pkl = self._session_path(session_id, archived=True)
            if os.path.exists(src_pkl):
                shutil.move(src_pkl, dst_pkl)
            # Move .meta.json
            src_meta = self._meta_path(session_id)
            dst_meta = self._meta_path(session_id, archived=True)
            if os.path.exists(src_meta):
                shutil.move(src_meta, dst_meta)
            self._sessions.pop(session_id, None)
            self._meta.pop(session_id, None)
            self._log(f"Archived session {session_id}")

    def restore_session(self, session_id):
        with self._lock:
            # Move .pkl
            src_pkl = self._session_path(session_id, archived=True)
            dst_pkl = self._session_path(session_id)
            if os.path.exists(src_pkl):
                shutil.move(src_pkl, dst_pkl)
            # Move .meta.json
            src_meta = self._meta_path(session_id, archived=True)
            dst_meta = self._meta_path(session_id)
            if os.path.exists(src_meta):
                shutil.move(src_meta, dst_meta)
            # Reload session
            try:
                with open(dst_pkl, "rb") as f:
                    cache = pickle.load(f)
                self._sessions[session_id] = cache
                meta = self._load_meta(session_id)
                if meta:
                    self._meta[session_id] = meta
                else:
                    self._meta[session_id] = {"created": os.path.getctime(dst_pkl), "status": "active"}
                self._log(f"Restored session {session_id}")
            except Exception as e:
                self._log(f"Failed to restore session {session_id}: {e}")

    def list_sessions(self):
        with self._lock:
            sessions = []
            for fname in os.listdir(self.cache_dir):
                if fname.endswith(".pkl"):
                    session_id = fname[:-4]
                    meta = self._load_meta(session_id)
                    if meta is None:
                        meta = {"created": os.path.getctime(self._session_path(session_id)), "status": "active"}
                    sessions.append({"session_id": session_id, **meta})
            return sessions

    def list_archived_sessions(self):
        with self._lock:
            archived = []
            for fname in os.listdir(self.archive_dir):
                if fname.endswith(".pkl"):
                    session_id = fname[:-4]
                    meta = None
                    try:
                        with open(self._meta_path(session_id, archived=True), "r") as f:
                            meta = json.load(f)
                    except Exception:
                        meta = {"created": os.path.getctime(self._session_path(session_id, archived=True)), "status": "archived"}
                    archived.append({"session_id": session_id, **meta, "status": "archived"})
            return archived

    def rename_session(self, session_id, new_name):
        with self._lock:
            # Prüfe, ob Session aktiv oder archiviert ist
            meta = self._meta.get(session_id)
            if meta is not None:
                self._meta[session_id]["name"] = new_name
                if session_id in self._sessions:
                    setattr(self._sessions[session_id], "session_name", new_name)
                self._save_meta(session_id)
                self._log(f"Renamed session {session_id} to {new_name}")
            else:
                # Archivierte Session: Metadaten-Datei im Archiv-Ordner anpassen
                meta_path = self._meta_path(session_id, archived=True)
                try:
                    with open(meta_path, "r") as f:
                        meta_arch = json.load(f)
                    meta_arch["name"] = new_name
                    with open(meta_path, "w") as f:
                        json.dump(meta_arch, f)
                    self._log(f"Renamed archived session {session_id} to {new_name}")
                except Exception as e:
                    self._log(f"Failed to rename archived session {session_id}: {e}")

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
            cache.session_name = name or "Session " + session_id[:8]
            self._sessions[session_id] = cache
            self._meta[session_id] = {
                "created": datetime.datetime.now().isoformat(),
                "name": name or "Session " + session_id[:8],
                "status": "active"
            }
            self.save_session(session_id)
            self._save_meta(session_id)
            self._log(f"Imported session {session_id}")
            return session_id

    def cleanup_sessions(self, max_sessions=None, max_age_days=None):
        raise NotImplementedError("Automated cleanup not implemented yet")
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
        cache.session_name = "Session " + session_id[:8]
        session_manager._sessions[session_id] = cache
        session_manager._meta[session_id] = {
            "created": datetime.datetime.now().isoformat(),
            "name": "Session " + session_id[:8],
            "status": "active"
        }
        session_manager.save_session(session_id)
        session_manager._save_meta(session_id)
        session_manager._log(f"Auto-created session cache for {session_id}")
    return cache

# Global singleton for the app
session_manager = SessionCacheManager()
