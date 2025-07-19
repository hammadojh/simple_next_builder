"""
Unified Edit Coordinator

Prevents conflicts between multiple edit systems by:
1. Ensuring only one edit strategy runs at a time per app
2. Coordinating between DiffApplier, CodeBuilder, IntentBasedEditor
3. Managing session-wide edit locks
4. Providing unified strategy selection and fallback
5. Maintaining consistency across all edit operations
"""

import time
import threading
from pathlib import Path
from typing import Dict, Set, Optional, List, Callable, Any
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import os


class EditStrategy(Enum):
    """Available edit strategies in order of preference."""
    ATOMIC_DIFF = "atomic_diff"        # Enhanced diff with atomic transactions
    INTENT_BASED = "intent_based"      # Structured JSON intents
    LINE_BASED = "line_based"          # Legacy line-based editing
    MANUAL_FIX = "manual_fix"          # Manual syntax fixes


@dataclass
class EditSession:
    """Represents an active edit session."""
    session_id: str
    app_path: str
    strategy: EditStrategy
    start_time: float
    thread_id: int
    description: str
    status: str = "active"  # active, completed, failed, aborted


@dataclass 
class EditRequest:
    """Represents a request to edit an application."""
    description: str
    app_path: str
    preferred_strategy: Optional[EditStrategy] = None
    priority: int = 1  # 1=high, 2=medium, 3=low
    timeout_seconds: int = 300  # 5 minutes default


class EditCoordinator:
    """
    Unified coordinator for all edit operations.
    
    Guarantees:
    - Only one edit operation per app at a time
    - No conflicts between different edit strategies
    - Proper cleanup on failures
    - Deadlock prevention
    - Resource isolation between apps
    """
    
    def __init__(self):
        self._active_sessions: Dict[str, EditSession] = {}
        self._app_locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()
        self._strategy_handlers: Dict[EditStrategy, Callable] = {}
        self._session_counter = 0
        
        # Strategy preference order (most reliable first)
        self._strategy_order = [
            EditStrategy.ATOMIC_DIFF,
            EditStrategy.INTENT_BASED, 
            EditStrategy.LINE_BASED,
            EditStrategy.MANUAL_FIX
        ]
        
        print("🎛️ Edit Coordinator initialized")
        print("🔒 Multi-strategy coordination enabled")
    
    def register_strategy_handler(self, strategy: EditStrategy, handler: Callable):
        """Register a handler function for an edit strategy."""
        self._strategy_handlers[strategy] = handler
        print(f"📝 Registered handler for {strategy.value}")
    
    @contextmanager
    def coordinate_edit(self, request: EditRequest):
        """
        Context manager for coordinated edit operations.
        
        Usage:
            with coordinator.coordinate_edit(request) as session:
                # perform edit operations
                # automatic cleanup on exit
        """
        session = None
        try:
            # Acquire session
            session = self._acquire_edit_session(request)
            if not session:
                raise RuntimeError(f"Failed to acquire edit session for {request.app_path}")
            
            print(f"🔓 Edit session acquired: {session.session_id}")
            yield session
            
            # Mark session as completed
            session.status = "completed"
            print(f"✅ Edit session completed: {session.session_id}")
            
        except Exception as e:
            # Mark session as failed
            if session:
                session.status = "failed"
            print(f"❌ Edit session failed: {e}")
            raise
            
        finally:
            # Always clean up
            if session:
                self._release_edit_session(session)
    
    def execute_coordinated_edit(self, request: EditRequest) -> bool:
        """
        Execute an edit request with full coordination and fallback strategies.
        
        Args:
            request: The edit request to execute
            
        Returns:
            bool: True if edit succeeded, False otherwise
        """
        print(f"🎯 Coordinating edit: {request.description}")
        print(f"📁 Target: {request.app_path}")
        
        try:
            with self.coordinate_edit(request) as session:
                return self._execute_edit_with_strategies(session, request)
                
        except Exception as e:
            print(f"❌ Coordinated edit failed: {e}")
            return False
    
    def _execute_edit_with_strategies(self, session: EditSession, request: EditRequest) -> bool:
        """Execute edit with automatic strategy fallback."""
        # Determine strategy order based on preference
        strategies = self._get_strategy_order(request.preferred_strategy)
        
        print(f"🔄 Trying {len(strategies)} strategies in order: {[s.value for s in strategies]}")
        
        for i, strategy in enumerate(strategies, 1):
            if strategy not in self._strategy_handlers:
                print(f"⚠️ Strategy {strategy.value} not available - skipping")
                continue
            
            print(f"\n🔧 Strategy {i}/{len(strategies)}: {strategy.value}")
            session.strategy = strategy
            
            try:
                # Execute the strategy
                handler = self._strategy_handlers[strategy]
                success = handler(request)
                
                if success:
                    print(f"✅ Edit successful with {strategy.value}")
                    return True
                else:
                    print(f"❌ Strategy {strategy.value} failed - trying next")
                    
            except Exception as e:
                print(f"❌ Strategy {strategy.value} threw exception: {e}")
                continue
        
        print("💥 All strategies exhausted - edit failed")
        return False
    
    def _acquire_edit_session(self, request: EditRequest) -> Optional[EditSession]:
        """Acquire an exclusive edit session for an app."""
        app_key = self._normalize_app_path(request.app_path)
        
        with self._global_lock:
            # Check if app is already being edited
            if app_key in self._active_sessions:
                existing = self._active_sessions[app_key]
                print(f"⏳ App {app_key} is already being edited by session {existing.session_id}")
                return None
            
            # Create app lock if it doesn't exist
            if app_key not in self._app_locks:
                self._app_locks[app_key] = threading.RLock()
            
            # Try to acquire app lock
            app_lock = self._app_locks[app_key]
            if not app_lock.acquire(blocking=False):
                print(f"🔒 Could not acquire lock for app {app_key}")
                return None
            
            # Create edit session
            self._session_counter += 1
            session = EditSession(
                session_id=f"edit_{self._session_counter}_{int(time.time())}",
                app_path=request.app_path,
                strategy=request.preferred_strategy or EditStrategy.ATOMIC_DIFF,
                start_time=time.time(),
                thread_id=threading.get_ident(),
                description=request.description
            )
            
            # Register active session
            self._active_sessions[app_key] = session
            
            return session
    
    def _release_edit_session(self, session: EditSession):
        """Release an edit session and its resources."""
        app_key = self._normalize_app_path(session.app_path)
        
        with self._global_lock:
            # Remove from active sessions
            if app_key in self._active_sessions:
                del self._active_sessions[app_key]
            
            # Release app lock
            if app_key in self._app_locks:
                try:
                    self._app_locks[app_key].release()
                except:
                    pass  # Lock might not be held
        
        duration = time.time() - session.start_time
        print(f"🔓 Released edit session {session.session_id} (duration: {duration:.2f}s)")
    
    def _get_strategy_order(self, preferred: Optional[EditStrategy]) -> List[EditStrategy]:
        """Get the order of strategies to try, with preferred strategy first."""
        if preferred and preferred in self._strategy_order:
            # Put preferred strategy first, then others in default order
            others = [s for s in self._strategy_order if s != preferred]
            return [preferred] + others
        else:
            return self._strategy_order.copy()
    
    def _normalize_app_path(self, app_path: str) -> str:
        """Normalize app path for consistent keying."""
        return str(Path(app_path).resolve())
    
    def get_active_sessions(self) -> Dict[str, EditSession]:
        """Get currently active edit sessions."""
        with self._global_lock:
            return self._active_sessions.copy()
    
    def is_app_being_edited(self, app_path: str) -> bool:
        """Check if an app is currently being edited."""
        app_key = self._normalize_app_path(app_path)
        with self._global_lock:
            return app_key in self._active_sessions
    
    def abort_session(self, session_id: str) -> bool:
        """Abort an active edit session."""
        with self._global_lock:
            for app_key, session in self._active_sessions.items():
                if session.session_id == session_id:
                    session.status = "aborted"
                    self._release_edit_session(session)
                    print(f"🛑 Aborted edit session: {session_id}")
                    return True
        
        print(f"❌ Session not found: {session_id}")
        return False
    
    def cleanup_stale_sessions(self, max_age_seconds: int = 3600):
        """Clean up sessions that have been running too long."""
        current_time = time.time()
        stale_sessions = []
        
        with self._global_lock:
            for app_key, session in self._active_sessions.items():
                if current_time - session.start_time > max_age_seconds:
                    stale_sessions.append(session)
        
        for session in stale_sessions:
            print(f"🧹 Cleaning up stale session: {session.session_id}")
            self._release_edit_session(session)
        
        return len(stale_sessions)
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        with self._global_lock:
            return {
                "active_sessions": len(self._active_sessions),
                "apps_with_locks": len(self._app_locks),
                "registered_strategies": len(self._strategy_handlers),
                "available_strategies": [s.value for s in self._strategy_handlers.keys()],
                "sessions": {
                    app: {
                        "session_id": session.session_id,
                        "strategy": session.strategy.value,
                        "duration": time.time() - session.start_time,
                        "status": session.status
                    }
                    for app, session in self._active_sessions.items()
                }
            }


# Global coordinator instance
_global_coordinator: Optional[EditCoordinator] = None


def get_edit_coordinator() -> EditCoordinator:
    """Get the global edit coordinator instance."""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = EditCoordinator()
    return _global_coordinator


def reset_edit_coordinator():
    """Reset the global coordinator (for testing)."""
    global _global_coordinator
    _global_coordinator = None 