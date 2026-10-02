import React, { useState, useEffect } from 'react';
import { flushSync } from 'react-dom';
import { useAuth } from './hooks/useAuth';
import { LoginPage } from './pages/Login';
import { UserDashboard } from './pages/UserDashboard';
import { TesterDashboard } from './pages/TesterDashboard';
import { CallPage } from './pages/Call';
import { mediaService } from './services/media';
import { Shield } from 'lucide-react';

import { LoadingScreen } from './components/LoadingScreen';

export function App() {
  const { user, token, isAuthenticated, loading, error, login, logout } = useAuth();
  const [activeRoomId, setActiveRoomId] = useState<string | null>(null);

  // Check URL query parameters for direct room join (e.g. ?room=ABC-123)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const roomParam = params.get('room');
    if (roomParam) {
      setActiveRoomId(roomParam.toUpperCase());
    }
  }, []);

  const transitionView = (callback: () => void) => {
    const doc = document as unknown as { startViewTransition?: (cb: () => void) => void };
    if (typeof doc.startViewTransition === 'function') {
      doc.startViewTransition(() => {
        flushSync(() => {
          callback();
        });
      });
    } else {
      callback();
    }
  };

  const handleStartCall = (roomId: string) => {
    transitionView(() => {
      setActiveRoomId(roomId);
      // Update URL without page reload
      const url = new URL(window.location.href);
      url.searchParams.set('room', roomId);
      window.history.pushState({}, '', url.toString());
    });
  };

  const handleExitCall = () => {
    mediaService.stopAllMedia();
    transitionView(() => {
      setActiveRoomId(null);
      const url = new URL(window.location.href);
      url.searchParams.delete('room');
      window.history.pushState({}, '', url.toString());
    });
  };

  const handleLogout = () => {
    mediaService.stopAllMedia();
    logout();
  };

  // Loading Screen
  if (loading) {
    return <LoadingScreen message="Loading DeepTrace..." />;
  }

  // Login
  if (!isAuthenticated || !user || !token) {
    return <LoginPage onLogin={login} error={error} />;
  }

  // Active Call
  if (activeRoomId) {
    return (
      <CallPage
        roomId={activeRoomId}
        user={user}
        token={token}
        onExit={handleExitCall}
      />
    );
  }

  // Dashboard
  if (user.role === 'tester') {
    return (
      <TesterDashboard
        user={user}
        onLogout={handleLogout}
        onStartCall={handleStartCall}
      />
    );
  }

  return (
    <UserDashboard
      user={user}
      onLogout={handleLogout}
      onStartCall={handleStartCall}
    />
  );
}


export default App;
