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
import { Phase6GTestPanel } from './components/Phase6GTestPanel';

export function App() {
  const { user, token, isAuthenticated, loading, error, login, logout } = useAuth();
  const [activeRoomId, setActiveRoomId] = useState<string | null>(null);
  const [showPhase6GTest, setShowPhase6GTest] = useState<boolean>(false);

  // Check URL query parameters for direct room join (e.g. ?room=ABC-123) or ?test=phase6g
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const roomParam = params.get('room');
    if (roomParam) {
      setActiveRoomId(roomParam.toUpperCase());
    }
    if (params.get('test') === 'phase6g') {
      setShowPhase6GTest(true);
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

  const handleOpenPhase6GTest = () => {
    transitionView(() => {
      setShowPhase6GTest(true);
      const url = new URL(window.location.href);
      url.searchParams.set('test', 'phase6g');
      window.history.pushState({}, '', url.toString());
    });
  };

  const handleClosePhase6GTest = () => {
    transitionView(() => {
      setShowPhase6GTest(false);
      const url = new URL(window.location.href);
      url.searchParams.delete('test');
      window.history.pushState({}, '', url.toString());
    });
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

  // Dedicated Phase 6G Neural Face Swap Test UI
  if (showPhase6GTest) {
    return <Phase6GTestPanel onBack={handleClosePhase6GTest} />;
  }

  // Loading Screen
  if (loading) {
    return <LoadingScreen message="Loading DeepTrace..." />;
  }

  // Login
  if (!isAuthenticated || !user || !token) {
    return <LoginPage onLogin={login} error={error} onOpenPhase6GTest={handleOpenPhase6GTest} />;
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
        onOpenPhase6GTest={handleOpenPhase6GTest}
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
