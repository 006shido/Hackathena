import React, { useState, useEffect } from 'react';
import { flushSync } from 'react-dom';
import { useAuth } from './hooks/useAuth';
import { HomePage } from './pages/Home';
import { LoginPage } from './pages/Login';
import { UserDashboard } from './pages/UserDashboard';
import { TesterDashboard } from './pages/TesterDashboard';
import { CallPage } from './pages/Call';
import { mediaService } from './services/media';
import { LoadingScreen } from './components/LoadingScreen';

export function App() {
  const { user, token, isAuthenticated, loading, error, login, register, logout } = useAuth();
  const [activeRoomId, setActiveRoomId] = useState<string | null>(null);
  const [authView, setAuthView] = useState<'home' | 'login'>('home');
  const [authMode, setAuthMode] = useState<'signin' | 'signup'>('signin');

  // Check URL query parameters for direct room join (e.g. ?room=ABC-123) or explicit login/signup
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const roomParam = params.get('room');
    if (roomParam) {
      setActiveRoomId(roomParam.toUpperCase());
      setAuthView('login');
    } else if (params.get('signup') === 'true') {
      setAuthMode('signup');
      setAuthView('login');
    } else if (params.get('login') === 'true') {
      setAuthMode('signin');
      setAuthView('login');
    }
  }, []);

  const transitionView = (callback: () => void, direction: 'forward' | 'back' = 'forward') => {
    document.documentElement.dataset.navDirection = direction;
    const doc = document as unknown as {
      startViewTransition?: (cb: () => void) => { finished?: Promise<void> };
    };

    const execute = () => {
      callback();
      window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
    };

    if (typeof doc.startViewTransition === 'function') {
      try {
        const transition = doc.startViewTransition(() => {
          flushSync(() => {
            execute();
          });
        });
        if (transition && transition.finished) {
          transition.finished.finally(() => {
            delete document.documentElement.dataset.navDirection;
          });
        } else {
          setTimeout(() => {
            delete document.documentElement.dataset.navDirection;
          }, 380);
        }
      } catch {
        delete document.documentElement.dataset.navDirection;
        execute();
      }
    } else {
      execute();
      delete document.documentElement.dataset.navDirection;
    }
  };

  const handleStartCall = (roomId: string) => {
    transitionView(() => {
      setActiveRoomId(roomId);
      // Update URL without page reload
      const url = new URL(window.location.href);
      url.searchParams.set('room', roomId);
      window.history.pushState({}, '', url.toString());
    }, 'forward');
  };

  const handleExitCall = () => {
    mediaService.stopAllMedia();
    transitionView(() => {
      setActiveRoomId(null);
      const url = new URL(window.location.href);
      url.searchParams.delete('room');
      window.history.pushState({}, '', url.toString());
    }, 'back');
  };

  const handleLogout = () => {
    mediaService.stopAllMedia();
    logout();
    transitionView(() => {
      setAuthView('home');
      setAuthMode('signin');
    }, 'back');
  };

  // Loading Screen
  if (loading) {
    return <LoadingScreen message="Loading DeepTrace..." />;
  }

  // First page: Home Landing Page or Login for unauthenticated visitors
  if (!isAuthenticated || !user || !token) {
    if (authView === 'login') {
      return (
        <LoginPage
          onLogin={login}
          onRegister={register}
          initialMode={authMode}
          error={error}
          onBackToHome={() => transitionView(() => setAuthView('home'), 'back')}
        />
      );
    }

    return (
      <HomePage
        onNavigate={(mode = 'signin') =>
          transitionView(() => {
            setAuthMode(mode);
            setAuthView('login');
          }, 'forward')
        }
        onQuickDemo={async (role) => {
          if (role === 'tester') {
            await login('tester', 'tester123');
          } else {
            await login('user', 'user123');
          }
        }}
      />
    );
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
