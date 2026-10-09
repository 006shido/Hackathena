import { lazy,Suspense,useState } from 'react';
import { flushSync } from 'react-dom';
import { LoadingScreen } from './components/LoadingScreen';
import { useAuth } from './hooks/useAuth';
import { HomePage } from './pages/Home';
import { LoginPage } from './pages/Login';
import { TesterDashboard } from './pages/TesterDashboard';
import { UserDashboard } from './pages/UserDashboard';
import { mediaService } from './services/media';

const CallPage = lazy(() => import('./pages/Call').then(module => ({ default: module.CallPage })));
const Phase6GTestPanel = lazy(() => import('./components/Phase6GTestPanel').then(module => ({ default: module.Phase6GTestPanel })));

export function App() {
  return <Suspense fallback={<LoadingScreen message="Loading DeepTrace..." />}><AppContent /></Suspense>;
}

function AppContent() {
  const { user, token, isAuthenticated, loading, error, login, register, logout } = useAuth();
  const [initialParams] = useState(() => new URLSearchParams(window.location.search));
  const [showPhase6GTest, setShowPhase6GTest] = useState(initialParams.get('test') === 'phase6g');
  const [authView, setAuthView] = useState<'home' | 'login'>(
    initialParams.get('room') || initialParams.get('signup') === 'true' || initialParams.get('login') === 'true' ? 'login' : 'home'
  );
  const [authMode, setAuthMode] = useState<'signin' | 'signup'>(
    !initialParams.get('room') && initialParams.get('signup') === 'true' ? 'signup' : 'signin'
  );
  const [demoRole, setDemoRole] = useState<'user' | 'tester' | undefined>();
  const [activeRoomId, setActiveRoomId] = useState<string | null>(initialParams.get('room')?.toUpperCase() || null);

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
      setActiveRoomId(null);
      setAuthView('home');
      setAuthMode('signin');
    }, 'back');
  };

  // Dedicated Phase 6G Neural Face Swap Test UI
  if (showPhase6GTest) {
    return <Phase6GTestPanel onBack={handleClosePhase6GTest} />;
  }

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
          initialDemoRole={demoRole}
          error={error}
          onBackToHome={() => transitionView(() => setAuthView('home'), 'back')}
          onOpenPhase6GTest={handleOpenPhase6GTest}
        />
      );
    }

    return (
      <HomePage
        onNavigate={(mode = 'signin') =>
          transitionView(() => {
            setDemoRole(undefined);
            setAuthMode(mode);
            setAuthView('login');
          }, 'forward')
        }
        onQuickDemo={async (role) => {
          const passwordRequired = import.meta.env.VITE_DEMO_PASSWORD_REQUIRED === 'true';
          if (passwordRequired && !['localhost', '127.0.0.1'].includes(window.location.hostname)) {
            transitionView(() => { setDemoRole(role); setAuthMode('signin'); setAuthView('login'); });
            return;
          }
          try {
            await login(role, passwordRequired ? '__LOCAL_DEMO__' : role === 'tester' ? 'tester123' : 'user123');
          } catch {
            transitionView(() => { setAuthMode('signin'); setAuthView('login'); });
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
