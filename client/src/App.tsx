import React, { useState, useEffect } from 'react';
import { useAuth } from './hooks/useAuth';
import { LoginPage } from './pages/Login';
import { UserDashboard } from './pages/UserDashboard';
import { TesterDashboard } from './pages/TesterDashboard';
import { CallPage } from './pages/Call';
import { Shield } from 'lucide-react';

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

  const handleStartCall = (roomId: string) => {
    setActiveRoomId(roomId);
    // Update URL without page reload
    const url = new URL(window.location.href);
    url.searchParams.set('room', roomId);
    window.history.pushState({}, '', url.toString());
  };

  const handleExitCall = () => {
    setActiveRoomId(null);
    const url = new URL(window.location.href);
    url.searchParams.delete('room');
    window.history.pushState({}, '', url.toString());
  };

  // Loading Splash Screen
  if (loading) {
    return (
      <div className="min-h-screen w-full bg-black flex flex-col items-center justify-center amoled-grid">
        <div className="h-14 w-14 rounded-lg bg-orange-500/10 border border-orange-500/40 flex items-center justify-center shadow-lg shadow-orange-950/40 mb-4 animate-pulse">
          <Shield className="h-7 w-7 text-orange-400" />
        </div>
        <span className="text-xs font-mono tracking-widest text-zinc-400 uppercase">
          Initializing DeepTrace Security Core...
        </span>
      </div>
    );
  }

  // Unauthenticated: Show Login
  if (!isAuthenticated || !user || !token) {
    return <LoginPage onLogin={login} error={error} />;
  }

  // Active Call Screen
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

  // Authenticated Dashboards based on role
  if (user.role === 'tester') {
    return (
      <TesterDashboard
        user={user}
        onLogout={logout}
        onStartCall={handleStartCall}
      />
    );
  }

  return (
    <UserDashboard
      user={user}
      onLogout={logout}
      onStartCall={handleStartCall}
    />
  );
}

export default App;
