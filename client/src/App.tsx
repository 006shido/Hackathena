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
      <div className="min-h-screen w-full bg-[#08090d] flex flex-col items-center justify-center cyber-grid">
        <div className="h-16 w-16 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-2xl shadow-cyan-950/60 mb-4 animate-pulse">
          <Shield className="h-8 w-8 text-white" />
        </div>
        <span className="text-sm font-mono tracking-widest text-cyan-400 uppercase">
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
