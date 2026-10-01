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

  // Loading Screen
  if (loading) {
    return (
      <div className="min-h-screen w-full bg-[#202124] flex flex-col items-center justify-center">
        <div className="h-20 w-20 rounded-2xl bg-white p-3 flex items-center justify-center mb-4 animate-pulse shadow-2xl shadow-black/50 border border-white/20">
          <img src="/logo.svg" alt="DeepTrace Logo" className="h-full w-full object-contain" />
        </div>
        <span className="text-sm text-[#9aa0a6]">
          Loading...
        </span>
      </div>
    );
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
