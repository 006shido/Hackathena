import React, { useState } from 'react';
import { Shield, Video, Plus, LogOut, ArrowRight, User } from 'lucide-react';
import { User as UserType } from '../types/auth';

interface UserDashboardProps {
  user: UserType;
  onLogout: () => void;
  onStartCall: (roomId: string) => void;
}

export const UserDashboard: React.FC<UserDashboardProps> = ({
  user,
  onLogout,
  onStartCall,
}) => {
  const [inputRoomId, setInputRoomId] = useState('');
  const [showJoinModal, setShowJoinModal] = useState(false);

  // Generate short readable room ID like ABC-123 or SEC-789
  const generateRoomId = () => {
    const letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ';
    const l1 = letters[Math.floor(Math.random() * letters.length)];
    const l2 = letters[Math.floor(Math.random() * letters.length)];
    const l3 = letters[Math.floor(Math.random() * letters.length)];
    const num = Math.floor(100 + Math.random() * 900);
    return `${l1}${l2}${l3}-${num}`;
  };

  const handleCreateCall = () => {
    const newRoomId = generateRoomId();
    onStartCall(newRoomId);
  };

  const handleJoinCall = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputRoomId.trim()) return;
    onStartCall(inputRoomId.trim().toUpperCase());
  };

  return (
    <div className="min-h-screen w-full bg-[#08090d] text-slate-100 flex flex-col relative cyber-grid">
      {/* Top Navbar */}
      <nav className="w-full flex items-center justify-between px-3.5 sm:px-6 py-2.5 sm:py-4 border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-md">
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 shadow-md shadow-cyan-900/40 shrink-0">
            <Shield className="h-4 w-4 sm:h-5 sm:w-5 text-white" />
          </div>
          <div>
            <h1 className="text-sm sm:text-base font-bold tracking-tight text-white leading-tight">DeepTrace</h1>
            <span className="text-[10px] sm:text-[11px] text-cyan-400 font-mono hidden xs:inline">Secure Video Communication</span>
          </div>
        </div>

        <div className="flex items-center gap-2 sm:gap-4">
          <div className="flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-[11px] sm:text-xs font-mono">
            <User className="h-3 w-3 sm:h-3.5 sm:w-3.5 text-cyan-400" />
            <span className="text-slate-300 font-medium truncate max-w-[80px] sm:max-w-none">{user.name}</span>
            <span className="text-[9px] sm:text-[10px] text-slate-500 uppercase bg-slate-800 px-1 sm:px-1.5 py-0.5 rounded hidden xs:inline">User</span>
          </div>

          <button
            onClick={onLogout}
            title="Log out"
            className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-400 hover:text-rose-400 hover:border-rose-500/40 transition-colors"
          >
            <LogOut className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Logout</span>
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 max-w-4xl w-full mx-auto p-4 sm:p-12 flex flex-col justify-center">
        <div className="text-center mb-10">
          <div className="inline-block px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-mono mb-3">
            VERIFIED SESSION
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-white mb-2">
            Welcome, {user.name}
          </h2>
          <p className="text-sm text-slate-400 max-w-md mx-auto">
            Connect securely with peer-to-peer WebRTC video calling protected by DeepTrace architecture.
          </p>
        </div>

        {/* Action Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 max-w-2xl mx-auto w-full">
          {/* Create Call */}
          <button
            onClick={handleCreateCall}
            className="group flex flex-col items-start p-7 rounded-3xl bg-slate-900/90 border border-slate-800 hover:border-cyan-500/60 shadow-xl hover:shadow-cyan-950/40 transition-all text-left relative overflow-hidden active:scale-[0.98]"
          >
            <div className="h-12 w-12 rounded-2xl bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <Plus className="h-6 w-6" />
            </div>
            <h3 className="text-lg font-bold text-white mb-1 group-hover:text-cyan-300 transition-colors">
              Create New Call
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Generate a secure 1-to-1 video room and invite a participant.
            </p>
            <div className="mt-6 flex items-center gap-1.5 text-xs font-mono font-semibold text-cyan-400">
              <span>Start Instant Meeting</span>
              <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </button>

          {/* Join Call */}
          <div className="flex flex-col justify-between p-7 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl">
            <div>
              <div className="h-12 w-12 rounded-2xl bg-blue-500/20 text-blue-400 border border-blue-500/40 flex items-center justify-center mb-4">
                <Video className="h-6 w-6" />
              </div>
              <h3 className="text-lg font-bold text-white mb-1">
                Join Call
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed mb-4">
                Enter an existing Room ID provided by the host.
              </p>
            </div>

            <form onSubmit={handleJoinCall} className="space-y-3">
              <input
                type="text"
                value={inputRoomId}
                onChange={(e) => setInputRoomId(e.target.value)}
                placeholder="e.g. ABC-123"
                className="w-full px-3.5 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono uppercase text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500"
              />
              <button
                type="submit"
                disabled={!inputRoomId.trim()}
                className="w-full py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono font-semibold uppercase tracking-wider disabled:opacity-40 transition-colors flex items-center justify-center gap-2"
              >
                <span>Join Room</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </form>
          </div>
        </div>

        {/* Security architecture footer note */}
        <div className="mt-12 text-center text-xs font-mono text-slate-500">
          Peer-to-Peer Encryption • Direct WebRTC Connection • 2 Participants Max
        </div>
      </main>
    </div>
  );
};
