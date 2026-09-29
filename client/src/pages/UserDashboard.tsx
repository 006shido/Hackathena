import React, { useState } from 'react';
import { Shield, Video, Plus, LogOut, ArrowRight, User, Lock, Activity, Users, Radio, Sparkles } from 'lucide-react';
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
    <div className="min-h-screen w-full bg-black text-zinc-100 flex flex-col relative amoled-grid">
      {/* Top Navbar */}
      <nav className="w-full flex items-center justify-between px-4 sm:px-8 py-3.5 border-b border-zinc-800/90 bg-[#050507] shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-orange-500/10 border border-orange-500/30 text-orange-400 shrink-0">
            <Shield className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold tracking-tight text-white leading-tight">DeepTrace</h1>
              <span className="flex items-center gap-1 rounded bg-emerald-500/10 px-1.5 py-0.5 text-[9px] font-mono text-emerald-400 border border-emerald-500/30 uppercase tracking-wider">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> Operational
              </span>
            </div>
            <span className="text-[10px] text-zinc-500 font-mono hidden xs:inline">1-to-1 WebRTC Video Security</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-zinc-950 border border-zinc-800 text-xs font-mono">
            <User className="h-3.5 w-3.5 text-zinc-400" />
            <span className="text-zinc-200 font-medium truncate max-w-[120px]">{user.name}</span>
            <span className="text-[9px] text-emerald-400 font-mono uppercase bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-800/40 hidden xs:inline">
              User
            </span>
          </div>

          <button
            onClick={onLogout}
            title="Log out"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-zinc-950 border border-zinc-800 text-xs font-mono text-zinc-400 hover:text-rose-400 hover:border-rose-500/40 transition-colors cursor-pointer"
          >
            <LogOut className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Logout</span>
          </button>
        </div>
      </nav>

      {/* Main Content - Well proportioned & expansive */}
      <main className="flex-1 max-w-5xl w-full mx-auto p-4 sm:p-8 md:p-10 flex flex-col justify-center">
        {/* Welcome Banner */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-mono mb-3">
            <Lock className="h-3 w-3" />
            <span>AUTHENTICATED CLIENT SESSION</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-white mb-2">
            Welcome, {user.name}
          </h2>
          <p className="text-sm text-zinc-400 max-w-xl mx-auto">
            Direct peer-to-peer WebRTC video calling. Isolated 2-party rooms with DTLS-SRTP encryption and DeepTrace biometric security verification.
          </p>
        </div>

        {/* Action Cards - Proportional, Handcrafted & Wide */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 w-full">
          {/* Card 1: Create Call */}
          <div className="flex flex-col justify-between p-6 sm:p-8 rounded-lg bg-[#070709] border border-zinc-800 hover:border-orange-500/50 transition-all text-left relative overflow-hidden group">
            {/* Top edge subtle highlight */}
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-orange-500/40 to-transparent" />

            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="h-11 w-11 rounded-md bg-orange-500/10 text-orange-400 border border-orange-500/30 flex items-center justify-center">
                  <Plus className="h-5 w-5" />
                </div>
                <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
                  Instant Host
                </span>
              </div>

              <h3 className="text-xl font-bold text-white mb-1.5 group-hover:text-orange-400 transition-colors">
                Create New Call
              </h3>
              <p className="text-xs text-zinc-400 leading-relaxed mb-6 font-mono">
                Generate an encrypted 1-to-1 video room and share the room code with your participant.
              </p>
            </div>

            <button
              onClick={handleCreateCall}
              className="w-full py-3 px-4 rounded-md bg-orange-600 hover:bg-orange-500 text-white text-xs font-mono font-bold uppercase tracking-wider shadow-lg shadow-orange-950/40 border border-orange-500 flex items-center justify-center gap-2 transition-all active:scale-[0.99] cursor-pointer"
            >
              <span>Start Instant Meeting</span>
              <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>

          {/* Card 2: Join Call */}
          <div className="flex flex-col justify-between p-6 sm:p-8 rounded-lg bg-[#070709] border border-zinc-800 hover:border-zinc-700 transition-all text-left relative overflow-hidden">
            {/* Top edge subtle highlight */}
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-zinc-700/60 to-transparent" />

            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="h-11 w-11 rounded-md bg-zinc-900 text-zinc-300 border border-zinc-800 flex items-center justify-center">
                  <Video className="h-5 w-5" />
                </div>
                <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
                  Join Peer
                </span>
              </div>

              <h3 className="text-xl font-bold text-white mb-1.5">
                Join Call
              </h3>
              <p className="text-xs text-zinc-400 leading-relaxed mb-4 font-mono">
                Enter an existing Room ID provided by the host.
              </p>
            </div>

            <form onSubmit={handleJoinCall} className="space-y-3">
              <input
                type="text"
                value={inputRoomId}
                onChange={(e) => setInputRoomId(e.target.value)}
                placeholder="e.g. ABC-123"
                className="w-full px-3.5 py-2.5 rounded-md bg-black border border-zinc-800 text-xs font-mono uppercase tracking-wider text-white placeholder-zinc-600 focus:outline-none focus:border-orange-500/80 focus:ring-1 focus:ring-orange-500/80 transition-colors"
              />
              <button
                type="submit"
                disabled={!inputRoomId.trim()}
                className="w-full py-2.5 px-4 rounded-md bg-zinc-800 hover:bg-zinc-700 text-zinc-100 text-xs font-mono font-semibold uppercase tracking-wider border border-zinc-700 disabled:opacity-40 transition-colors flex items-center justify-center gap-2 cursor-pointer"
              >
                <span>Enter Room</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </form>
          </div>
        </div>

        {/* Security & System Telemetry Bar */}
        <div className="mt-8 p-3.5 rounded-lg bg-[#070709] border border-zinc-800/90 grid grid-cols-2 sm:grid-cols-4 gap-3 text-center text-xs font-mono">
          <div className="flex flex-col items-center justify-center p-2 rounded bg-black/50 border border-zinc-900">
            <span className="text-[10px] text-zinc-500 uppercase">Architecture</span>
            <span className="text-zinc-200 font-semibold mt-0.5">Direct WebRTC</span>
          </div>
          <div className="flex flex-col items-center justify-center p-2 rounded bg-black/50 border border-zinc-900">
            <span className="text-[10px] text-zinc-500 uppercase">Room Limit</span>
            <span className="text-emerald-400 font-semibold mt-0.5">Strict 2 Peers</span>
          </div>
          <div className="flex flex-col items-center justify-center p-2 rounded bg-black/50 border border-zinc-900">
            <span className="text-[10px] text-zinc-500 uppercase">Encryption</span>
            <span className="text-zinc-200 font-semibold mt-0.5">DTLS-SRTP P2P</span>
          </div>
          <div className="flex flex-col items-center justify-center p-2 rounded bg-black/50 border border-zinc-900">
            <span className="text-[10px] text-zinc-500 uppercase">DeepTrace</span>
            <span className="text-orange-400 font-semibold mt-0.5">Active Defense</span>
          </div>
        </div>
      </main>
    </div>
  );
};
