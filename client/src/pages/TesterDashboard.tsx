import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  Video,
  LogOut,
  ArrowRight,
  Shield,
  User as UserIcon,
} from 'lucide-react';
import { User as UserType } from '../types/auth';

interface TesterDashboardProps {
  user: UserType;
  onLogout: () => void;
  onStartCall: (roomId: string) => void;
}

export const TesterDashboard: React.FC<TesterDashboardProps> = ({
  user,
  onLogout,
  onStartCall,
}) => {
  const [inputRoomId, setInputRoomId] = useState('');

  const [greeting, setGreeting] = useState('Welcome');
  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting('Good morning');
    else if (hour < 18) setGreeting('Good afternoon');
    else setGreeting('Good evening');
  }, []);

  const generateRoomId = () => {
    const letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ';
    const l1 = letters[Math.floor(Math.random() * letters.length)];
    const l2 = letters[Math.floor(Math.random() * letters.length)];
    const l3 = letters[Math.floor(Math.random() * letters.length)];
    const num = Math.floor(100 + Math.random() * 900);
    return `${l1}${l2}${l3}-${num}`;
  };

  const handleCreateTestCall = () => {
    const newRoomId = generateRoomId();
    onStartCall(newRoomId);
  };

  const handleJoinCall = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputRoomId.trim()) return;
    onStartCall(inputRoomId.trim().toUpperCase());
  };

  return (
    <div className="min-h-screen w-full bg-[#f8fafc] text-slate-900 flex flex-col font-sans animate-fade-in">
      {/* Top Bar */}
      <header className="w-full border-b border-slate-200/80 bg-white sticky top-0 z-30 animate-slide-down">
        <div className="max-w-5xl mx-auto px-4 sm:px-8 h-14 sm:h-16 flex items-center justify-between gap-3">
          {/* Left Side: Brand Identity Only */}
          <div className="flex items-center gap-2 sm:gap-2.5 shrink-0">
            <div className="h-8 w-8 sm:h-9 sm:w-9 shrink-0 flex items-center justify-center">
              <img src="/logo.svg" alt="DeepTrace Logo" className="h-full w-full" />
            </div>
            <span className="font-bold text-base sm:text-xl tracking-tight text-slate-900 leading-none">
              DeepTrace
            </span>
          </div>

          {/* Right Side: User Profile + Logout */}
          <div className="flex items-center gap-2 sm:gap-3 ml-auto shrink-0">
            {/* User Profile Tag */}
            <div className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-full bg-slate-100/90 border border-slate-200/60 text-xs sm:text-sm font-medium text-slate-700 shadow-2xs">
              <Shield className="h-3.5 w-3.5 text-blue-600 shrink-0 stroke-[2]" />
              <span className="truncate max-w-[100px] xs:max-w-[140px] sm:max-w-[180px] leading-none">
                {user.name}
              </span>
            </div>

            {/* Logout Action */}
            <button
              onClick={onLogout}
              title="Logout"
              className="p-2 rounded-full text-slate-400 hover:text-red-600 hover:bg-red-50 transition-colors cursor-pointer shrink-0"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-3xl w-full mx-auto px-4 sm:px-6 py-12 sm:py-20 flex flex-col justify-center animate-fade-in-up">
        {/* Simple Heading */}
        <div className="text-center mb-10">
          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-slate-900 mb-2">
            {greeting}, {user.name.split(' ')[0]}
          </h1>
          <p className="text-sm sm:text-base text-slate-500">
            Security testing & deepfake simulation mode.
          </p>
        </div>

        {/* Action Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-5 w-full">
          {/* Card 1: Launch Test Room */}
          <div className="flex flex-col justify-between p-6 rounded-2xl bg-white border border-slate-200/80 shadow-2xs hover:border-slate-300 hover:shadow-xs transition-all">
            <div>
              <div className="h-11 w-11 rounded-2xl bg-gradient-to-br from-rose-50 via-rose-100/60 to-orange-50 border border-rose-200/80 shadow-xs flex items-center justify-center mb-4 text-rose-600">
                <ShieldAlert className="h-5 w-5 text-rose-600 stroke-[2]" />
              </div>
              <h2 className="text-base sm:text-lg font-bold text-slate-900 mb-1">
                Launch Test Room
              </h2>
              <p className="text-xs text-slate-500 mb-6">
                Start a test room with the Attack Simulator panel.
              </p>
            </div>

            <button
              onClick={handleCreateTestCall}
              className="w-full py-2.5 px-4 rounded-xl bg-blue-600 hover:bg-blue-700 active:scale-[0.99] text-white text-sm font-semibold flex items-center justify-center gap-2 shadow-2xs transition-all cursor-pointer"
            >
              <span>Launch Test</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>

          {/* Card 2: Join Meeting */}
          <div className="flex flex-col justify-between p-6 rounded-2xl bg-white border border-slate-200/80 shadow-2xs hover:border-slate-300 hover:shadow-xs transition-all">
            <div>
              <div className="h-11 w-11 rounded-2xl bg-gradient-to-br from-slate-50 to-slate-100/90 border border-slate-200/80 shadow-xs flex items-center justify-center mb-4 text-slate-700">
                <Video className="h-5 w-5 text-slate-700 stroke-[2]" />
              </div>
              <h2 className="text-base sm:text-lg font-bold text-slate-900 mb-1">
                Join Target Room
              </h2>
              <p className="text-xs text-slate-500 mb-6">
                Enter code to connect and test against a peer.
              </p>
            </div>

            <form onSubmit={handleJoinCall} className="space-y-2.5">
              <input
                type="text"
                value={inputRoomId}
                onChange={(e) => setInputRoomId(e.target.value)}
                placeholder="Meeting code (e.g. ABC-123)"
                className="w-full px-3.5 py-2 rounded-xl bg-slate-50 border border-slate-200 text-xs sm:text-sm uppercase tracking-wider text-slate-900 placeholder:text-slate-400 focus:bg-white focus:outline-none focus:border-blue-500 transition-all text-center font-mono font-medium"
              />
              <button
                type="submit"
                disabled={!inputRoomId.trim()}
                className="w-full py-2 px-4 rounded-xl bg-slate-800 hover:bg-slate-900 disabled:opacity-40 text-white text-xs sm:text-sm font-semibold transition-all flex items-center justify-center gap-1.5 cursor-pointer disabled:cursor-not-allowed"
              >
                <span>Join</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </form>
          </div>
        </div>

        {/* Minimal Footer */}
        <div className="mt-12 text-center text-xs text-slate-400 font-medium">
          Canvas 30FPS • WebAudio DSP • replaceTrack() Stream Injection
        </div>
      </main>
    </div>
  );
};
