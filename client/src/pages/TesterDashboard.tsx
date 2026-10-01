import React, { useState } from 'react';
import { Shield, ShieldAlert, Video, Plus, LogOut, ArrowRight, AlertTriangle, Flame } from 'lucide-react';
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
    <div className="min-h-screen w-full bg-[#202124] text-[#e8eaed] flex flex-col">
      {/* Navbar */}
      <nav className="w-full flex items-center justify-between gap-3 px-3.5 sm:px-6 md:px-8 py-3 border-b border-[#3c4043] bg-[#202124] shrink-0 select-none">
        <div className="flex items-center gap-2.5 sm:gap-3 min-w-0">
          <img src="/logo.svg" alt="DeepTrace Logo" className="h-8 w-8 sm:h-9 sm:w-9 shrink-0 rounded-xl object-contain shadow-sm" />
          <div className="flex items-center gap-1.5 sm:gap-2 min-w-0">
            <h1 className="text-sm sm:text-base font-medium text-[#e8eaed] tracking-tight">DeepTrace</h1>
            <span className="text-[10px] sm:text-[11px] text-[#8ab4f8] bg-[#1a73e8]/20 px-1.5 sm:px-2 py-0.5 rounded-full font-medium shrink-0">
              Tester
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2 sm:gap-2.5 shrink-0">
          <div className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-full bg-[#303134] border border-[#3c4043] text-xs sm:text-sm">
            <Shield className="h-3.5 w-3.5 text-[#8ab4f8] shrink-0" />
            <span className="text-[#e8eaed] font-medium truncate max-w-[85px] sm:max-w-[140px] md:max-w-[180px]">
              {user.name}
            </span>
          </div>

          <button
            onClick={onLogout}
            title="Log out"
            className="flex items-center justify-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-full bg-[#303134] hover:bg-[#3c4043] border border-[#3c4043] text-xs sm:text-sm text-[#9aa0a6] hover:text-[#ea4335] transition-colors shrink-0 cursor-pointer"
          >
            <LogOut className="h-3.5 w-3.5 shrink-0" />
            <span className="hidden sm:inline">Logout</span>
          </button>
        </div>
      </nav>

      {/* Main */}
      <main className="flex-1 max-w-3xl w-full mx-auto p-4 sm:p-8 flex flex-col justify-center">
        <div className="text-center mb-6">
          <h2 className="text-2xl sm:text-3xl font-medium text-[#e8eaed] mb-2">
            Welcome, {user.name}
          </h2>
          <p className="text-sm text-[#9aa0a6] max-w-md mx-auto">
            Test deepfake attacks with real-time face overlay & voice synthesis tools.
          </p>
        </div>

        <div className="w-full mb-5 p-3.5 rounded-2xl bg-[#28292c] border border-[#3c4043] flex items-center gap-3">
          <div className="h-8 w-8 rounded-full bg-[#8ab4f8]/15 flex items-center justify-center shrink-0 text-[#8ab4f8]">
            <AlertTriangle className="h-4 w-4" />
          </div>
          <div className="flex-1">
            <span className="text-xs font-medium text-[#8ab4f8] block">Tester Access</span>
            <p className="text-xs text-[#9aa0a6] mt-0.5">
              You have access to the in-call Attack Simulator panel for security testing.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
          <div className="flex flex-col justify-between p-6 rounded-2xl bg-[#28292c] border border-[#3c4043] hover:border-[#5f6368] transition-colors">
            <div>
              <div className="h-11 w-11 rounded-full bg-[#ea4335]/15 text-[#ea4335] flex items-center justify-center mb-4">
                <Flame className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-medium text-[#e8eaed] mb-1">Test Room</h3>
              <p className="text-sm text-[#9aa0a6] mb-6">
                Launch a testing room with attack simulator controls.
              </p>
            </div>

            <button
              onClick={handleCreateTestCall}
              className="w-full py-3 px-4 rounded-full bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium flex items-center justify-center gap-2 transition-colors active:scale-[0.99] cursor-pointer"
            >
              Launch Test Room
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>

          <div className="flex flex-col justify-between p-6 rounded-2xl bg-[#28292c] border border-[#3c4043] transition-colors">
            <div>
              <div className="h-11 w-11 rounded-full bg-[#303134] text-[#9aa0a6] flex items-center justify-center mb-4">
                <Video className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-medium text-[#e8eaed] mb-1">Join Meeting</h3>
              <p className="text-sm text-[#9aa0a6] mb-4">
                Enter a meeting code to test against a participant.
              </p>
            </div>

            <form onSubmit={handleJoinCall} className="space-y-3">
              <input
                type="text"
                value={inputRoomId}
                onChange={(e) => setInputRoomId(e.target.value)}
                placeholder="e.g. ABC-123"
                className="w-full px-3.5 py-2.5 rounded-xl bg-[#202124] border border-[#3c4043] text-sm uppercase tracking-wider text-[#e8eaed] placeholder-[#80868b] focus:outline-none focus:border-[#8ab4f8] transition-colors"
              />
              <button
                type="submit"
                disabled={!inputRoomId.trim()}
                className="w-full py-2.5 px-4 rounded-full bg-[#3c4043] hover:bg-[#4a4d51] text-[#e8eaed] text-sm font-medium disabled:opacity-40 transition-colors flex items-center justify-center gap-2 cursor-pointer"
              >
                Join Room
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </form>
          </div>
        </div>

        <div className="mt-6 p-3 rounded-2xl bg-[#28292c] border border-[#3c4043] grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
          <div className="p-2 rounded-xl bg-[#202124]">
            <span className="text-[#9aa0a6] block text-[10px]">Face Swap</span>
            <span className="text-[#8ab4f8] font-medium">Canvas 30FPS</span>
          </div>
          <div className="p-2 rounded-xl bg-[#202124]">
            <span className="text-[#9aa0a6] block text-[10px]">Voice</span>
            <span className="text-[#8ab4f8] font-medium">WebAudio</span>
          </div>
          <div className="p-2 rounded-xl bg-[#202124]">
            <span className="text-[#9aa0a6] block text-[10px]">Stream</span>
            <span className="text-[#e8eaed] font-medium">replaceTrack()</span>
          </div>
          <div className="p-2 rounded-xl bg-[#202124]">
            <span className="text-[#9aa0a6] block text-[10px]">Auth</span>
            <span className="text-[#34a853] font-medium">Verified</span>
          </div>
        </div>
      </main>
    </div>
  );
};
