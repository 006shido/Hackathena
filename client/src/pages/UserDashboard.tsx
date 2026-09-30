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
    <div className="min-h-screen w-full bg-[#202124] text-[#e8eaed] flex flex-col">
      {/* Navbar */}
      <nav className="w-full flex items-center justify-between px-4 sm:px-8 py-3.5 border-b border-[#3c4043] bg-[#202124] shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#1a73e8]/15 text-[#8ab4f8]">
            <Shield className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-base font-medium text-[#e8eaed]">DeepTrace</h1>
            <span className="text-xs text-[#9aa0a6] hidden xs:inline">Secure Video Calls</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#303134] border border-[#3c4043] text-sm">
            <User className="h-3.5 w-3.5 text-[#9aa0a6]" />
            <span className="text-[#e8eaed] truncate max-w-[120px]">{user.name}</span>
          </div>

          <button
            onClick={onLogout}
            title="Log out"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#303134] border border-[#3c4043] text-sm text-[#9aa0a6] hover:text-[#ea4335] transition-colors cursor-pointer"
          >
            <LogOut className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Logout</span>
          </button>
        </div>
      </nav>

      {/* Main */}
      <main className="flex-1 max-w-3xl w-full mx-auto p-4 sm:p-8 flex flex-col justify-center">
        <div className="text-center mb-8">
          <h2 className="text-2xl sm:text-3xl font-medium text-[#e8eaed] mb-2">
            Welcome, {user.name}
          </h2>
          <p className="text-sm text-[#9aa0a6] max-w-md mx-auto">
            Start a new video call or join an existing meeting.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
          {/* Create Call */}
          <div className="flex flex-col justify-between p-6 rounded-2xl bg-[#28292c] border border-[#3c4043] hover:border-[#5f6368] transition-colors">
            <div>
              <div className="h-11 w-11 rounded-full bg-[#1a73e8]/15 text-[#8ab4f8] flex items-center justify-center mb-4">
                <Plus className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-medium text-[#e8eaed] mb-1">New Meeting</h3>
              <p className="text-sm text-[#9aa0a6] mb-6">
                Create an encrypted 1-to-1 video room.
              </p>
            </div>

            <button
              onClick={handleCreateCall}
              className="w-full py-3 px-4 rounded-full bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium flex items-center justify-center gap-2 transition-colors active:scale-[0.99] cursor-pointer"
            >
              Start Meeting
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>

          {/* Join Call */}
          <div className="flex flex-col justify-between p-6 rounded-2xl bg-[#28292c] border border-[#3c4043] transition-colors">
            <div>
              <div className="h-11 w-11 rounded-full bg-[#303134] text-[#9aa0a6] flex items-center justify-center mb-4">
                <Video className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-medium text-[#e8eaed] mb-1">Join Meeting</h3>
              <p className="text-sm text-[#9aa0a6] mb-4">
                Enter a meeting code to connect with your peer.
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
                Join
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </form>
          </div>
        </div>

        {/* Info */}
        <div className="mt-6 p-3 rounded-2xl bg-[#28292c] border border-[#3c4043] grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
          <div className="p-2 rounded-xl bg-[#202124]">
            <span className="text-[#9aa0a6] block text-[10px]">Type</span>
            <span className="text-[#e8eaed] font-medium">Peer-to-Peer</span>
          </div>
          <div className="p-2 rounded-xl bg-[#202124]">
            <span className="text-[#9aa0a6] block text-[10px]">Max Peers</span>
            <span className="text-[#34a853] font-medium">2</span>
          </div>
          <div className="p-2 rounded-xl bg-[#202124]">
            <span className="text-[#9aa0a6] block text-[10px]">Encryption</span>
            <span className="text-[#e8eaed] font-medium">DTLS-SRTP</span>
          </div>
          <div className="p-2 rounded-xl bg-[#202124]">
            <span className="text-[#9aa0a6] block text-[10px]">DeepTrace</span>
            <span className="text-[#8ab4f8] font-medium">Active</span>
          </div>
        </div>
      </main>
    </div>
  );
};
