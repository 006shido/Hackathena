import React, { useState } from 'react';
import { Shield, Lock, User as UserIcon, ArrowRight, AlertCircle, KeyRound } from 'lucide-react';

interface LoginFormProps {
  onLogin: (username: string, password: string) => Promise<void>;
  error?: string | null;
  loading?: boolean;
}

export const LoginForm: React.FC<LoginFormProps> = ({
  onLogin,
  error: externalError,
  loading = false,
}) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);

    if (!username.trim() || !password.trim()) {
      setLocalError('Please enter both username and password.');
      return;
    }

    try {
      await onLogin(username.trim(), password.trim());
    } catch {
      // Handled by parent or hook
    }
  };

  const handleFillDemo = (u: string, p: string) => {
    setUsername(u);
    setPassword(p);
    setLocalError(null);
  };

  const displayError = localError || externalError;

  return (
    <div className="w-full max-w-md mx-auto p-6 sm:p-8 rounded-2xl bg-[#28292c] border border-[#3c4043] shadow-2xl">
      {/* Brand Header */}
      <div className="text-center mb-6">
        <div className="inline-flex h-20 w-20 items-center justify-center rounded-2xl bg-white p-2.5 mb-3 shadow-xl shadow-black/40 border border-white/20">
          <img src="/logo.svg" alt="DeepTrace Logo" className="h-full w-full object-contain" />
        </div>
        <h1 className="text-xl font-medium text-[#e8eaed] mb-1">DeepTrace</h1>
        <p className="text-sm text-[#9aa0a6]">Secure Video Call Platform</p>
      </div>

      {/* Error Message */}
      {displayError && (
        <div className="mb-4 flex items-center gap-2 p-3 rounded-xl bg-[#ea4335]/15 border border-[#ea4335]/30 text-[#ea4335] text-sm">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>{displayError}</span>
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm text-[#9aa0a6] mb-1.5 font-medium">
            Username
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-[#9aa0a6]">
              <UserIcon className="h-4 w-4" />
            </div>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username"
              required
              className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-[#202124] border border-[#3c4043] text-sm text-[#e8eaed] placeholder-[#80868b] focus:outline-none focus:border-[#8ab4f8] focus:ring-1 focus:ring-[#8ab4f8] transition-colors"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm text-[#9aa0a6] mb-1.5 font-medium">
            Password
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-[#9aa0a6]">
              <Lock className="h-4 w-4" />
            </div>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-[#202124] border border-[#3c4043] text-sm text-[#e8eaed] placeholder-[#80868b] focus:outline-none focus:border-[#8ab4f8] focus:ring-1 focus:ring-[#8ab4f8] transition-colors"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full mt-3 py-2.5 px-4 rounded-full bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium text-sm flex items-center justify-center gap-2 transition-colors disabled:opacity-50 cursor-pointer"
        >
          {loading ? (
            <span>Signing in...</span>
          ) : (
            <>
              <span>Sign In</span>
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </button>
      </form>

      {/* Demo Accounts */}
      <div className="mt-6 pt-5 border-t border-[#3c4043]">
        <div className="flex items-center gap-2 mb-3">
          <KeyRound className="h-3.5 w-3.5 text-[#9aa0a6]" />
          <span className="text-xs text-[#9aa0a6] font-medium">Quick Demo Access</span>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => handleFillDemo('user', 'user123')}
            className="p-3 rounded-xl bg-[#202124] border border-[#3c4043] hover:border-[#34a853] transition-colors text-left cursor-pointer"
          >
            <div className="flex items-center gap-2 mb-1">
              <div className="h-6 w-6 rounded-full bg-[#34a853] flex items-center justify-center text-[10px] font-bold text-white">U</div>
              <span className="text-xs font-medium text-[#e8eaed]">User</span>
            </div>
            <div className="text-[11px] text-[#9aa0a6]">user / user123</div>
          </button>

          <button
            type="button"
            onClick={() => handleFillDemo('tester', 'tester123')}
            className="p-3 rounded-xl bg-[#202124] border border-[#3c4043] hover:border-[#8ab4f8] transition-colors text-left cursor-pointer"
          >
            <div className="flex items-center gap-2 mb-1">
              <div className="h-6 w-6 rounded-full bg-[#1a73e8] flex items-center justify-center text-[10px] font-bold text-white">T</div>
              <span className="text-xs font-medium text-[#e8eaed]">Tester</span>
            </div>
            <div className="text-[11px] text-[#9aa0a6]">tester / tester123</div>
          </button>
        </div>
      </div>
    </div>
  );
};
