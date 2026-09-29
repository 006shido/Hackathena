import React, { useState } from 'react';
import { Shield, Lock, User as UserIcon, ArrowRight, AlertCircle, KeyRound, Sparkles } from 'lucide-react';

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
    <div className="w-full max-w-lg mx-auto p-6 sm:p-9 rounded-lg bg-[#070709] border border-zinc-800 shadow-2xl relative overflow-hidden">
      {/* Subtle top edge highlight */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-orange-500/40 to-transparent" />

      {/* Brand Header */}
      <div className="text-center mb-7 relative z-10">
        <div className="inline-flex h-12 w-12 items-center justify-center rounded-md bg-orange-500/10 border border-orange-500/30 text-orange-400 mb-3 shadow-md">
          <Shield className="h-6 w-6 text-orange-400" />
        </div>
        <h1 className="text-2xl font-bold tracking-tight text-white mb-1">DeepTrace</h1>
        <p className="text-xs font-mono tracking-widest text-zinc-400 uppercase">Secure 1-to-1 WebRTC Video Security</p>
      </div>

      {/* Error Message */}
      {displayError && (
        <div className="mb-5 flex items-start gap-2.5 p-3 rounded-md bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs font-mono">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5 text-rose-400" />
          <span>{displayError}</span>
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-4 relative z-10">
        <div>
          <label className="block text-xs font-mono text-zinc-300 mb-1.5 uppercase tracking-wider">
            Email / Username
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-zinc-500">
              <UserIcon className="h-4 w-4" />
            </div>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="user or tester"
              required
              className="w-full pl-10 pr-4 py-2.5 rounded-md bg-black border border-zinc-800 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-orange-500/80 focus:ring-1 focus:ring-orange-500/80 transition-colors font-mono"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs font-mono text-zinc-300 mb-1.5 uppercase tracking-wider">
            Password
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-zinc-500">
              <Lock className="h-4 w-4" />
            </div>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              className="w-full pl-10 pr-4 py-2.5 rounded-md bg-black border border-zinc-800 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-orange-500/80 focus:ring-1 focus:ring-orange-500/80 transition-colors font-mono"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full mt-3 py-2.5 px-4 rounded-md bg-orange-600 hover:bg-orange-500 text-white font-mono font-semibold text-xs uppercase tracking-wider shadow-lg shadow-orange-950/40 border border-orange-500 flex items-center justify-center gap-2 transition-all active:scale-[0.99] disabled:opacity-50 cursor-pointer"
        >
          {loading ? (
            <span>Authenticating...</span>
          ) : (
            <>
              <span>Authenticate & Enter</span>
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </button>
      </form>

      {/* Demo Accounts Section */}
      <div className="mt-7 pt-5 border-t border-zinc-800/80 relative z-10">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
            <KeyRound className="h-3.5 w-3.5 text-orange-400" />
            Demo Accounts
          </span>
          <span className="text-[10px] text-zinc-500 font-mono">1-Click Quick Fill</span>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {/* User Account */}
          <button
            type="button"
            onClick={() => handleFillDemo('user', 'user123')}
            className="p-3 rounded-md bg-black border border-zinc-800 hover:border-emerald-500/60 hover:bg-zinc-900/40 transition-all text-left group cursor-pointer"
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-bold text-zinc-200 group-hover:text-emerald-400 transition-colors">USER</span>
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/40 px-1.5 py-0.5 rounded border border-emerald-800/50">
                Normal
              </span>
            </div>
            <div className="text-[11px] font-mono text-zinc-400 space-y-0.5">
              <div>user: <span className="text-zinc-200">user</span></div>
              <div>pass: <span className="text-zinc-200">user123</span></div>
            </div>
          </button>

          {/* Tester Account */}
          <button
            type="button"
            onClick={() => handleFillDemo('tester', 'tester123')}
            className="p-3 rounded-md bg-black border border-zinc-800 hover:border-orange-500/70 hover:bg-zinc-900/40 transition-all text-left group cursor-pointer"
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-bold text-orange-400 group-hover:text-orange-300 transition-colors">TESTER</span>
              <span className="text-[10px] font-mono text-orange-400 bg-orange-950/40 px-1.5 py-0.5 rounded border border-orange-800/50">
                Security
              </span>
            </div>
            <div className="text-[11px] font-mono text-zinc-400 space-y-0.5">
              <div>user: <span className="text-zinc-200">tester</span></div>
              <div>pass: <span className="text-zinc-200">tester123</span></div>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
};
