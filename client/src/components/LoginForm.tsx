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
    <div className="w-full max-w-md mx-auto p-5 sm:p-8 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-2xl backdrop-blur-xl cyber-grid relative overflow-hidden">
      {/* Top accent glow */}
      <div className="pointer-events-none absolute -top-24 left-1/2 -translate-x-1/2 h-48 w-48 rounded-full bg-cyan-500/20 blur-3xl" />

      {/* Brand Header */}
      <div className="text-center mb-8 relative z-10">
        <div className="inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-500 via-blue-600 to-indigo-600 shadow-lg shadow-cyan-950/60 mb-3 border border-white/10">
          <Shield className="h-7 w-7 text-white" />
        </div>
        <h1 className="text-2xl font-bold tracking-tight text-white mb-1">DeepTrace</h1>
        <p className="text-xs font-mono tracking-widest text-cyan-400 uppercase">Secure Video Communication</p>
      </div>

      {/* Error Message */}
      {displayError && (
        <div className="mb-6 flex items-start gap-2.5 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5 text-rose-400" />
          <span>{displayError}</span>
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-4 relative z-10">
        <div>
          <label className="block text-xs font-mono text-slate-300 mb-1.5 uppercase tracking-wider">
            Email / Username
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-500">
              <UserIcon className="h-4 w-4" />
            </div>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="user or tester"
              required
              className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-slate-950/90 border border-slate-800 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-colors"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs font-mono text-slate-300 mb-1.5 uppercase tracking-wider">
            Password
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-500">
              <Lock className="h-4 w-4" />
            </div>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-slate-950/90 border border-slate-800 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-colors"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full mt-2 py-3 px-4 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-mono font-semibold text-xs uppercase tracking-wider shadow-lg shadow-cyan-950/50 flex items-center justify-center gap-2 transition-all active:scale-[0.98] disabled:opacity-50"
        >
          {loading ? (
            <span>Authenticating...</span>
          ) : (
            <>
              <span>Login</span>
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </button>
      </form>

      {/* Demo Accounts Section */}
      <div className="mt-8 pt-6 border-t border-slate-800/80 relative z-10">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <KeyRound className="h-3.5 w-3.5 text-cyan-400" />
            Demo Accounts
          </span>
          <span className="text-[10px] text-slate-500 font-mono">1-Click Auto Fill</span>
        </div>

        <div className="grid grid-cols-2 gap-2.5">
          {/* User Account */}
          <button
            type="button"
            onClick={() => handleFillDemo('user', 'user123')}
            className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 hover:border-cyan-500/50 hover:bg-slate-800/50 transition-all text-left group"
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-bold text-slate-200 group-hover:text-cyan-300">USER</span>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 px-1.5 py-0.2 rounded border border-cyan-800/40">
                Normal
              </span>
            </div>
            <div className="text-[11px] font-mono text-slate-400 space-y-0.5">
              <div>user: <span className="text-slate-300">user</span></div>
              <div>pass: <span className="text-slate-300">user123</span></div>
            </div>
          </button>

          {/* Tester Account */}
          <button
            type="button"
            onClick={() => handleFillDemo('tester', 'tester123')}
            className="p-3 rounded-xl bg-slate-950/80 border border-amber-500/30 hover:border-amber-500/70 hover:bg-amber-950/20 transition-all text-left group"
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-bold text-amber-300 group-hover:text-amber-200">TESTER</span>
              <span className="text-[10px] font-mono text-amber-400 bg-amber-950/60 px-1.5 py-0.2 rounded border border-amber-800/40">
                Security
              </span>
            </div>
            <div className="text-[11px] font-mono text-slate-400 space-y-0.5">
              <div>user: <span className="text-slate-300">tester</span></div>
              <div>pass: <span className="text-slate-300">tester123</span></div>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
};
