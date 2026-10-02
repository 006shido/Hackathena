import React, { useState } from 'react';
import { User as UserIcon, Lock, Eye, EyeOff, ArrowRight, AlertCircle, Users } from 'lucide-react';

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
  const [showPassword, setShowPassword] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);

    if (!username.trim() || !password.trim()) {
      setLocalError('Please enter both email/username and password.');
      return;
    }

    try {
      await onLogin(username.trim(), password.trim());
    } catch {
      // Handled by parent or hook
    }
  };

  const handleQuickDemo = async (u: string, p: string) => {
    setUsername(u);
    setPassword(p);
    setLocalError(null);
    try {
      await onLogin(u, p);
    } catch {
      // Handled by parent
    }
  };

  const displayError = localError || externalError;

  return (
    <div className="w-full max-w-[400px] mx-auto">
      {/* Welcome Title */}
      <div className="mb-8">
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-slate-900 mb-2">
          Welcome back
        </h1>
        <p className="text-sm sm:text-base text-slate-500">
          Sign in to continue to DeepTrace
        </p>
      </div>

      {/* Error Alert */}
      {displayError && (
        <div className="mb-5 flex items-center gap-2.5 p-3.5 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs">
          <AlertCircle className="h-4 w-4 shrink-0 text-red-600" />
          <span>{displayError}</span>
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
              <UserIcon className="h-4 w-4" />
            </div>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Email address"
              required
              className="w-full pl-10 pr-4 py-3 rounded-xl bg-white border border-slate-200 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-600 focus:ring-4 focus:ring-blue-50 transition-all"
            />
          </div>
        </div>

        <div>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
              <Lock className="h-4 w-4" />
            </div>
            <input
              type={showPassword ? 'text' : 'password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              required
              className="w-full pl-10 pr-11 py-3 rounded-xl bg-white border border-slate-200 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-600 focus:ring-4 focus:ring-blue-50 transition-all"
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-slate-400 hover:text-slate-600 transition-colors cursor-pointer"
            >
              {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full mt-2 py-3 px-4 rounded-xl bg-blue-600 hover:bg-blue-700 active:scale-[0.99] text-white font-medium text-sm flex items-center justify-center gap-2 shadow-xs transition-all disabled:opacity-50 cursor-pointer"
        >
          {loading ? (
            <span>Signing in...</span>
          ) : (
            <>
              <span>Sign in</span>
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </button>
      </form>

      {/* Divider */}
      <div className="relative my-7">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-slate-200" />
        </div>
        <div className="relative flex justify-center text-xs">
          <span className="bg-white px-3 text-slate-400 font-medium">or</span>
        </div>
      </div>

      {/* Two Demo Buttons Matching Screenshot */}
      <div className="grid grid-cols-2 gap-3.5">
        <button
          type="button"
          onClick={() => handleQuickDemo('user', 'user123')}
          className="py-3 px-3 rounded-xl border border-slate-200 hover:bg-slate-50 hover:border-slate-300 text-slate-700 text-xs sm:text-sm font-medium transition-all flex items-center justify-center gap-2 cursor-pointer shadow-2xs"
        >
          <UserIcon className="h-4 w-4 text-slate-500" />
          <span>User Demo</span>
        </button>

        <button
          type="button"
          onClick={() => handleQuickDemo('tester', 'tester123')}
          className="py-3 px-3 rounded-xl border border-slate-200 hover:bg-slate-50 hover:border-slate-300 text-slate-700 text-xs sm:text-sm font-medium transition-all flex items-center justify-center gap-2 cursor-pointer shadow-2xs"
        >
          <Users className="h-4 w-4 text-slate-500" />
          <span>Tester Demo</span>
        </button>
      </div>

      {/* Footer link matching screenshot */}
      <div className="mt-10 text-left text-xs sm:text-sm text-slate-500">
        <span>Don't have an account? </span>
        <button
          type="button"
          onClick={() => handleQuickDemo('user', 'user123')}
          className="font-medium text-blue-600 hover:text-blue-700 transition-colors cursor-pointer"
        >
          Sign up
        </button>
      </div>
    </div>
  );
};
