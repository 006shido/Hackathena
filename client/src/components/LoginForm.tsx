import React, { useState } from 'react';
import { User as UserIcon, Lock, Eye, EyeOff, ArrowRight, AlertCircle, Users, Check, Loader2 } from 'lucide-react';

interface LoginFormProps {
  onLogin: (
    username: string,
    password: string,
    onVerified?: () => Promise<void> | void
  ) => Promise<any>;
  error?: string | null;
  loading?: boolean;
  onExitingChange?: (isExiting: boolean) => void;
}

export const LoginForm: React.FC<LoginFormProps> = ({
  onLogin,
  error: externalError,
  loading: externalLoading = false,
  onExitingChange,
}) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  // Status progression: idle -> submitting -> success
  const [status, setStatus] = useState<'idle' | 'submitting' | 'success'>('idle');
  const [activeAction, setActiveAction] = useState<'form' | 'user-demo' | 'tester-demo' | null>(null);

  const isBusy = status !== 'idle' || externalLoading;

  const executeLoginFlow = async (
    u: string,
    p: string,
    action: 'form' | 'user-demo' | 'tester-demo'
  ) => {
    setLocalError(null);
    setStatus('submitting');
    setActiveAction(action);

    try {
      await onLogin(u, p, async () => {
        // When the sign is successfully verified:
        setStatus('success');

        // 1. Give satisfying visual confirmation of success (emerald checkmark)
        await new Promise((resolve) => setTimeout(resolve, 450));

        // 2. Smoothly initiate the exit transition
        if (onExitingChange) {
          onExitingChange(true);
        }
        await new Promise((resolve) => setTimeout(resolve, 250));
      });
    } catch {
      setStatus('idle');
      setActiveAction(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      setLocalError('Please enter both email/username and password.');
      return;
    }
    await executeLoginFlow(username.trim(), password.trim(), 'form');
  };

  const handleQuickDemo = async (u: string, p: string, action: 'user-demo' | 'tester-demo') => {
    setUsername(u);
    setPassword(p);
    await executeLoginFlow(u, p, action);
  };

  const displayError = localError || externalError;

  return (
    <div className="w-full max-w-[400px] mx-auto transition-opacity duration-300">
      {/* Welcome Title */}
      <div className="mb-5 sm:mb-7 lg:mb-8">
        <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight text-slate-900 mb-1.5 sm:mb-2">
          Welcome back
        </h1>
        <p className="text-xs sm:text-sm lg:text-base text-slate-500">
          Sign in to continue to DeepTrace
        </p>
      </div>

      {/* Error Alert */}
      {displayError && (
        <div className="mb-4 sm:mb-5 flex items-center gap-2.5 p-3 sm:p-3.5 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs animate-shake">
          <AlertCircle className="h-4 w-4 shrink-0 text-red-600" />
          <span>{displayError}</span>
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-3 sm:space-y-4">
        <div>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
              <UserIcon className="h-4 w-4" />
            </div>
            <input
              id="login-username-input"
              type="text"
              value={username}
              disabled={isBusy}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Email address"
              required
              className={`w-full pl-10 pr-4 py-2.5 sm:py-3 rounded-xl bg-white border text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none transition-all duration-200 ${
                status === 'success'
                  ? 'border-emerald-300 bg-emerald-50/20'
                  : 'border-slate-200 focus:border-blue-600 focus:ring-4 focus:ring-blue-50'
              } disabled:opacity-75 disabled:cursor-not-allowed`}
            />
          </div>
        </div>

        <div>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
              <Lock className="h-4 w-4" />
            </div>
            <input
              id="login-password-input"
              type={showPassword ? 'text' : 'password'}
              value={password}
              disabled={isBusy}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              required
              className={`w-full pl-10 pr-11 py-2.5 sm:py-3 rounded-xl bg-white border text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none transition-all duration-200 ${
                status === 'success'
                  ? 'border-emerald-300 bg-emerald-50/20'
                  : 'border-slate-200 focus:border-blue-600 focus:ring-4 focus:ring-blue-50'
              } disabled:opacity-75 disabled:cursor-not-allowed`}
            />
            <button
              type="button"
              id="login-toggle-password-button"
              disabled={isBusy}
              onClick={() => setShowPassword(!showPassword)}
              className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-slate-400 hover:text-slate-600 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
        </div>

        {/* Main Action Button with Smooth Morphing Transition */}
        <button
          id="login-submit-button"
          type="submit"
          disabled={isBusy}
          className={`w-full mt-1.5 sm:mt-2 py-2.5 sm:py-3 px-4 rounded-xl font-medium text-sm flex items-center justify-center gap-2 shadow-xs transition-all duration-300 cursor-pointer ${
            status === 'success'
              ? 'bg-emerald-600 text-white ring-4 ring-emerald-100 scale-[1.01] shadow-emerald-600/20'
              : status === 'submitting' && activeAction === 'form'
              ? 'bg-blue-600 text-white opacity-90 cursor-wait'
              : 'bg-blue-600 hover:bg-blue-700 active:scale-[0.99] text-white disabled:opacity-50 disabled:cursor-not-allowed'
          }`}
        >
          {status === 'success' ? (
            <span className="flex items-center gap-2 animate-scale-in">
              <Check className="h-4 w-4 animate-checkmark-pop stroke-[2.5]" />
              <span className="font-semibold tracking-wide">Access Verified</span>
            </span>
          ) : status === 'submitting' && activeAction === 'form' ? (
            <span className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Authenticating...</span>
            </span>
          ) : (
            <>
              <span>Sign in</span>
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </>
          )}
        </button>
      </form>

      {/* Divider */}
      <div className="relative my-4 sm:my-6">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-slate-200" />
        </div>
        <div className="relative flex justify-center text-xs">
          <span className="bg-white px-3 text-slate-400 font-medium">or</span>
        </div>
      </div>

      {/* Two Demo Buttons with Reactive Transitions */}
      <div className="grid grid-cols-2 gap-2.5 sm:gap-3.5">
        <button
          id="login-user-demo-button"
          type="button"
          disabled={isBusy}
          onClick={() => handleQuickDemo('user', 'user123', 'user-demo')}
          className={`py-2.5 sm:py-3 px-3 rounded-xl border text-xs sm:text-sm font-medium transition-all duration-200 flex items-center justify-center gap-1.5 sm:gap-2 cursor-pointer shadow-2xs ${
            status === 'success' && activeAction === 'user-demo'
              ? 'bg-emerald-50 border-emerald-300 text-emerald-700 ring-2 ring-emerald-200'
              : status === 'submitting' && activeAction === 'user-demo'
              ? 'bg-blue-50 border-blue-200 text-blue-700 cursor-wait'
              : 'border-slate-200 hover:bg-slate-50 hover:border-slate-300 text-slate-700 disabled:opacity-50 disabled:cursor-not-allowed'
          }`}
        >
          {status === 'success' && activeAction === 'user-demo' ? (
            <>
              <Check className="h-4 w-4 text-emerald-600 animate-checkmark-pop stroke-[2.5]" />
              <span className="font-semibold">Verified</span>
            </>
          ) : status === 'submitting' && activeAction === 'user-demo' ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-600" />
              <span>Entering...</span>
            </>
          ) : (
            <>
              <UserIcon className="h-4 w-4 text-slate-500 shrink-0" />
              <span>User Demo</span>
            </>
          )}
        </button>

        <button
          id="login-tester-demo-button"
          type="button"
          disabled={isBusy}
          onClick={() => handleQuickDemo('tester', 'tester123', 'tester-demo')}
          className={`py-2.5 sm:py-3 px-3 rounded-xl border text-xs sm:text-sm font-medium transition-all duration-200 flex items-center justify-center gap-1.5 sm:gap-2 cursor-pointer shadow-2xs ${
            status === 'success' && activeAction === 'tester-demo'
              ? 'bg-emerald-50 border-emerald-300 text-emerald-700 ring-2 ring-emerald-200'
              : status === 'submitting' && activeAction === 'tester-demo'
              ? 'bg-blue-50 border-blue-200 text-blue-700 cursor-wait'
              : 'border-slate-200 hover:bg-slate-50 hover:border-slate-300 text-slate-700 disabled:opacity-50 disabled:cursor-not-allowed'
          }`}
        >
          {status === 'success' && activeAction === 'tester-demo' ? (
            <>
              <Check className="h-4 w-4 text-emerald-600 animate-checkmark-pop stroke-[2.5]" />
              <span className="font-semibold">Verified</span>
            </>
          ) : status === 'submitting' && activeAction === 'tester-demo' ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-600" />
              <span>Entering...</span>
            </>
          ) : (
            <>
              <Users className="h-4 w-4 text-slate-500 shrink-0" />
              <span>Tester Demo</span>
            </>
          )}
        </button>
      </div>

      {/* Footer link */}
      <div className="mt-5 sm:mt-7 lg:mt-9 text-left text-xs sm:text-sm text-slate-500">
        <span>Don't have an account? </span>
        <button
          id="login-signup-button"
          type="button"
          disabled={isBusy}
          onClick={() => handleQuickDemo('user', 'user123', 'user-demo')}
          className="font-medium text-blue-600 hover:text-blue-700 transition-colors cursor-pointer disabled:opacity-50"
        >
          Sign up
        </button>
      </div>
    </div>
  );
};
