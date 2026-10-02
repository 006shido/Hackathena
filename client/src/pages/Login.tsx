import React from 'react';
import { LoginForm } from '../components/LoginForm';

interface LoginPageProps {
  onLogin: (username: string, password: string) => Promise<any>;
  loading?: boolean;
  error?: string | null;
}

export const LoginPage: React.FC<LoginPageProps> = ({
  onLogin,
  loading,
  error,
}) => {
  return (
    <div className="min-h-screen w-full flex bg-white text-slate-900 font-sans selection:bg-blue-100">
      {/* Left Column: Top-Left Logo + Centered Form */}
      <div className="w-full lg:w-[48%] xl:w-[45%] min-h-screen flex flex-col justify-between p-6 sm:p-10 lg:p-12 z-10 bg-white">
        {/* Top-Left Logo */}
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 sm:h-9 sm:w-9 shrink-0">
            <img src="/logo.svg" alt="DeepTrace Logo" className="h-full w-full" />
          </div>
          <span className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">DeepTrace</span>
        </div>

        {/* Centered Login Form */}
        <div className="my-auto w-full max-w-[390px] mx-auto py-8">
          <LoginForm onLogin={onLogin} loading={loading} error={error} />
        </div>

        {/* Bottom spacer for symmetrical balance */}
        <div className="h-4" />
      </div>

      {/* Right Column: Exact Reference Design with DeepTrace-Specific Taglines */}
      <div className="hidden lg:flex flex-1 relative bg-[#f2f6fd] flex-col justify-between p-10 lg:p-14 xl:p-16 overflow-hidden select-none border-l border-slate-100">
        {/* Soft Geometric Shapes: exactly 1 middle-right disc + 1 bottom-right dome (no ripples) */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 800 900"
          preserveAspectRatio="xMaxYMid slice"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <linearGradient id="bgGrad" x1="0" y1="0" x2="800" y2="900" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#f8faff" />
              <stop offset="50%" stopColor="#f2f6fe" />
              <stop offset="100%" stopColor="#eaf1fc" />
            </linearGradient>

            <linearGradient id="discGrad" x1="640" y1="240" x2="820" y2="420" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#dce8fa" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#cfdff7" stopOpacity="0.9" />
            </linearGradient>

            <linearGradient id="domeGrad" x1="360" y1="900" x2="800" y2="620" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#e2eeff" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#d2e3fa" stopOpacity="0.95" />
            </linearGradient>
          </defs>

          {/* Smooth Background */}
          <rect width="800" height="900" fill="url(#bgGrad)" />

          {/* Middle-Right Soft Disc */}
          <circle cx="820" cy="350" r="180" fill="url(#discGrad)" />

          {/* Single Smooth Bottom-Right Arc (No concentric ripples) */}
          <circle cx="860" cy="1120" r="680" fill="url(#domeGrad)" />
        </svg>

        {/* Top-Right Tagline */}
        <div className="w-full flex justify-end z-10">
          <span className="text-[13px] sm:text-sm font-normal text-slate-400 tracking-wide">
            Simple. Secure. Verified.
          </span>
        </div>

        {/* Hero Copy (DeepTrace Website Specific) */}
        <div className="w-full max-w-lg my-auto pl-6 sm:pl-10 xl:pl-16 z-10">
          <h2 className="text-3xl sm:text-4xl xl:text-[42px] font-bold tracking-tight text-slate-900 leading-[1.2] mb-4">
            Secure conversations,<br />
            anywhere.
          </h2>

          <p className="text-sm sm:text-base text-slate-500 leading-relaxed max-w-sm">
            Connect with your team, friends and collaborators — protected with real-time deepfake defense.
          </p>
        </div>

        {/* Bottom space */}
        <div className="w-full h-4 z-10" />
      </div>
    </div>
  );
};
