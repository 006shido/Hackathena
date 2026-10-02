import React from 'react';

interface LoadingScreenProps {
  message?: string;
}

export const LoadingScreen: React.FC<LoadingScreenProps> = ({
  message = 'Loading...',
}) => {
  return (
    <div className="min-h-screen w-full bg-white flex flex-col items-center justify-center p-6 select-none font-sans">
      <div className="flex flex-col items-center animate-in fade-in duration-300">
        {/* Logo at the center with generous padding and smooth glow */}
        <div className="relative mb-5 flex items-center justify-center">
          <div className="absolute inset-0 bg-blue-500/15 rounded-3xl blur-xl scale-125" />
          <div className="relative h-20 w-20 sm:h-24 sm:w-24 shadow-xl shadow-blue-500/20 animate-pulse">
            <img
              src="/logo.svg"
              alt="DeepTrace Logo"
              className="h-full w-full"
            />
          </div>
        </div>

        {/* Brand Name */}
        <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900 mb-2">
          DeepTrace
        </h1>

        {/* Animated Loading Bar */}
        <div className="w-32 sm:w-40 h-1 bg-slate-100 rounded-full overflow-hidden mb-3">
          <div className="h-full bg-blue-600 rounded-full w-1/2 animate-[pulse_1.5s_ease-in-out_infinite]" />
        </div>

        {/* Status Message */}
        <span className="text-xs font-medium text-slate-400 tracking-wider uppercase">
          {message}
        </span>
      </div>
    </div>
  );
};
