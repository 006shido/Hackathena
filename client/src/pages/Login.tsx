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
    <div className="min-h-screen w-full flex items-center justify-center p-4 bg-black relative overflow-hidden amoled-grid">
      {/* Subtle orange accent glow at top */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 h-72 w-[600px] bg-orange-600/[0.04] blur-[120px] pointer-events-none" />

      <div className="w-full relative z-10">
        <LoginForm onLogin={onLogin} loading={loading} error={error} />
      </div>
    </div>
  );
};
