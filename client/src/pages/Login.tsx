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
    <div className="min-h-screen w-full flex items-center justify-center p-4 bg-[#08090d] relative overflow-hidden">
      {/* Background cyber grid */}
      <div className="absolute inset-0 cyber-grid opacity-30 pointer-events-none" />

      {/* Decorative ambient gradient */}
      <div className="absolute top-1/4 left-1/4 h-96 w-96 rounded-full bg-cyan-600/10 blur-[128px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 h-96 w-96 rounded-full bg-blue-600/10 blur-[128px] pointer-events-none" />

      <div className="w-full relative z-10">
        <LoginForm onLogin={onLogin} loading={loading} error={error} />
      </div>
    </div>
  );
};
