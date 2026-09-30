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
    <div className="min-h-screen w-full flex items-center justify-center p-4 bg-[#202124]">
      <div className="w-full relative z-10">
        <LoginForm onLogin={onLogin} loading={loading} error={error} />
      </div>
    </div>
  );
};
