import { useState, useEffect, useCallback } from 'react';
import { User, AuthState } from '../types/auth';
import { authService } from '../services/auth';

export function useAuth() {
  const [state, setState] = useState<AuthState>({
    user: authService.getUser(),
    token: authService.getToken(),
    isAuthenticated: !!authService.getToken(),
    loading: true,
  });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    async function checkAuth() {
      try {
        const user = await authService.verifySession();
        if (mounted) {
          if (user) {
            setState({
              user,
              token: authService.getToken(),
              isAuthenticated: true,
              loading: false,
            });
          } else {
            setState({
              user: null,
              token: null,
              isAuthenticated: false,
              loading: false,
            });
          }
        }
      } catch {
        if (mounted) {
          setState((prev) => ({ ...prev, loading: false }));
        }
      }
    }

    checkAuth();
    return () => {
      mounted = false;
    };
  }, []);

  const login = useCallback(async (username: string, password: string): Promise<User> => {
    setError(null);
    try {
      const data = await authService.login(username, password);
      setState({
        user: data.user,
        token: data.token,
        isAuthenticated: true,
        loading: false,
      });
      return data.user;
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Login failed';
      setError(message);
      throw err;
    }
  }, []);

  const logout = useCallback(() => {
    authService.logout();
    setState({
      user: null,
      token: null,
      isAuthenticated: false,
      loading: false,
    });
  }, []);

  return {
    user: state.user,
    token: state.token,
    isAuthenticated: state.isAuthenticated,
    loading: state.loading,
    error,
    login,
    logout,
  };
}
