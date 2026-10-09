import { useCallback,useEffect,useState } from 'react';
import { flushSync } from 'react-dom';
import { authService } from '../services/auth';
import { AuthState,User } from '../types/auth';

function commitWithTransition(commit: () => void): void {
  const doc = document as unknown as { startViewTransition?: (callback: () => void) => void };
  if (typeof doc.startViewTransition === 'function') {
    doc.startViewTransition(() => flushSync(commit));
  } else {
    commit();
  }
}
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

  const login = useCallback(
    async (
      username: string,
      password: string,
      onVerified?: () => Promise<void> | void
    ): Promise<User> => {
      setError(null);
      try {
        const data = await authService.login(username, password);

        // Allow UI to execute smooth verification feedback / exit choreography before committing
        if (onVerified) {
          await onVerified();
        }

        const commitState = () => {
          setState({
            user: data.user,
            token: data.token,
            isAuthenticated: true,
            loading: false,
          });
        };

        commitWithTransition(commitState);

        return data.user;
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Login failed';
        setError(message);
        throw err;
      }
    },
    []
  );

  const register = useCallback(
    async (
      username: string,
      password: string,
      name?: string,
      onVerified?: () => Promise<void> | void
    ): Promise<User> => {
      setError(null);
      try {
        const data = await authService.register(username, password, name);

        if (onVerified) {
          await onVerified();
        }

        const commitState = () => {
          setState({
            user: data.user,
            token: data.token,
            isAuthenticated: true,
            loading: false,
          });
        };

        commitWithTransition(commitState);

        return data.user;
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Registration failed';
        setError(message);
        throw err;
      }
    },
    []
  );

  const logout = useCallback(() => {
    const commitState = () => {
      authService.logout();
      setState({
        user: null,
        token: null,
        isAuthenticated: false,
        loading: false,
      });
    };

    commitWithTransition(commitState);
  }, []);

  return {
    user: state.user,
    token: state.token,
    isAuthenticated: state.isAuthenticated,
    loading: state.loading,
    error,
    login,
    register,
    logout,
  };
}
