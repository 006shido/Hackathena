import { AuthResponse,User } from '../types/auth';

const TOKEN_KEY = 'deeptrace_auth_token';
const USER_KEY = 'deeptrace_auth_user';

async function saveAuthResponse(response: Response, fallbackError: string): Promise<AuthResponse> {
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || fallbackError);
  }
  const data: AuthResponse = await response.json();
  localStorage.setItem(TOKEN_KEY, data.token);
  localStorage.setItem(USER_KEY, JSON.stringify(data.user));
  return data;
}

export const authService = {
  async login(username: string, password: string): Promise<AuthResponse> {
    const localDemo = password === '__LOCAL_DEMO__' && ['localhost', '127.0.0.1'].includes(window.location.hostname);
    const res = await fetch(localDemo ? '/api/auth/demo' : '/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    return saveAuthResponse(res, 'Authentication failed. Please verify your credentials.');
  },

  async register(username: string, password: string, name?: string): Promise<AuthResponse> {
    const res = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, name }),
    });

    return saveAuthResponse(res, 'Registration failed. Please check your details.');
  },

  getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  },

  getUser(): User | null {
    const raw = localStorage.getItem(USER_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch {
      return null;
    }
  },

  async verifySession(): Promise<User | null> {
    const token = this.getToken();
    if (!token) return null;

    try {
      const res = await fetch('/api/auth/me', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!res.ok) {
        this.logout();
        return null;
      }

      const data = await res.json();
      localStorage.setItem(USER_KEY, JSON.stringify(data.user));
      return data.user;
    } catch {
      // If network offline or endpoint error, fallback to stored user if token exists
      return this.getUser();
    }
  },

  logout(): void {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  },
};
