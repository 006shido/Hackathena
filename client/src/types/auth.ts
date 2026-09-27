export type UserRole = 'user' | 'tester';

export interface User {
  username: string;
  role: UserRole;
  name: string;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  loading: boolean;
}
