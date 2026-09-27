import jwt from 'jsonwebtoken';
import { User, UserRole } from './types.js';

const JWT_SECRET = process.env.JWT_SECRET || 'deeptrace-hackathon-super-secret-key-2026';

// Demo accounts database
const DEMO_USERS: Record<string, { password: string; role: UserRole; name: string }> = {
  user: {
    password: 'user123',
    role: 'user',
    name: 'Normal User',
  },
  tester: {
    password: 'tester123',
    role: 'tester',
    name: 'Security Tester',
  },
};

export function authenticateUser(username: string, password: string): User | null {
  const account = DEMO_USERS[username.toLowerCase().trim()];
  if (!account || account.password !== password) {
    return null;
  }
  return {
    username: username.toLowerCase().trim(),
    role: account.role,
    name: account.name,
  };
}

export function generateToken(user: User): string {
  return jwt.sign(
    {
      username: user.username,
      role: user.role,
      name: user.name,
    },
    JWT_SECRET,
    { expiresIn: '12h' }
  );
}

export function verifyToken(token: string): User | null {
  try {
    const decoded = jwt.verify(token, JWT_SECRET) as User;
    if (decoded && decoded.username && (decoded.role === 'user' || decoded.role === 'tester')) {
      return {
        username: decoded.username,
        role: decoded.role,
        name: decoded.name,
      };
    }
    return null;
  } catch {
    return null;
  }
}
