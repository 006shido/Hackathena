import jwt from 'jsonwebtoken';
const JWT_SECRET = process.env.JWT_SECRET || 'deeptrace-hackathon-super-secret-key-2026';
if (process.env.NODE_ENV === 'production' && (!process.env.JWT_SECRET || process.env.JWT_SECRET.length < 32 ||
    process.env.JWT_SECRET === 'deeptrace-hackathon-super-secret-key-2026' ||
    !process.env.DEMO_USER_PASSWORD || process.env.DEMO_USER_PASSWORD.length < 12 ||
    !process.env.DEMO_TESTER_PASSWORD || process.env.DEMO_TESTER_PASSWORD.length < 12 ||
    process.env.DEMO_USER_PASSWORD === process.env.DEMO_TESTER_PASSWORD)) {
    throw new Error('Production requires a unique JWT secret and distinct strong account passwords.');
}
if (process.env.PUBLIC_RESEARCH_DEMO === '1' && (!process.env.JWT_SECRET || process.env.JWT_SECRET.length < 32 ||
    process.env.JWT_SECRET === 'deeptrace-hackathon-super-secret-key-2026' ||
    (process.env.DEMO_DEFAULT_PASSWORDS !== '1' && (!process.env.DEMO_USER_PASSWORD || process.env.DEMO_USER_PASSWORD.length < 12 ||
        !process.env.DEMO_TESTER_PASSWORD || process.env.DEMO_TESTER_PASSWORD.length < 12 ||
        process.env.DEMO_USER_PASSWORD === 'user123' || process.env.DEMO_TESTER_PASSWORD === 'tester123' ||
        process.env.DEMO_USER_PASSWORD === process.env.DEMO_TESTER_PASSWORD)))) {
    throw new Error('Public research demo requires a unique JWT secret and distinct strong demo passwords.');
}
// Demo accounts database
const DEMO_USERS = {
    user: {
        password: process.env.DEMO_USER_PASSWORD || 'user123',
        role: 'user',
        name: 'Normal User',
    },
    tester: {
        password: process.env.DEMO_TESTER_PASSWORD || 'tester123',
        role: 'tester',
        name: 'Security Tester',
    },
};
export function authenticateUser(username, password) {
    if (typeof username !== 'string' || typeof password !== 'string')
        return null;
    const normalizedUsername = username.toLowerCase().trim();
    if (!Object.hasOwn(DEMO_USERS, normalizedUsername))
        return null;
    const account = DEMO_USERS[normalizedUsername];
    if (!account || account.password !== password) {
        return null;
    }
    return {
        username: normalizedUsername,
        role: account.role,
        name: account.name,
    };
}
export function generateToken(user) {
    return jwt.sign({
        username: user.username,
        role: user.role,
        name: user.name,
    }, JWT_SECRET, { expiresIn: '12h' });
}
export function verifyToken(token) {
    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        if (decoded && typeof decoded.username === 'string' && decoded.username && (decoded.role === 'user' || decoded.role === 'tester')) {
            return {
                username: decoded.username,
                role: decoded.role,
                name: decoded.name,
            };
        }
        return null;
    }
    catch {
        return null;
    }
}
