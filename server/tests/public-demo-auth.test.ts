import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';

const base = { ...process.env };
for (const key of ['PUBLIC_RESEARCH_DEMO', 'JWT_SECRET', 'DEMO_USER_PASSWORD', 'DEMO_TESTER_PASSWORD', 'DEMO_DEFAULT_PASSWORDS']) delete base[key];
const run = (extra: Record<string, string>, script: string) => spawnSync(process.execPath,
  ['--import', 'tsx', '--input-type=module', '-e', script],
  { cwd: new URL('../', import.meta.url), env: { ...base, ...extra }, windowsHide: true, encoding: 'utf8' });
const load = "await import('./src/auth.ts');";
assert.notEqual(run({ NODE_ENV: 'production' }, load).status, 0);
assert.equal(run({ NODE_ENV: 'production', JWT_SECRET: 'test-only-production-secret-abcdefghijklmnopqrstuvwxyz',
  DEMO_USER_PASSWORD: 'test-only-user-password', DEMO_TESTER_PASSWORD: 'test-only-tester-password' }, load).status, 0);
assert.notEqual(run({ PUBLIC_RESEARCH_DEMO: '1' }, load).status, 0);
const configured = {
  PUBLIC_RESEARCH_DEMO: '1', JWT_SECRET: 'test-only-secret-abcdefghijklmnopqrstuvwxyz-123456',
  DEMO_USER_PASSWORD: 'test-only-user-password-1234', DEMO_TESTER_PASSWORD: 'test-only-tester-password-5678',
};
assert.notEqual(run({ ...configured, JWT_SECRET: 'deeptrace-hackathon-super-secret-key-2026' }, load).status, 0);
assert.notEqual(run({ ...configured, DEMO_TESTER_PASSWORD: configured.DEMO_USER_PASSWORD }, load).status, 0);
const valid = run(configured, `
  const assert = (await import('node:assert/strict')).default;
  const auth = await import('./src/auth.ts');
  assert.equal(auth.authenticateUser('user', 'user123'), null);
  assert.equal(auth.authenticateUser('tester', 'tester123'), null);
  const user = auth.authenticateUser('user', process.env.DEMO_USER_PASSWORD);
  const tester = auth.authenticateUser('tester', process.env.DEMO_TESTER_PASSWORD);
  assert.equal(user.role, 'user'); assert.equal(tester.role, 'tester');
  assert.equal(auth.authenticateUser('tester', process.env.DEMO_USER_PASSWORD), null);
  assert.equal(auth.verifyToken(auth.generateToken(tester)).role, 'tester');
`);
assert.equal(valid.status, 0, valid.stderr);
console.log('PASS: public demo rejects default secrets/shared passwords, requires generated credentials and preserves signed roles.');
const simple = run({ ...configured, DEMO_DEFAULT_PASSWORDS: '1', DEMO_USER_PASSWORD: 'user123', DEMO_TESTER_PASSWORD: 'tester123' }, `
  const assert = (await import('node:assert/strict')).default;
  const auth = await import('./src/auth.ts');
  assert.equal(auth.authenticateUser('user','user123').role,'user');
  assert.equal(auth.authenticateUser('tester','tester123').role,'tester');
  assert.equal(auth.authenticateUser('user','tester123'),null);
  assert.equal(auth.authenticateUser('tester','wrong'),null);
`);
assert.equal(simple.status,0,simple.stderr);
assert.notEqual(run({ ...configured, DEMO_DEFAULT_PASSWORDS: '1', JWT_SECRET: 'deeptrace-hackathon-super-secret-key-2026' },load).status,0);
console.log('PASS: explicitly enabled simple demo credentials preserve password checks and require a unique signing secret.');
