import assert from 'node:assert/strict';
import { authenticateUser } from '../src/auth.js';

for (const username of [null, {}, [], 42, '__proto__', 'constructor', 'toString']) {
  assert.equal(authenticateUser(username, 'invalid'), null);
}
for (const password of [null, {}, [], 42]) {
  assert.equal(authenticateUser('user', password), null);
}
const password = process.env.DEMO_USER_PASSWORD || 'user123';
assert.equal(authenticateUser(' USER ', password)?.role, 'user');
assert.equal(authenticateUser('user', `${password}-wrong`), null);
console.log('PASS: malformed credentials and inherited property names are rejected.');
