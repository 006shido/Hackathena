import { readdirSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const files = readdirSync(new URL('./', import.meta.url))
  .filter(file => file.endsWith('.test.ts') && file !== 'ml-face-swap.test.ts')
  .sort();
let failed = 0;
for (const file of files) {
  console.log(`\nRunning ${file}`);
  const result = spawnSync(process.execPath, ['--import', 'tsx', `tests/${file}`], {
    cwd: new URL('../', import.meta.url), stdio: 'inherit', windowsHide: true, timeout: 120000,
  });
  if (result.error) console.error(result.error.message);
  if (result.status !== 0) failed++;
}
console.log(`\n${files.length - failed}/${files.length} suites passed.`);
process.exitCode = failed ? 1 : 0;
