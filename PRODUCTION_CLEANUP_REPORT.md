# Production cleanup report

Completed 2026-10-09. Existing local changes and application behavior were preserved. The initial cleanup created no commit; the subsequent user-authorized GitHub release commits include the cleaned platform. See `RELEASE_VALIDATION.md` for release preparation checks.

Follow-up ML separation: four unchanged Phase 6D/6E helpers now live in `ml/inference/face_correspondence.py`, with legacy research imports preserved. Nine unconsumed Phase 6C/6E/6F diagnostic checkpoints were removed, freeing 2.09 GiB. See `ml/RUNTIME_LAYOUT.md` for the exact list and required runtime assets. Fresh-service three-pair inference, error/concurrency integration checks, actual face anomaly inference, frame projection/fallback, service metadata, legacy-import compatibility, and parsing of 120 Python files passed. All services remain running.

## Repository analysis

Reviewed React/Vite frontend, Express/Socket.IO backend, authentication and room lifecycle, HTTP-to-Python ML proxy, FastAPI inference and cancellation, research model/training code, browser media pipelines, configuration, environment-variable references, lockfiles, tests, and Docker setup. The application uses in-memory demo accounts and rooms; there is no active database integration. Research folders contain runtime dependencies and were retained.

## Removed files and dead code

- `client/src/components/AttackSimulator.tsx`: obsolete component; active controls use AIFaceSwapPanel.
- `client/src/App.css`: unimported starter stylesheet.
- `client/src/hooks/useKeyboardShortcuts.ts`: unreferenced hook.
- `client/public/logo.svg.bak`, `logo_vector_test.svg`, `icons.svg`, `morrow_camera.svg`: unused assets/backup.
- Unused imports, bindings, private fields, cancellation handler, and uncalled signaling, room, and custom-avatar methods.
- Pure debug logs; retained diagnostics and the feedback log because it is currently the only feedback output.
- Removed 4,189 generated dependency files from Git's index. These removals are staged; installed files remain on disk. Added ignore rules for dependencies, build/cache output, secrets, and large local ML artifacts.

## Dependencies

Removed unused client `@supabase/supabase-js` (and its unused transitive dependencies) and server `dotenv`. Applied compatible security fixes: concurrently 9.2.5, shell-quote 1.12.0, source-map-js 1.2.2. No major upgrades. All three npm audits report zero known vulnerabilities.

## Refactoring and performance

- Shared dashboard room/greeting helpers and authentication response/transition logic.
- Lazy-loaded call and Phase 6G screens. Initial JavaScript decreased from approximately 652.54 KB to 282.45 KB (about 57%); production build no longer reports an oversized chunk.
- Corrected React dependencies, state initialization/reset patterns, captured cleanup targets, and callback/ref ordering.
- Aborted discarded health requests and revoked preview object URLs on replacement/unmount.
- Bounded audio pitch search to valid sample windows.
- Bounded image-upload reads and kept inference locks/resources alive until cancelled workers actually finish.
- Shared image data-URL encoding; reproducible Docker installs use lockfiles and prune backend development dependencies.
- Enabled unused-code TypeScript checks, scoped lint to project files, and added a runner for all isolated JavaScript suites.

## Security

Production startup now rejects missing/known-default signing secrets and weak/shared account passwords. Malformed credentials and inherited object-property names are rejected. Regression tests cover these cases. No secret values are included in this report. Docker excludes environment files and unnecessary local ML assets.

Production deployment must inject environment variables through the host/process environment: this project does not automatically load `.env` files. Set `NODE_ENV=production`, a unique `JWT_SECRET` of at least 32 characters, and distinct `DEMO_USER_PASSWORD`/`DEMO_TESTER_PASSWORD` values of at least 12 characters. Configure existing ML host/port, client feature flags, and TURN settings for the deployment.

## Validation

- Final `npm run lint`, `npm run typecheck`, and `npm run build`: passed.
- Final `npm test`: 11/11 suites passed with localhost access. Restricted sandbox networking initially prevented four integration suites from connecting; the authorized rerun passed.
- Live backend ML integration and Python localhost API tests: passed, including all three retained CelebA pairs, invalid uploads, unavailable upstream, and concurrent-request locking. Identity gains match measurements captured before the ML refactor (1.0636, 0.7785, 0.6639).
- Python unit suites: 11 cases passed; worker cancellation test passed; model architecture checks: 16 assertions passed.
- Setup, frame projection/fallback, service metadata, flow stability, occlusion composite, and Torch reference parity checks: passed.
- Recorded-media browser acceptance: 12 assertions passed for joins, video/audio, screen sharing, source replacement, and stopping. Actual call UI verified two connected participants and microphone/camera/share cleanup. Demo login and Phase 6G inference UI verified.
- Final static scan: 66 JavaScript/TypeScript modules, no broken relative imports or circular imports; 119 Python files parsed successfully. Git whitespace checks passed.
- Docker executable unavailable: container build was not verified.

Local validation evidence and the pre-cleanup source/change snapshots are stored in ignored `.run-logs/production-cleanup/`.

## Remaining debt and manual review

- Two research tests still require previously deleted images: `test_identity_guard.py` uses held-out identity-specific fixtures; `test_video_tracking.py` uses image 098180. They fail because those fixtures are absent. Assertions were preserved. Full-dataset training/evaluation cannot run without restoring the dataset.
- Frontend registration references an API route that is not implemented. Feedback currently logs locally instead of persisting. Adding these behaviors would change the product and was outside cleanup scope.
- Accounts and rooms are in memory. Production identity management, persistence, password hashing, rate limits, and session policy require product decisions.
- Broad CORS and public ML photo/health endpoints remain compatible with the existing public test panel. Review authentication, access limits, and origin restrictions before internet deployment. Keep the Python service private; it trusts the backend's owner header.
- Browser TURN credentials should be short-lived. Vite variables are public browser configuration.
- Detection outputs remain experimental; research weights and third-party model licenses require review for production use.
- `acousticScore.ts` and `voiceReference.ts` are exercised by tests but are not imported by the production frontend. Retained rather than deleting tested algorithms or changing live scoring.
- Git ignore rules do not remove previously committed secrets or large artifacts from history. Historical cleanup requires a separate review.

The local platform remains running on frontend 5173, backend 5001, and ML 8001. This cleanup does not certify an internet-facing deployment as production-ready while the above gaps remain.
