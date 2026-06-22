# Local HTTPS for development (portless)

Operator runbook for serving the dev stack over HTTPS at `https://medtimeline.localhost`.

## Why this exists

The plain dev stack binds loopback HTTP only (`http://localhost:8000` for the API, `:3000` for the frontend). The backend's security-headers middleware emits `Strict-Transport-Security` (HSTS) only when it observes an HTTPS request, so on the loopback HTTP stack that header never fires. That leaves the HSTS path untested locally even though it matters in any TLS deployment.

[portless](https://github.com/vercel-labs/portless) fronts both apps with a trusted local certificate and forwards `X-Forwarded-Proto: https` to the upstream. The backend already honors that header in development (`app/middleware/security_headers.py`, `should_emit_hsts`), so running behind portless makes it emit HSTS exactly as it would behind a production TLS proxy. This is a development convenience, not a deployment topology: for real network exposure, front the app with a real TLS proxy and turn on full-disk encryption (see `docs/operations-backup-restore.md`).

The wiring:

```
browser ──https──> https://medtimeline.localhost      (portless ─> Next.js on an ephemeral $PORT)
browser ──https──> https://api.medtimeline.localhost   (portless ─> uvicorn on an ephemeral $PORT)
```

The frontend at `medtimeline.localhost` calls the API at `api.medtimeline.localhost`. Both are HTTPS, so there is no mixed-content blocking, and the API call is cross-origin to the api subdomain, which is why CORS has to allow the frontend origin (the run script sets that up).

## What portless costs

- Apache-2.0 license. Compatible with the project's permissive-OSS rule.
- About 0.4 MB unpacked, with no runtime npm dependencies (a single bundled `dist/cli.js`).
- A development-only CLI. It is not added to either app's dependency manifest, so it never ships in the backend or frontend runtime.
- It generates a local certificate authority and binds port 443. Trusting the CA and binding 443 use `sudo` once, which is the operator step below.

## One-time setup (requires sudo)

These commands install the CLI and trust the local CA. They are intentionally left for you to run, because they modify the system trust store.

```bash
npm install -g portless
portless trust
```

`portless trust` generates a local CA, adds it to the system/browser trust store, and authorizes binding port 443. macOS and Linux prompt for `sudo`. This is what lets the browser accept `https://*.localhost` without a certificate warning. Verify it reports the CA as trusted before continuing.

Restart any already-open browser after the first `portless trust` so it picks up the new CA.

## Start the HTTPS stack

You still need Postgres and Redis up (`just dev` brings up the db/redis containers, or use your native services). Then, from the repo root:

```bash
just up-https
# or: bash scripts/run-https.sh
```

The script starts both apps under portless on ephemeral ports (it never binds 8000 or 3000, so it will not collide with a plain `just dev` stack), and it exports the two settings the HTTPS topology needs:

- `NEXT_PUBLIC_API_URL=https://api.medtimeline.localhost/api/v1` so the browser calls the HTTPS API.
- `CORS_ORIGINS=https://medtimeline.localhost,http://localhost:3000` so the backend accepts the HTTPS frontend origin (environment values override `.env`).

`APP_ENV` stays `development`, so the production-only HTTPS-redirect and trusted-host middleware are not installed. The backend still terminates plain HTTP itself; portless does the TLS. Stop the stack with Ctrl-C, which tears down both apps. The shared portless proxy daemon keeps running; stop it with `portless proxy stop` if you want it gone.

First boot is slower than usual: the backend warm-loads its models and Next.js compiles on first request.

## Verification checklist

1. The API emits HSTS over HTTPS:

   ```bash
   curl -sI https://api.medtimeline.localhost/api/v1/health | grep -i strict-transport
   ```

   Expected: `strict-transport-security: max-age=31536000; includeSubDomains` (the `HSTS_VALUE` in `app/middleware/security_headers.py`). If the CA is not trusted yet, add `-k` to confirm the header is present while you sort out trust, but the goal is for it to work without `-k`.

2. Plain loopback HTTP still does not emit HSTS (the dev behavior is unchanged):

   ```bash
   curl -sI http://localhost:8000/api/v1/health | grep -ci strict-transport   # expect 0, when a plain dev backend is also running
   ```

3. The browser trusts the certificate. Open `https://medtimeline.localhost`, confirm there is no certificate warning, log in, and confirm API calls succeed (no CORS or mixed-content errors in the console). The lock icon should show a valid certificate issued by the portless local CA.

4. Optional, confirm the cross-origin API call is allowed:

   ```bash
   curl -sI -H "Origin: https://medtimeline.localhost" \
     https://api.medtimeline.localhost/api/v1/health | grep -i access-control-allow-origin
   ```

   Expected: `access-control-allow-origin: https://medtimeline.localhost`.

## Switching back to plain HTTP

Nothing to undo. Stop the HTTPS stack and run `just dev` (or `just backend` / `just frontend`) as before. The plain loopback HTTP flow is untouched: no code path changed, only the run script sets HTTPS-specific environment variables, and only while it runs.

## Operator TODO (the human-gated steps)

- Run `npm install -g portless` and `portless trust` once (the `sudo` CA install above). The automation deliberately does not do this.
- After the first trust, restart the browser and do the final visual check in step 3: trusted certificate, login works, no console errors.
