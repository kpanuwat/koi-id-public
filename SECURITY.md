# Security policy / นโยบายความปลอดภัย

## Reporting a vulnerability / แจ้งช่องโหว่
Please do not open a public issue for security problems. Use GitHub's private vulnerability reporting
("Security" tab, "Report a vulnerability") on this repository. We aim to reply within 7 days.
กรุณาแจ้งช่องโหว่ผ่าน "Report a vulnerability" ในแท็บ Security ของ repo นี้ อย่าเปิด issue สาธารณะ

## Scope / ขอบเขต
- The PWA in `app/` and the Pages Functions API in `functions/`.
- The reference deployment at https://koi-id.pages.dev is operated by the project owner: do not run load tests,
  scanners or write tests against it. Deploy your own instance (SETUP.md) to test.

## Notes for self-hosters / ข้อควรรู้สำหรับผู้ deploy เอง
- The sync API (`/api/*`) issues device tokens on enrolment and is meant for a small trusted group of devices.
  Before opening it to the public, add per-device quotas, pond ownership checks and rate limits
  (e.g. a Cloudflare WAF rate-limiting rule on `/api/*`).
- Keep the admin token (`SYNC_TOKEN` Pages secret) out of git, logs and AI-agent prompts.
- Diagnostic logs contain the browser user agent and app state; publish a privacy notice and set a retention period
  that fits your local law (e.g. Thailand's PDPA).
