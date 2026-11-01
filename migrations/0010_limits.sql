-- abuse limits before the repo goes public (docs/public/HARDENING.md). The API also applies this itself once per isolate
-- (CREATE ... IF NOT EXISTS; the ALTERs are skipped when the column exists), so running it by hand is optional.
-- quota: calls and bytes per device (or '*' = all devices) per kind per window (UTC day, or hour for 'puth')
CREATE TABLE IF NOT EXISTS quota (who TEXT NOT NULL, kind TEXT NOT NULL, win INTEGER NOT NULL, n INTEGER NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0, at INTEGER, PRIMARY KEY (who, kind, win));
CREATE INDEX IF NOT EXISTS quota_at ON quota(at);
-- gallery roll-back history kept per device and per day, not only the last 30 versions
ALTER TABLE gallery_keys ADD COLUMN device TEXT;
ALTER TABLE gallery_keys ADD COLUMN at INTEGER;
