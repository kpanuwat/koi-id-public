-- zero-setup sync: every device enrols itself and gets its own token (only its hash is stored); trust grows with active days
CREATE TABLE IF NOT EXISTS devices (id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, ip_hash TEXT, created INTEGER, seen INTEGER, days INTEGER NOT NULL DEFAULT 1);
CREATE INDEX IF NOT EXISTS devices_ip ON devices(ip_hash, created);
