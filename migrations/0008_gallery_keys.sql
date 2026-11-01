-- each saved gallery version gets its own object key (two writers on the same base no longer share, and then delete,
-- one object); the row points at the current one, gallery_keys lists them for clean-up
ALTER TABLE ponds ADD COLUMN gkey TEXT;
CREATE TABLE IF NOT EXISTS gallery_keys (pond TEXT NOT NULL, ver INTEGER NOT NULL, key TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS gallery_keys_pond ON gallery_keys(pond, ver);
