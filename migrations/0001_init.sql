-- Koi ID central store: index rows here, images and big JSON in R2 (bucket koi-id-data). One pond = one fish-name set.
CREATE TABLE IF NOT EXISTS ponds (
  id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL, fish INTEGER, embed_tag TEXT, device TEXT, updated_at INTEGER);
CREATE TABLE IF NOT EXISTS samples (
  id TEXT PRIMARY KEY, pond TEXT NOT NULL, fish INTEGER, label TEXT, q REAL, len REAL, s REAL, view INTEGER, tag TEXT,
  hist TEXT, at INTEGER, session TEXT, device TEXT, img_key TEXT, updated_at INTEGER);
CREATE INDEX IF NOT EXISTS samples_pond ON samples(pond, fish);
CREATE TABLE IF NOT EXISTS frames (
  id TEXT PRIMARY KEY, pond TEXT NOT NULL, w INTEGER, h INTEGER, at INTEGER, session TEXT, device TEXT, img_key TEXT, dets_key TEXT, updated_at INTEGER);
CREATE INDEX IF NOT EXISTS frames_pond ON frames(pond, at);
