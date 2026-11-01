-- R2 is dropped (30 Oct 2026): every object it held (pond galleries, sample/frame images, frame dets) lives here.
-- A key keeps its old object-key string, so ponds.gkey, gallery_keys.key, samples.img_key and frames.img_key/dets_key
-- stay valid. An object larger than one row allows is split into parts 0, 1, 2 ... read back in part order.
CREATE TABLE IF NOT EXISTS blobs (key TEXT NOT NULL, part INTEGER NOT NULL, type TEXT, data BLOB NOT NULL, at INTEGER, PRIMARY KEY (key, part));
