-- central gallery version: a merge is accepted only on top of the version it was built from (no silent overwrite)
ALTER TABLE ponds ADD COLUMN ver INTEGER NOT NULL DEFAULT 0;
