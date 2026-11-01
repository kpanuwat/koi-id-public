-- green/red flags and capture info: grade = gold (owner confirmed) | silver | bronze; cam/quality = JSON for analysis only
ALTER TABLE samples ADD COLUMN grade TEXT;
ALTER TABLE samples ADD COLUMN cam TEXT;
ALTER TABLE frames ADD COLUMN quality TEXT;
