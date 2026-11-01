-- ponds found to be one water body (the same fish seen in both, e.g. either side of a footbridge) point at the survivor
ALTER TABLE ponds ADD COLUMN merged_into TEXT;
