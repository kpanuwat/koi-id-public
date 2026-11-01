-- highest fish number handed out per pond: a gallery may never go back below it (numbers are never reused)
ALTER TABLE ponds ADD COLUMN next_fish INTEGER NOT NULL DEFAULT 0;
