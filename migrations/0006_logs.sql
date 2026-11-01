-- field diagnostics from devices (no personal data): browser, backend, crash stage, detection rate; read to fix device-specific failures
CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, device TEXT, at INTEGER, ev TEXT, data TEXT);
CREATE INDEX IF NOT EXISTS logs_at ON logs(at);
