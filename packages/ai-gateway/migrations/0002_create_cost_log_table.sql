-- Migration: Create cost_log table for AI spend tracking
-- Run with: wrangler d1 execute screenpipe-usage --file=./migrations/0002_create_cost_log_table.sql

CREATE TABLE IF NOT EXISTS cost_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL DEFAULT (datetime('now')),
  device_id TEXT,
  user_id TEXT,
  tier TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  input_tokens INTEGER,
  output_tokens INTEGER,
  estimated_cost_usd REAL,
  endpoint TEXT NOT NULL,
  stream INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cost_log_timestamp ON cost_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_cost_log_day ON cost_log(date(timestamp));
CREATE INDEX IF NOT EXISTS idx_cost_log_model ON cost_log(model);
