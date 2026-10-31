-- Short keyword search queries for a lead, as a JSON array; empty means search the lead text.
ALTER TABLE leads ADD COLUMN queries TEXT NOT NULL DEFAULT '';
