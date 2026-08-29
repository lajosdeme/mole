-- The citation receipt, stored so it is paid for once.
--
-- Verification re-reads every cited source and checks the quote is still in it.
-- That is one fetch per source — the only part of an export that is not free, and
-- the part a consumer rendering the same session twice must not pay twice.
--
-- Opaque JSON rather than columns. The store has no business knowing the shape of
-- a receipt, the same way it carries report_md without parsing markdown; whatever
-- writes it owns the schema, and a shape change here is not a migration.
--
-- A column on sessions rather than a table, for the reasons report_md is: one
-- session has at most one receipt, it is written after the session finishes, and
-- nothing queries receipts independently of the session that produced them.
--
-- Empty means never verified. That is distinct from a receipt reporting zero
-- verified quotes, and a consumer must be able to tell them apart — an absent
-- check and a failed one are not the same fact.

ALTER TABLE sessions ADD COLUMN receipt_json TEXT NOT NULL DEFAULT '';
