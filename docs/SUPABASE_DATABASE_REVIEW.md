# Supabase database review

Review date: 2026-07-14. The review used a read-only PostgreSQL transaction and inspected schema metadata and row counts, not user record contents.

## Current state

- Supabase Auth contains zero users; the legacy Brain registration table contains 26 users.
- Product data currently includes 9 devices, 8,220 device inventory rows, 4,457 uploaded-file rows, 117 assistant query rows, and 3 preprocessing pipelines.
- Most training, federation, labs, folders, and organization tables are empty.
- Active product queries need composite or partial indexes for device activity, physical-device minutes, cloud history, Stripe invoice idempotency, and verified account lookup.
- The legacy application user table had no email or Supabase Auth identity columns even though Stripe requires an account email.

## Implemented simplification

Brain now owns one small idempotent product-core schema initializer. It adds only:

- `user_account.email`, `email_verified`, and `supabase_auth_user_id`;
- unique identity and payment-invoice indexes;
- active-device, on-device-minute, and cloud-history partial indexes.

New registrations use Supabase email confirmation, while Brain continues issuing its existing account-scoped JWT after confirmation. This is an incremental bridge that avoids breaking currently deployed device tokens.

## Data reset boundary

The reset is intentionally not executed until the owner chooses whether legacy `user_account` rows—especially the admin login—must be preserved. Supabase Auth is already empty. Once confirmed, reset application data in a single transaction, retain schema and migrations, reset sequences, then create the first verified owner account through Portal registration.

Do not truncate `auth.schema_migrations`, Supabase internal schemas, storage metadata, or migration history.
