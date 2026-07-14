# Stripe deployment setup

Prices and discounts are controlled in Stripe and referenced by ID. Never put API keys or monetary amounts in source code.

1. In Stripe, create the Thoth hardware product and recurring Home, Pro, and Research prices for monthly and annual billing.
2. Enable Stripe Tax, configure allowed shipping countries and reusable shipping rates, and configure the Customer Portal for supported plan changes and cancellation.
3. Create promotion codes in Stripe with the required product eligibility, expiry, redemption cap, and activation status.
4. Store all values listed in `.env.example` as deployment secrets/configuration. Do not commit a populated `.env`.
5. Register `/api/stripe/webhook` and subscribe it to Checkout, payment, invoice, subscription, cancellation, refund, and dispute events. Store its signing secret as `STRIPE_WEBHOOK_SECRET`.
6. In test mode, forward events with `stripe listen --forward-to localhost:8000/api/stripe/webhook` and exercise duplicate and out-of-order fixtures before switching identifiers and keys to live mode.
7. Run `python server/setup_stripe.py` in the deployment environment to validate that all required identifiers are present.

Redirect success is informational only. Fulfillment and entitlements must be driven by signature-verified webhooks.
