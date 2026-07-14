# Stripe + Railway setup for Thoth

This guide connects the Brain backend on Railway to Stripe Checkout, Stripe Tax, promotion codes, shipping, subscriptions, and the Customer Portal. Complete everything in Stripe **test mode first**. Prices and discounts live in Stripe; this repository stores only their IDs.

## 1. Revoke old credentials

The repository previously contained a Stripe test secret. In Stripe Workbench, roll/revoke that restricted or secret key before deploying. Never reuse it, even in test mode.

Also rotate any webhook signing secret copied into source or chat. A webhook secret is specific to one event destination and to test or live mode.

## 2. Create the Stripe catalogue

In Stripe Dashboard → Product catalogue, create:

1. **Thoth hardware** — one-time Price. Copy its `prod_...` and `price_...` IDs.
2. **Thoth Home** — recurring monthly and annual Prices.
3. **Thoth Pro** — recurring monthly and annual Prices.
4. **Thoth Research** — recurring monthly and annual Prices.

Set the amounts, currency, annual discounts, product descriptions, and tax codes in Stripe. Do not add amounts to Python or Portal source.

Keep a test-mode worksheet:

| Railway variable | Stripe value |
| --- | --- |
| `STRIPE_HARDWARE_PRODUCT_ID` | Hardware `prod_...` |
| `STRIPE_HARDWARE_PRICE_ID` | Hardware one-time `price_...` |
| `STRIPE_PRICE_ID_HOME_MONTHLY` | Home monthly `price_...` |
| `STRIPE_PRICE_ID_HOME_ANNUAL` | Home annual `price_...` |
| `STRIPE_PRICE_ID_PRO_MONTHLY` | Pro monthly `price_...` |
| `STRIPE_PRICE_ID_PRO_ANNUAL` | Pro annual `price_...` |
| `STRIPE_PRICE_ID_RESEARCH_MONTHLY` | Research monthly `price_...` |
| `STRIPE_PRICE_ID_RESEARCH_ANNUAL` | Research annual `price_...` |

## 3. Configure tax, shipping, promotions, and the portal

### Stripe Tax

Open Tax → Settings, select the business origin, choose the correct default tax category, and add registrations where the business is registered to collect tax. The backend creates Checkout Sessions with `automatic_tax.enabled=true`.

### Shipping

In Product catalogue → Shipping rates, create reusable rates for the hardware product. Copy each `shr_...` ID. Set the comma-separated IDs in `STRIPE_SHIPPING_RATE_IDS`.

Set `STRIPE_ALLOWED_SHIPPING_COUNTRIES` to comma-separated ISO two-letter codes, for example:

```text
CA,US
```

Hardware Checkout collects a shipping address. Subscription Checkout does not collect shipping.

### Coupons and promotion codes

Create coupons and customer-facing promotion codes in Product catalogue → Coupons. Configure duration, product eligibility, expiry, redemption cap, and active status in Stripe. Checkout enables customer-entered promotion codes; no coupon amount is hardcoded by Brain.

### Customer Portal

Open Settings → Billing → Customer portal and configure:

- payment-method updates;
- invoice-history access;
- cancellation rules;
- monthly/annual changes for Home, Pro, and Research;
- allowed upgrades and downgrades.

Save the portal configuration in both test and live mode.

## 4. Create the Railway services

Use one Railway project containing:

- a PostgreSQL service;
- the Brain service from the `Brain` repository;
- optionally the ResearchPortal service if it is also hosted on Railway.

For Brain, Railway detects the repository `Dockerfile`. Generate a public HTTPS domain under Brain → Settings → Networking. Railway exposes `RAILWAY_PUBLIC_DOMAIN`; the webhook URL will be:

```text
https://YOUR-BRAIN-DOMAIN/api/stripe/webhook
```

Reference the PostgreSQL service variable instead of copying credentials. Set Brain's `DATABASE_URL` to Railway's Postgres `DATABASE_URL` reference.

## 5. Configure Brain variables on Railway

Open Brain → Variables and add:

```text
DATABASE_URL=${{Postgres.DATABASE_URL}}
SECRET_KEY=<long-random-application-secret>
NEXTAUTH_URL=https://YOUR-PORTAL-DOMAIN

STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...

STRIPE_HARDWARE_PRODUCT_ID=prod_...
STRIPE_HARDWARE_PRICE_ID=price_...
STRIPE_PRICE_ID_HOME_MONTHLY=price_...
STRIPE_PRICE_ID_HOME_ANNUAL=price_...
STRIPE_PRICE_ID_PRO_MONTHLY=price_...
STRIPE_PRICE_ID_PRO_ANNUAL=price_...
STRIPE_PRICE_ID_RESEARCH_MONTHLY=price_...
STRIPE_PRICE_ID_RESEARCH_ANNUAL=price_...

STRIPE_SHIPPING_RATE_IDS=shr_...,shr_...
STRIPE_ALLOWED_SHIPPING_COUNTRIES=CA,US

STRIPE_SUCCESS_URL=https://YOUR-PORTAL-DOMAIN/checkout/success?session_id={CHECKOUT_SESSION_ID}
STRIPE_CANCEL_URL=https://YOUR-PORTAL-DOMAIN/checkout/cancel
STRIPE_PORTAL_RETURN_URL=https://YOUR-PORTAL-DOMAIN/settings
```

Use Railway sealed variables for secret values. Do not set Stripe secrets as `NEXT_PUBLIC_*`; those variables are included in browser bundles.

After saving variables, redeploy Brain and inspect the deployment log. In a Railway shell run:

```bash
python server/setup_stripe.py
```

It exits nonzero and lists any missing required identifiers.

## 6. Connect ResearchPortal to Brain

On the Portal deployment set one server-side variable:

```text
BACKEND_BASE_URL=https://YOUR-BRAIN-DOMAIN
```

The Portal proxy automatically appends `/api`. Do not set the value to a URL ending in an unrelated path. `NEXT_PUBLIC_BACKEND_URL` and `NEXT_PUBLIC_API_URL` are fallback names, but `BACKEND_BASE_URL` is preferred because it stays server-side.

Set the Portal's public domain in Brain:

```text
NEXTAUTH_URL=https://YOUR-PORTAL-DOMAIN
```

Redeploy Portal after changing its backend URL. Verify:

```bash
curl -i https://YOUR-BRAIN-DOMAIN/api/health
curl -i https://YOUR-PORTAL-DOMAIN/api/proxy/health
```

Both should reach the same Brain deployment. A `401` on an authenticated route is expected without a bearer token; `404`, `502`, or `503` indicates a URL or deployment problem.

## 7. Register the Stripe event destination

In Stripe Workbench → Webhooks, create an HTTPS event destination pointing to:

```text
https://YOUR-BRAIN-DOMAIN/api/stripe/webhook
```

Select at least:

```text
checkout.session.completed
checkout.session.async_payment_succeeded
checkout.session.async_payment_failed
customer.subscription.created
customer.subscription.updated
customer.subscription.deleted
invoice.payment_succeeded
invoice.payment_failed
charge.refunded
refund.created
```

Reveal the destination's test signing secret and set it as `STRIPE_WEBHOOK_SECRET` in Railway. Redeploy Brain. Stripe does not guarantee event order and can deliver duplicates, so redirects must never grant access or fulfill hardware; signed webhook state is authoritative.

## 8. Test the backend endpoints

Obtain a Brain JWT by signing in through Portal, then export it locally:

```bash
export BRAIN=https://YOUR-BRAIN-DOMAIN
export TOKEN='<Brain JWT>'
```

Create subscription Checkout Sessions:

```bash
curl -sS -X POST "$BRAIN/api/stripe/create-checkout-session?plan=home&billing_period=monthly" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool

curl -sS -X POST "$BRAIN/api/stripe/create-checkout-session?plan=research&billing_period=annual" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

Create hardware Checkout:

```bash
curl -sS -X POST "$BRAIN/api/stripe/create-hardware-checkout-session" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

Create a Customer Portal session after the account has a Stripe customer:

```bash
curl -sS -X POST "$BRAIN/api/stripe/billing-portal" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

Each response contains a Stripe-hosted `url`. Open it in a browser and complete the test transaction.

## 9. Test webhooks with Stripe CLI

For a local Brain server:

```bash
stripe login
stripe listen --forward-to localhost:8000/api/stripe/webhook
```

Put the temporary `whsec_...` printed by `stripe listen` in the **local process environment only**, restart Brain, and complete a real test Checkout Session. Use test card `4242 4242 4242 4242`, any future expiry, and any CVC.

Useful synthetic checks include:

```bash
stripe trigger customer.subscription.created
stripe trigger invoice.payment_succeeded
stripe trigger invoice.payment_failed
```

Then use Stripe Workbench to resend the same event and confirm no duplicate payment/order is created. Also resend events out of order and confirm the final subscription status matches Stripe.

## 10. Production cutover

1. Complete test checkout, promotion, tax, shipping, cancellation, failed-payment, refund, and portal-change scenarios.
2. Create or activate the equivalent live Products, Prices, shipping rates, coupons, portal configuration, and webhook destination.
3. Replace every `pk_test_`, `sk_test_`, test `price_`, test `shr_`, and test `whsec_` Railway value with its live-mode counterpart in one controlled deployment.
4. Run `python server/setup_stripe.py` in the deployed Brain shell.
5. Make a low-value live transaction and verify the Stripe event destination returns `2xx`.
6. Monitor Railway logs and Stripe Workbench deliveries. Retry failed events only after correcting the cause.

## Troubleshooting

### Checkout says “Invalid plan”

The requested `plan` must be `home`, `pro`, or `research`; `billing_period` must be `monthly` or `annual`; and the corresponding Railway Price variable must exist in the deployed service.

### Webhook signature is invalid

Confirm the request reaches Brain without body transformation and that `STRIPE_WEBHOOK_SECRET` belongs to this exact destination and mode. The API secret (`sk_...`) is not a webhook signing secret (`whsec_...`).

### Portal reaches the wrong backend

Set `BACKEND_BASE_URL`, redeploy Portal, and inspect `/api/proxy/health`. Remove stale fallback values if they point to an older Railway deployment.

### Tax or shipping is missing

Confirm Stripe Tax is enabled, the Product has a tax code, registrations are configured, allowed country codes are valid, and every `STRIPE_SHIPPING_RATE_IDS` entry is an active `shr_...` from the same Stripe mode.

### Settings save but Thoth does not change

Confirm the device is online and heartbeating. Brain increments the settings revision; Thoth receives the canonical revision on the next heartbeat or live chunk metadata update and applies it to the next chunk snapshot.

## Official references

- Stripe Checkout: https://docs.stripe.com/payments/checkout/quickstarts
- Stripe Tax with Checkout: https://docs.stripe.com/tax/checkout
- Stripe webhook security and ordering: https://docs.stripe.com/webhooks
- Stripe webhook-based fulfillment: https://docs.stripe.com/checkout/fulfillment
- Railway variables: https://docs.railway.com/variables/reference
- Railway deployment practices: https://docs.railway.com/overview/best-practices
