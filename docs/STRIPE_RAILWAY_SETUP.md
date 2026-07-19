# Stripe + Railway setup for Thoth

## Short version: do these steps now

1. In Stripe Workbench, open webhook `we_1TtDcrROEVpJQ0fpam7PeQls` and copy its **test signing secret**.
2. In Railway → Brain → Variables, add `STRIPE_WEBHOOK_SECRET` with that value and confirm the price variables in the block below are present.
3. Add `SUPABASE_EMAIL_REDIRECT_URL=https://portal-three-rho.vercel.app/auth?verified=1`, then redeploy Brain.
4. In Supabase → Authentication → URL Configuration, set the Site URL to `https://portal-three-rho.vercel.app` and add `https://portal-three-rho.vercel.app/auth**` as an allowed redirect. Under Email provider, keep **Confirm email** enabled.
5. Test one registration, one $5 Home subscription, one plan change from Portal Settings, and one $500 hardware checkout in test mode. If all five work, repeat the same objects and variables in Stripe live mode.

That is the complete owner action list. The sections below are reference material and troubleshooting.

This guide connects the Brain backend on Railway to Stripe Checkout, Stripe Tax, promotion codes, shipping, subscriptions, and the Customer Portal. Complete everything in Stripe **test mode first**. Prices and discounts live in Stripe; this repository stores only their IDs.

## Current setup status (2026-07-14)

The following Stripe **test-mode** objects have already been created and Checkout-validated:

| Item | Amount | Stripe ID |
| --- | ---: | --- |
| Thoth hardware | $500 USD once | `price_1TtDcSROEVpJQ0fpKuhCeau3` |
| Home monthly | $5 USD/month | `price_1TtDcTROEVpJQ0fpCxZRklYH` |
| Home annual | $60 USD/year | `price_1TtDcTROEVpJQ0fpIOCAIklc` |
| Pro monthly | $10 USD/month | `price_1TtDcTROEVpJQ0fpv1kZzvoR` |
| Pro annual | $120 USD/year | `price_1TtDcUROEVpJQ0fpIaxnEo9x` |
| Research monthly | $20 USD/month | `price_1TtDcUROEVpJQ0fprpyCTVdX` |
| Research annual | $240 USD/year | `price_1TtDcVROEVpJQ0fpqmzahbj2` |

Annual prices currently use the requested 0% annual discount (12 monthly payments). Stripe Tax is active with a Canadian head office. The default test Customer Portal configuration is active, and test Checkout Sessions for hardware and Research annual were created and expired successfully.

The test webhook destination `we_1TtDcrROEVpJQ0fpam7PeQls` is registered at:

```text
https://web-production-d7d37.up.railway.app/api/stripe/webhook
```

### What the account owner must still do

Complete these steps in order. They require dashboard access and cannot safely be committed to Git:

1. Rotate the Stripe API key that was previously present in local configuration. Prefer a restricted API key with only the permissions Brain needs.
2. In Stripe Workbench → Webhooks, open `we_1TtDcrROEVpJQ0fpam7PeQls`, reveal its **test signing secret**, and copy it directly into Railway as `STRIPE_WEBHOOK_SECRET`. Do not paste it into source, an issue, or chat.
3. Add the complete test variable block from section 5 to the Brain Railway service. Use the newly rotated test API key, not an old key.
4. Confirm whether worldwide hardware shipping is operationally supported. Checkout accepts all Stripe-supported shipping addresses, but the business remains responsible for fulfillment, customs, export controls, and carrier coverage.
5. If shipping is not free, create up to five reusable test Shipping Rates and set their `shr_...` IDs in `STRIPE_SHIPPING_RATE_IDS`. Leaving it blank means Checkout collects the address without adding a shipping charge.
6. Add `BACKEND_BASE_URL=https://web-production-d7d37.up.railway.app` to the Vercel Portal project and redeploy it.
7. Redeploy Brain, check `/health`, run `python server/setup_stripe.py` in Railway, and complete the test purchases in section 8.
8. Only after every test passes, repeat the catalogue, prices, Customer Portal, and webhook setup in **live mode**. Test IDs cannot be used for real payments.

## Optional: connect the official Stripe MCP to Codex

Stripe hosts an OAuth MCP server; no local npm package or secret-key command line is required:

```bash
codex mcp add stripe --url https://mcp.stripe.com
codex mcp login stripe
codex mcp get stripe
```

Approve the browser OAuth prompt, then restart Codex or start a new session so its Stripe tools are discovered. Use OAuth or a restricted API key rather than passing an unrestricted `sk_...` key to a local MCP process.

## 1. Revoke old credentials

The repository previously contained a Stripe test secret. In Stripe Workbench, roll/revoke that restricted or secret key before deploying. Never reuse it, even in test mode.

Also rotate any webhook signing secret copied into source or chat. A webhook secret is specific to one event destination and to test or live mode.

## 2. Create the Stripe catalogue

In Stripe Dashboard → Product catalogue, create:

1. **Thoth hardware** — one-time Price. Copy its `prod_...` and `price_...` IDs.
2. **Thoth Home** — recurring monthly and annual Prices.
3. **Thoth Pro** — recurring monthly and annual Prices.
4. **Thoth Research** — recurring monthly and annual Prices.

Set the amounts, currency, annual discounts, product descriptions, and tax codes in Stripe. Do not add amounts to Python or Portal source. The current test catalogue and prices are recorded above; recreate equivalent objects in live mode during cutover.

Keep a test-mode worksheet:

| Railway variable | Stripe value |
| --- | --- |
| `STRIPE_HARDWARE_PRODUCT_ID` | `prod_UsuCLB9MGrpFSL` |
| `STRIPE_HARDWARE_PRICE_ID` | `price_1TtDcSROEVpJQ0fpKuhCeau3` |
| `STRIPE_PRICE_ID_HOME_MONTHLY` | `price_1TtDcTROEVpJQ0fpCxZRklYH` |
| `STRIPE_PRICE_ID_HOME_ANNUAL` | `price_1TtDcTROEVpJQ0fpIOCAIklc` |
| `STRIPE_PRICE_ID_PRO_MONTHLY` | `price_1TtDcTROEVpJQ0fpv1kZzvoR` |
| `STRIPE_PRICE_ID_PRO_ANNUAL` | `price_1TtDcUROEVpJQ0fpIaxnEo9x` |
| `STRIPE_PRICE_ID_RESEARCH_MONTHLY` | `price_1TtDcUROEVpJQ0fprpyCTVdX` |
| `STRIPE_PRICE_ID_RESEARCH_ANNUAL` | `price_1TtDcVROEVpJQ0fpqmzahbj2` |

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
- optionally the thothHUB service if it is also hosted on Railway.

For Brain, Railway detects the repository `Dockerfile`. Generate a public HTTPS domain under Brain → Settings → Networking. Railway exposes `RAILWAY_PUBLIC_DOMAIN`; the webhook URL will be:

```text
https://web-production-d7d37.up.railway.app/api/stripe/webhook
```

Reference the PostgreSQL service variable instead of copying credentials. Set Brain's `DATABASE_URL` to Railway's Postgres `DATABASE_URL` reference.

## 5. Configure Brain variables on Railway

Open Brain → Variables and add:

```text
DATABASE_URL=${{Postgres.DATABASE_URL}}
SECRET_KEY=<long-random-application-secret>
NEXTAUTH_URL=https://portal-three-rho.vercel.app
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_ANON_KEY=<Supabase publishable/anon key>
SUPABASE_SERVICE_KEY=<Supabase service-role key; Railway secret only>
SUPABASE_EMAIL_REDIRECT_URL=https://portal-three-rho.vercel.app/auth?verified=1

STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_SECRET_KEY=<new test restricted-or-secret key stored only in Railway>
STRIPE_WEBHOOK_SECRET=<signing secret from test endpoint we_1TtDcrROEVpJQ0fpam7PeQls>

STRIPE_HARDWARE_PRODUCT_ID=prod_UsuCLB9MGrpFSL
STRIPE_HARDWARE_PRICE_ID=price_1TtDcSROEVpJQ0fpKuhCeau3
STRIPE_PRICE_ID_HOME_MONTHLY=price_1TtDcTROEVpJQ0fpCxZRklYH
STRIPE_PRICE_ID_HOME_ANNUAL=price_1TtDcTROEVpJQ0fpIOCAIklc
STRIPE_PRICE_ID_PRO_MONTHLY=price_1TtDcTROEVpJQ0fpv1kZzvoR
STRIPE_PRICE_ID_PRO_ANNUAL=price_1TtDcUROEVpJQ0fpIaxnEo9x
STRIPE_PRICE_ID_RESEARCH_MONTHLY=price_1TtDcUROEVpJQ0fprpyCTVdX
STRIPE_PRICE_ID_RESEARCH_ANNUAL=price_1TtDcVROEVpJQ0fpqmzahbj2

STRIPE_SHIPPING_RATE_IDS=
STRIPE_ALLOWED_SHIPPING_COUNTRIES=<paste the validated Appendix A value>

STRIPE_SUCCESS_URL=https://portal-three-rho.vercel.app/checkout/success?session_id={CHECKOUT_SESSION_ID}
STRIPE_CANCEL_URL=https://portal-three-rho.vercel.app/checkout/cancel
STRIPE_PORTAL_RETURN_URL=https://portal-three-rho.vercel.app/settings
```

Use Railway sealed variables for secret values. Do not set Stripe secrets as `NEXT_PUBLIC_*`; those variables are included in browser bundles.

After saving variables, redeploy Brain and inspect the deployment log. In a Railway shell run:

```bash
python server/setup_stripe.py
```

It exits nonzero and lists any missing required identifiers.

## 6. Connect thothHUB to Brain

On the Portal deployment set one server-side variable:

```text
BACKEND_BASE_URL=https://web-production-d7d37.up.railway.app
```

The Portal proxy automatically appends `/api`. Do not set the value to a URL ending in an unrelated path. `NEXT_PUBLIC_BACKEND_URL` and `NEXT_PUBLIC_API_URL` are fallback names, but `BACKEND_BASE_URL` is preferred because it stays server-side.

Set the Portal's public domain in Brain:

```text
NEXTAUTH_URL=https://portal-three-rho.vercel.app
```

Redeploy Portal after changing its backend URL. Verify:

```bash
curl -i https://web-production-d7d37.up.railway.app/health
curl -i https://portal-three-rho.vercel.app/api/proxy/health
```

Both should reach the same Brain deployment. A `401` on an authenticated route is expected without a bearer token; `404`, `502`, or `503` indicates a URL or deployment problem.

## 7. Register the Stripe event destination

In Stripe Workbench → Webhooks, create an HTTPS event destination pointing to:

```text
https://web-production-d7d37.up.railway.app/api/stripe/webhook
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
export BRAIN=https://web-production-d7d37.up.railway.app
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

If the endpoint returns `503 Payment service not available`, Railway is still running a build without the Stripe Python SDK. Confirm the deployment includes `stripe>=15.0.0,<16.0.0` from `requirements-deploy.txt`, then rebuild without cache. If it returns `500 Webhook not configured`, add the endpoint-specific signing secret and redeploy. An unsigned probe should return `400 Invalid signature` only after both pieces are configured.

### Portal reaches the wrong backend

Set `BACKEND_BASE_URL`, redeploy Portal, and inspect `/api/proxy/health`. Remove stale fallback values if they point to an older Railway deployment.

### Tax or shipping is missing

Confirm Stripe Tax is enabled, the Product has a tax code, registrations are configured, allowed country codes are valid, and every `STRIPE_SHIPPING_RATE_IDS` entry is an active `shr_...` from the same Stripe mode.

### Settings save but Thoth does not change

Confirm the device is online and heartbeating. Brain increments the settings revision; Thoth receives the canonical revision on the next heartbeat or live chunk metadata update and applies it to the next chunk snapshot.

## Appendix A: validated Stripe Checkout shipping countries

This value was accepted by a Stripe test Checkout Session on 2026-07-14. It includes every address country/territory currently accepted by Stripe Checkout and excludes Stripe's documented unsupported codes. Operational shipping coverage is still the merchant's responsibility.

```text
AC,AD,AE,AF,AG,AI,AL,AM,AO,AQ,AR,AT,AU,AW,AX,AZ,BA,BB,BD,BE,BF,BG,BH,BI,BJ,BL,BM,BN,BO,BQ,BR,BS,BT,BV,BW,BY,BZ,CA,CD,CF,CG,CH,CI,CK,CL,CM,CN,CO,CR,CV,CW,CY,CZ,DE,DJ,DK,DM,DO,DZ,EC,EE,EG,EH,ER,ES,ET,FI,FJ,FK,FO,FR,GA,GB,GD,GE,GF,GG,GH,GI,GL,GM,GN,GP,GQ,GR,GS,GT,GU,GW,GY,HK,HN,HR,HT,HU,ID,IE,IL,IM,IN,IO,IQ,IS,IT,JE,JM,JO,JP,KE,KG,KH,KI,KM,KN,KR,KW,KY,KZ,LA,LB,LC,LI,LK,LR,LS,LT,LU,LV,LY,MA,MC,MD,ME,MF,MG,MK,ML,MM,MN,MO,MQ,MR,MS,MT,MU,MV,MW,MX,MY,MZ,NA,NC,NE,NG,NI,NL,NO,NP,NR,NU,NZ,OM,PA,PE,PF,PG,PH,PK,PL,PM,PN,PR,PS,PT,PY,QA,RE,RO,RS,RU,RW,SA,SB,SC,SD,SE,SG,SH,SI,SJ,SK,SL,SM,SN,SO,SR,SS,ST,SV,SX,SZ,TC,TD,TF,TG,TH,TJ,TK,TL,TM,TN,TO,TR,TT,TV,TW,TZ,UA,UG,UY,UZ,VA,VC,VE,VG,VN,VU,WF,WS,XK,YE,YT,ZA,ZM,ZW
```

## Official references

- Stripe Checkout: https://docs.stripe.com/payments/checkout/quickstarts
- Stripe Tax with Checkout: https://docs.stripe.com/tax/checkout
- Stripe webhook security and ordering: https://docs.stripe.com/webhooks
- Stripe webhook-based fulfillment: https://docs.stripe.com/checkout/fulfillment
- Stripe MCP: https://docs.stripe.com/mcp
- Codex MCP configuration: https://developers.openai.com/codex/mcp
- Railway variables: https://docs.railway.com/variables/reference
- Railway deployment practices: https://docs.railway.com/overview/best-practices
