#!/usr/bin/env python3
"""Validate deployment-provided Stripe identifiers without creating prices."""

import os
import sys


REQUIRED = (
    "STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET", "STRIPE_HARDWARE_PRICE_ID",
    "STRIPE_PRICE_ID_HOME_MONTHLY", "STRIPE_PRICE_ID_HOME_ANNUAL",
    "STRIPE_PRICE_ID_PRO_MONTHLY", "STRIPE_PRICE_ID_PRO_ANNUAL",
    "STRIPE_PRICE_ID_RESEARCH_MONTHLY", "STRIPE_PRICE_ID_RESEARCH_ANNUAL",
)


def main() -> int:
    missing = [name for name in REQUIRED if not os.getenv(name)]
    if missing:
        print("Missing deployment configuration: " + ", ".join(missing), file=sys.stderr)
        return 1
    print("Stripe deployment configuration is present. Products, prices, tax, shipping, and discounts remain authoritative in Stripe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
