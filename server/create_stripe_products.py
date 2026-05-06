#!/usr/bin/env python3
"""Create Stripe products directly."""

import stripe

# Your secret key from .env
stripe.api_key = "sk_test_51TTt3YROEVpJQ0fpIKcerLkZdBTnaPqhVuFg6eWDrwvJW0dMkpi3hcn5i1ow90zgx4CVKR7S65s1RcD3gihRmPGM00o8Cmhv1D"

print("Creating Stripe products...")

# Researcher Plan
try:
    product = stripe.Product.create(
        name="Researcher Plan",
        description="Advanced features for researchers",
        metadata={"plan_id": "researcher"}
    )
    price = stripe.Price.create(
        product=product.id,
        unit_amount=2900,  # $29.00
        currency="usd",
        recurring={"interval": "month"},
        metadata={"plan_id": "researcher"}
    )
    print(f"✅ Researcher Plan created:")
    print(f"   Product ID: {product.id}")
    print(f"   Price ID: {price.id}")
    print(f"   Add to .env: STRIPE_PRICE_ID_RESEARCHER={price.id}")
except Exception as e:
    print(f"❌ Error creating Researcher Plan: {e}")

print()

# Organization Plan
try:
    product = stripe.Product.create(
        name="Organization Plan",
        description="Full features for organizations and teams",
        metadata={"plan_id": "organization"}
    )
    price = stripe.Price.create(
        product=product.id,
        unit_amount=9900,  # $99.00
        currency="usd",
        recurring={"interval": "month"},
        metadata={"plan_id": "organization"}
    )
    print(f"✅ Organization Plan created:")
    print(f"   Product ID: {product.id}")
    print(f"   Price ID: {price.id}")
    print(f"   Add to .env: STRIPE_PRICE_ID_ORGANIZATION={price.id}")
except Exception as e:
    print(f"❌ Error creating Organization Plan: {e}")

print("\nNext steps:")
print("1. Add the price IDs above to your .env file")
print("2. Set up webhook at: https://your-domain.com/api/stripe/webhook")
print("3. Test the checkout flow")
