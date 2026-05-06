#!/usr/bin/env python3
"""Setup Stripe products and prices for ThothCraft plans."""

import os
import sys
import stripe

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv('../.env')  # Load from parent directory
except ImportError:
    pass

# Configure Stripe with your secret key
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
if not stripe.api_key:
    print("❌ STRIPE_SECRET_KEY not set in environment")
    sys.exit(1)

# Plan configuration
PLANS = {
    "researcher": {
        "name": "Researcher Plan",
        "description": "Advanced features for researchers",
        "price": 2900,  # $29.00 in cents
        "interval": "month"
    },
    "organization": {
        "name": "Organization Plan", 
        "description": "Full features for organizations and teams",
        "price": 9900,  # $99.00 in cents
        "interval": "month"
    }
}

def create_products():
    """Create Stripe products and prices."""
    print("[stripe] Creating products and prices...")
    
    for plan_id, config in PLANS.items():
        try:
            # Check if product already exists
            existing = stripe.Product.list(limit=100, active=True)
            product = None
            for p in existing.auto_paging_iter():
                if p.metadata.get("plan_id") == plan_id:
                    product = p
                    break
            
            if product:
                print(f"  ✅ Product for {plan_id} already exists: {product.id}")
            else:
                # Create new product
                product = stripe.Product.create(
                    name=config["name"],
                    description=config["description"],
                    metadata={"plan_id": plan_id}
                )
                print(f"  ✅ Created product for {plan_id}: {product.id}")
            
            # Create price
            price = stripe.Price.create(
                product=product.id,
                unit_amount=config["price"],
                currency="usd",
                recurring={"interval": config["interval"]},
                metadata={"plan_id": plan_id}
            )
            print(f"  ✅ Created price for {plan_id}: ${config['price']/100:.2f}/{config['interval']} (ID: {price.id})")
            
            # Store for reference
            print(f"     → Add to your .env: STRIPE_PRICE_ID_{plan_id.upper()}={price.id}")
            
        except stripe.error.StripeError as e:
            print(f"  ❌ Error creating {plan_id}: {e}")

def setup_webhook():
    """Instructions for webhook setup."""
    print("\n[stripe] Webhook Setup:")
    print("1. Go to Stripe Dashboard → Developers → Webhooks")
    print("2. Add endpoint: https://your-domain.com/api/stripe/webhook")
    print("3. Select events: customer.subscription.created, customer.subscription.updated, customer.subscription.deleted, invoice.payment_succeeded")
    print("4. Copy the webhook secret and add to .env as STRIPE_WEBHOOK_SECRET")

if __name__ == "__main__":
    create_products()
    setup_webhook()
    print("\n✅ Stripe setup complete!")
    print("\nNext steps:")
    print("1. Add the price IDs to your environment")
    print("2. Implement the webhook endpoint")
    print("3. Update Settings page with Stripe checkout")
