"""Stripe webhook endpoints for payment processing."""

import json
import logging
import os
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import Response
import stripe

from server.db import get_db, User, Payment
from server.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stripe", tags=["stripe"])

# Configure Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# Price IDs (update these after running setup_stripe.py)
PRICE_IDS = {
    "researcher": os.getenv("STRIPE_PRICE_ID_RESEARCHER"),
    "organization": os.getenv("STRIPE_PRICE_ID_ORGANIZATION"),
}

PLAN_PRICES = {
    "researcher": 2900,  # $29.00 in cents
    "organization": 9900,  # $99.00 in cents
}


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not WEBHOOK_SECRET:
        logger.error("Stripe webhook secret not configured")
        raise HTTPException(status_code=500, detail="Webhook not configured")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, WEBHOOK_SECRET)
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle the event
    if event.type == "customer.subscription.created":
        await handle_subscription_created(event.data.object)
    elif event.type == "customer.subscription.updated":
        await handle_subscription_updated(event.data.object)
    elif event.type == "customer.subscription.deleted":
        await handle_subscription_deleted(event.data.object)
    elif event.type == "invoice.payment_succeeded":
        await handle_payment_succeeded(event.data.object)
    else:
        logger.info(f"Unhandled event type: {event.type}")

    return Response(status_code=200)


async def handle_subscription_created(subscription):
    """Handle new subscription creation."""
    db = next(get_db())
    try:
        customer_id = subscription.customer
        price_id = subscription.items.data[0].price.id
        
        # Find which plan this is
        plan = None
        for plan_name, pid in PRICE_IDS.items():
            if pid == price_id:
                plan = plan_name
                break
        
        if not plan:
            logger.error(f"Unknown price ID: {price_id}")
            return
        
        # Find user by Stripe customer ID
        user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
        if not user:
            logger.error(f"User not found for customer: {customer_id}")
            return
        
        # Update user
        user.plan = plan
        user.stripe_subscription_id = subscription.id
        user.plan_expires_at = subscription.current_period_end
        
        db.commit()
        logger.info(f"Updated user {user.username} to plan {plan}")
        
    except Exception as e:
        logger.error(f"Error handling subscription created: {e}")
        db.rollback()
    finally:
        db.close()


async def handle_subscription_updated(subscription):
    """Handle subscription updates."""
    db = next(get_db())
    try:
        customer_id = subscription.customer
        user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
        
        if user:
            user.plan_expires_at = subscription.current_period_end
            db.commit()
            logger.info(f"Updated subscription for user {user.username}")
            
    except Exception as e:
        logger.error(f"Error handling subscription updated: {e}")
        db.rollback()
    finally:
        db.close()


async def handle_subscription_deleted(subscription):
    """Handle subscription cancellation."""
    db = next(get_db())
    try:
        customer_id = subscription.customer
        user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
        
        if user:
            user.plan = "free"
            user.stripe_subscription_id = None
            user.plan_expires_at = None
            db.commit()
            logger.info(f"Cancelled subscription for user {user.username}")
            
    except Exception as e:
        logger.error(f"Error handling subscription deleted: {e}")
        db.rollback()
    finally:
        db.close()


async def handle_payment_succeeded(invoice):
    """Handle successful payment."""
    db = next(get_db())
    try:
        customer_id = invoice.customer
        subscription_id = invoice.subscription
        
        user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
        if not user:
            return
        
        # Create payment record
        payment = Payment(
            user_id=user.userId,
            stripe_payment_intent=invoice.payment_intent,
            stripe_invoice_id=invoice.id,
            amount=invoice.total,
            currency=invoice.currency,
            plan=user.plan,
            status="succeeded"
        )
        db.add(payment)
        db.commit()
        logger.info(f"Recorded payment for user {user.username}: ${invoice.total/100}")
        
    except Exception as e:
        logger.error(f"Error handling payment succeeded: {e}")
        db.rollback()
    finally:
        db.close()


@router.post("/create-checkout-session")
async def create_checkout_session(
    plan: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a Stripe checkout session for plan upgrade."""
    if plan not in PRICE_IDS or not PRICE_IDS[plan]:
        raise HTTPException(status_code=400, detail="Invalid plan")
    
    try:
        # Get or create Stripe customer
        if not current_user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=f"{current_user.username}@example.com",
                metadata={"user_id": current_user.userId}
            )
            current_user.stripe_customer_id = customer.id
            db.commit()
        
        # Create checkout session
        session = stripe.checkout.Session.create(
            customer=current_user.stripe_customer_id,
            payment_method_types=["card"],
            line_items=[{
                "price": PRICE_IDS[plan],
                "quantity": 1,
            }],
            mode="subscription",
            success_url=os.getenv("STRIPE_SUCCESS_URL", "http://localhost:3000/settings?success=true"),
            cancel_url=os.getenv("STRIPE_CANCEL_URL", "http://localhost:3000/settings?cancel=true"),
            metadata={"plan": plan, "user_id": current_user.userId}
        )
        
        return {"url": session.url}
        
    except Exception as e:
        logger.error(f"Error creating checkout session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")
