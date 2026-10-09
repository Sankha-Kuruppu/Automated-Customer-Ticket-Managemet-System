"""Fixed label sets and the category -> team routing table from the API contract."""

CATEGORIES = [
    "payment_refund",
    "ride_trip_issue",
    "lost_item",
    "order_missing_wrong",
    "delivery_delay",
    "food_quality",
    "account_promo",
    "safety_conduct",
    "app_technical",
    "general_inquiry",
    "spam_irrelevant",
]

CATEGORY_TO_TEAM = {
    "payment_refund": "Payments & Refunds",
    "ride_trip_issue": "Ride Operations",
    "lost_item": "Lost & Found",
    "order_missing_wrong": "Food Operations",
    "delivery_delay": "Delivery Operations",
    "food_quality": "Restaurant Quality",
    "account_promo": "Account Services",
    "safety_conduct": "Trust & Safety",
    "app_technical": "Tech Support",
    "general_inquiry": "Front-line Support",
    "spam_irrelevant": "Auto-close / Spam Filter",
}

CHANNELS = ["email", "chat", "call_transcript"]

NO_SECONDARY = "none"
