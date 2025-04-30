def get_sentiment_label(score):
    """Convert sentiment score to text label"""
    if score >= 0.8:
        return "Very Positive"
    elif score >= 0.6:
        return "Positive"
    elif score <= 0.2:
        return "Very Negative"
    elif score <= 0.4:
        return "Negative"
    else:
        return "Neutral"

def get_confidence_color(sentiment):
    """Get color based on sentiment label"""
    colors = {
        "Very Positive": "#2E8B57",  # Sea Green
        "Positive": "#4CAF50",  # Green
        "Neutral": "#FF9800",  # Orange
        "Negative": "#F44336",  # Red
        "Very Negative": "#B71C1C",  # Dark Red
    }
    return colors.get(sentiment, "#9E9E9E")  # Default to gray

def get_domain_options():
    """Get available analysis domains"""
    return [
        "General",
        "Customer Service",
        "Product Reviews",
        "Social Media",
        "News Articles",
        "Academic",
        "Medical",
        "Financial"
    ]

def format_confidence(value):
    """Format confidence value for display"""
    return f"{value:.2f}" if isinstance(value, float) else value
