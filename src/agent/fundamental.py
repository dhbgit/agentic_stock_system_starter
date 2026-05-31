import json
import yaml
from pathlib import Path
import yfinance as yf
from google import genai
from google.genai import types

REPO_ROOT = Path(__file__).parent.parent.parent

def get_gemini_client():
    config_path = REPO_ROOT / "config.yaml"
    if not config_path.exists():
        return None
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    api_key = config.get("gemini_api_key")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def fetch_ticker_context(ticker_symbol):
    """Fetches fundamental parameters and recent news for context."""
    t = yf.Ticker(ticker_symbol)
    
    # 1. Fundamentals
    info = {}
    try:
        raw_info = t.info
        info = {
            "market_cap": raw_info.get("marketCap", "Unknown"),
            "trailing_pe": raw_info.get("trailingPE", "N/A"),
            "forward_pe": raw_info.get("forwardPE", "N/A"),
            "current_price": raw_info.get("currentPrice", "N/A"),
            "52_week_high": raw_info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": raw_info.get("fiftyTwoWeekLow", "N/A"),
            "price_to_book": raw_info.get("priceToBook", "N/A"),
            "profit_margins": raw_info.get("profitMargins", "N/A")
        }
    except Exception:
        pass

    # 2. News
    news_titles = []
    try:
        news = t.news[:5] # Limit to top 5 to save context/focus on recent
        for n in news:
            title = n.get("title", "")
            publisher = n.get("publisher", "")
            if title:
                news_titles.append(f"[{publisher}] {title}")
    except Exception:
        pass
        
    return info, news_titles

def run_fundamental_debate(ticker, technical_signal, technical_confidence, rationale):
    """
    Submits the technical signal to Gemini along with fundamental data,
    asking for a qualitative debate and sentiment score.
    """
    client = get_gemini_client()
    if not client:
        return {"sentiment_score": 0.5, "gemini_rationale": "Missing Gemini API Key."}

    info, news = fetch_ticker_context(ticker)
    
    news_text = "\n".join(news) if news else "No recent news available."
    
    prompt = f"""
You are an elite quantitative Fundamental Analyst at a top-tier hedge fund. 
Your proprietary Machine Learning technical model has just issued a tactical signal for {ticker}.

[TECHNICAL MODEL OUTPUT]
Signal: {technical_signal}
Confidence (Calibrated Win Rate): {technical_confidence}%
Mathematical Rationale: {rationale}

[FUNDAMENTAL DATA CONTEXT]
Current Price: {info.get('current_price')}
Market Cap: {info.get('market_cap')}
Trailing P/E: {info.get('trailing_pe')}
Forward P/E: {info.get('forward_pe')}
Price/Book Ratio: {info.get('price_to_book')}
Profit Margins: {info.get('profit_margins')}

[LATEST NEWS HEADLINES (48 HOURS)]
{news_text}

[YOUR TASK]
Your job is to debate the technical model. Take into consideration the current stock price, market cap, P/E ratios, and the context of the recent news against the technical '{technical_signal}' signal. 
Calculate a "Fundamental Sentiment Score" between 0.0 (Extreme Bearish) and 1.0 (Extreme Bullish). 
If the news is devastating but the math model says LONG, you must have a low sentiment score to veto the math. If fundamentals support the math, score it high (matching the direction).

Respond ONLY with a valid JSON object matching this schema exactly:
{{
  "sentiment_score": float (from 0.0 to 1.0),
  "gemini_rationale": "Max 2 sentences explaining your fundamental view against the technicals.",
  "consensus_reached": boolean (true if you agree with the model's direction, false if you strongly disagree)
}}
"""

    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
        )
        
        # Clean JSON markdown if model outputs it
        out = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(out)
        return {
            "sentiment_score": result.get("sentiment_score", 0.5),
            "gemini_rationale": result.get("gemini_rationale", "No response"),
            "consensus_reached": result.get("consensus_reached", True)
        }
    except Exception as e:
        return {"sentiment_score": 0.5, "gemini_rationale": f"API Error: {str(e)}", "consensus_reached": True}

if __name__ == "__main__":
    # Test block
    res = run_fundamental_debate("AAPL", "Long", 72.5, "Structural momentum is peaking.")
    print(json.dumps(res, indent=2))
