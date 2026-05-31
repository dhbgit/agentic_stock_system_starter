import math

def calculate_kelly_fraction(win_prob, win_loss_ratio=1.0, half_kelly=True, max_position=0.25):
    """
    Calculates the optimal portfolio allocation using the Kelly Criterion.
    
    Formula: f* = p - (1-p)/b
    Where:
      p = probability of a win (calibrated y_prob)
      b = ratio of the size of the win to the size of the loss (assumed 1.0 for 1-day hold)
      
    Args:
        win_prob: Float between 0 and 1.
        win_loss_ratio: Reward/Risk ratio.
        half_kelly: If True, halves the allocation for risk management.
        max_position: Maximum % allocation allowed per trade.
        
    Returns:
        float: Recommended allocation percentage (0.0 to max_position).
    """
    # Kelly requires an edge. If win_prob <= 0.5, allocation is 0.
    if win_prob <= 0.50:
        return 0.0

    kelly_f = win_prob - ((1.0 - win_prob) / win_loss_ratio)
    
    if half_kelly:
        kelly_f = kelly_f / 2.0
        
    # Cap position size to respect portfolio concentration risk
    position = min(max(kelly_f, 0.0), max_position)
    
    return position

def get_recommended_position(y_prob, is_long=True):
    """
    Wrapper for easy API access.
    Returns the string format of the Kelly size.
    """
    if is_long:
        prob = y_prob
    else:
        prob = 1.0 - y_prob
        
    allocation = calculate_kelly_fraction(prob)
    return round(allocation * 100, 1)

def simulate_portfolio_pnl(preds_df, initial_capital=10000.0, avg_daily_move=0.015):
    """
    Simulates a paper portfolio running the Kelly sizing strategy historically.
    Assumes an average daily volatile move (up or down).
    
    Args:
        preds_df: Pandas DataFrame of the 'preds' table with y_true, y_pred, y_prob.
        initial_capital: Starting capital.
        avg_daily_move: Fixed approximate return per trade (e.g. 1.5%).
        
    Returns:
        dict: Summary of simulation performance.
    """
    capital = initial_capital
    
    # Filter out rows where the actual outcome is unknown (-1)
    df = preds_df[preds_df['y_true'] != -1].copy()
    
    if len(df) == 0:
        return {"total_return_pct": 0, "final_capital": capital, "trades": 0}

    # Group by date so we don't compound intra-day, but process days sequentially
    daily_groups = df.groupby('date')
    
    for date, day_trades in daily_groups:
        daily_allocations = []
        
        # 1. Collect all intended allocations for the day
        for _, trade in day_trades.iterrows():
            y_prob = trade['y_prob']
            y_pred = trade['y_pred']
            y_true = trade['y_true']
            is_long = bool(y_pred == 1)
            allocation_pct = get_recommended_position(y_prob, is_long) / 100.0
            
            if allocation_pct > 0:
                daily_allocations.append({
                    "alloc": allocation_pct,
                    "y_pred": y_pred,
                    "y_true": y_true
                })
        
        if not daily_allocations:
            continue
            
        # 2. Normalize if total > 100% (No margin trading in simulator)
        total_intended = sum(a['alloc'] for a in daily_allocations)
        scale = 1.0
        if total_intended > 1.0:
            scale = 1.0 / total_intended
            
        # 3. Apply trades (Using Simple Interest to prevent exponential compounding explosions)
        daily_pnl = 0.0
        for a in daily_allocations:
            final_alloc = a['alloc'] * scale
            invested_amount = initial_capital * final_alloc  # Base on initial, not compounding
            
            won_trade = (a['y_pred'] == a['y_true'])
            if won_trade:
                daily_pnl += (invested_amount * avg_daily_move)
            else:
                daily_pnl -= (invested_amount * avg_daily_move)
                
        # Apply day's results linearly
        capital += daily_pnl
        
        # 4. Global fallback to prevent runway math
        if capital > 1e12: 
            capital = 1e12
            break
        
    total_return_pct = ((capital - initial_capital) / initial_capital) * 100
    
    return {
        "start_date": str(df['date'].min())[:10],
        "end_date": str(df['date'].max())[:10],
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_return_pct": round(total_return_pct, 2),
        "total_trades": len(df)
    }

