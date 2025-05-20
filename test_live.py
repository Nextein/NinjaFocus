import pytest
import os
from datetime import datetime
from bot import (
    CONFIG,
    get_ohlcv,
    calculate_indicators,
    generate_chart,
    send_notification,
    setup_logging
)

@pytest.mark.live  # Mark this as a live test (run separately)
def test_live_chart_generation_and_notification():
    """End-to-end test: Fetch real data, generate chart, send to Telegram"""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting live chart generation test")
    
    # 1. Fetch real data
    try:
        logger.info("Fetching OHLCV data...")
        df = get_ohlcv('15min')
        assert not df.empty, "Fetched empty DataFrame"
        logger.info(f"Fetched {len(df)} candles")
        
        # 2. Calculate indicators
        logger.info("Calculating indicators...")
        df_with_indicators = calculate_indicators(df)
        assert df_with_indicators is not None, "Indicator calculation failed"
        assert 'rsi' in df_with_indicators.columns, "RSI not calculated"
        logger.info("Indicators calculated successfully")
        
        # 3. Generate chart
        logger.info("Generating chart...")
        chart_path = generate_chart(df_with_indicators, '15min', 'test')
        assert os.path.exists(chart_path), "Chart file not created"
        logger.info(f"Chart generated at {chart_path}")
        
        # 4. Send notification
        logger.info("Sending Telegram notification...")
        test_message = (
            "🔧 TEST CHART 🔧\n"
            "This is an automated test of the chart generation system.\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        send_notification(test_message, chart_path)
        logger.info("Notification sent successfully")
        
    except Exception as e:
        logger.error(f"Live test failed: {str(e)}")
        raise
    finally:
        # Cleanup
        if 'chart_path' in locals() and os.path.exists(chart_path):
            os.remove(chart_path)
        logger.info("Live test completed")

if __name__ == "__main__":
    # Run just this test directly
    test_live_chart_generation_and_notification()