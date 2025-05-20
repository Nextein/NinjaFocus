import pytest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
import logging

# Import the functions from your bot script
from bot import (calculate_required_candles, get_ohlcv, resample_data, 
                calculate_indicators, check_conditions, generate_chart,
                analyze_timeframe, CONFIG)

# Setup test data
def create_test_data(rows=100, start_price=100):
    """Create synthetic OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=rows, freq='5min')
    prices = np.cumsum(np.random.randn(rows) + start_price)
    volume = np.random.randint(100, 1000, size=rows)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.rand(rows),
        'low': prices - np.random.rand(rows),
        'close': prices,
        'volume': volume
    }, index=dates)
    
    return df

# Fixtures
@pytest.fixture
def test_data():
    return create_test_data(rows=200)

@pytest.fixture
def indicators_data(test_data):
    return calculate_indicators(test_data)

# Test cases
def test_calculate_required_candles():
    """Test the candle requirement calculation"""
    # Test for 5min timeframe
    assert calculate_required_candles('5min') > 2000  # Should be enough for 7 days
    
    # Test for 12h timeframe
    assert 20 < calculate_required_candles('12h') < 50
    
    # Test max_fetch_limit enforcement
    CONFIG['max_fetch_limit'] = 100
    assert calculate_required_candles('5min') == 100
    CONFIG['max_fetch_limit'] = 1000  # Reset

@patch('bot.exchange.fetch_ohlcv')
def test_get_ohlcv(mock_fetch):
    """Test OHLCV data fetching"""
    # Setup mock return
    mock_data = [
        [int(datetime.now().timestamp())*1000, 100, 101, 99, 100.5, 1000],
        [int((datetime.now() - timedelta(minutes=5)).timestamp())*1000, 100.5, 101.5, 99.5, 101, 1500]
    ]
    mock_fetch.return_value = mock_data
    
    # Test fetching
    df = get_ohlcv('5min')
    assert len(df) == 2
    assert 'close' in df.columns
    assert df.index[0] > df.index[1]  # Should be sorted newest first

def test_resample_data(test_data):
    """Test timeframe resampling"""
    # Test resampling to 15min
    df_15min = resample_data(test_data, '15min')
    assert len(df_15min) <= len(test_data) / 3  # Roughly 1/3 as many candles
    
    # Test resampling to 1h
    df_1h = resample_data(test_data, '1h')
    assert len(df_1h) <= len(test_data) / 12
    
    # Test with insufficient data
    small_data = create_test_data(rows=10)
    assert resample_data(small_data, '1h') is None

def test_calculate_indicators(test_data):
    """Test indicator calculation"""
    df = calculate_indicators(test_data)
    
    # Check all required columns exist
    assert 'middle_band' in df.columns
    assert 'upper_band' in df.columns
    assert 'lower_band' in df.columns
    assert 'rsi' in df.columns
    
    # Check RSI values are within bounds
    assert 0 <= df['rsi'].min() <= 100
    assert 0 <= df['rsi'].max() <= 100
    
    # Test with insufficient data
    small_data = create_test_data(rows=10)
    assert calculate_indicators(small_data) is None

def test_check_conditions(indicators_data):
    """Test signal detection"""
    # Test no signal
    signal, pct = check_conditions(indicators_data)
    assert signal is None or signal in ['overbought', 'oversold']
    
    # Create overbought condition
    ob_data = indicators_data.copy()
    ob_data.iloc[-1, ob_data.columns.get_loc('close')] = ob_data['upper_band'].iloc[-1] * 1.1
    ob_data.iloc[-1, ob_data.columns.get_loc('rsi')] = 75
    
    signal, pct = check_conditions(ob_data)
    assert signal == 'overbought'
    assert pct > 0
    
    # Create oversold condition
    os_data = indicators_data.copy()
    os_data.iloc[-1, os_data.columns.get_loc('close')] = os_data['lower_band'].iloc[-1] * 0.9
    os_data.iloc[-1, os_data.columns.get_loc('rsi')] = 25
    
    signal, pct = check_conditions(os_data)
    assert signal == 'oversold'
    assert pct > 0

def test_generate_chart(indicators_data):
    """Test chart generation"""
    # Setup test
    test_file = 'test_chart.png'
    
    # Cleanup if file exists
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Test normal generation
    filename = generate_chart(indicators_data, '5min', 'overbought')
    assert os.path.exists(filename)
    
    # Verify chart contents
    try:
        img = plt.imread(filename)
        assert img.shape[0] > 0  # Has height
        assert img.shape[1] > 0  # Has width
    except Exception as e:
        pytest.fail(f"Generated image is invalid: {str(e)}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)
    
    # Test with insufficient data
    with pytest.raises(Exception):
        generate_chart(pd.DataFrame(), '5min', 'overbought')

@patch('bot.send_notification')
@patch('bot.generate_chart')
def test_analyze_timeframe(mock_chart, mock_notify, indicators_data):
    """Test timeframe analysis"""
    # Setup mocks
    mock_chart.return_value = 'test_chart.png'
    
    # Test with good data
    with patch('bot.calculate_indicators', return_value=indicators_data):
        analyze_timeframe('5min')
        assert mock_notify.called
    
    # Test with no signal
    no_signal_data = indicators_data.copy()
    no_signal_data['close'] = no_signal_data['middle_band']
    no_signal_data['rsi'] = 50
    
    with patch('bot.calculate_indicators', return_value=no_signal_data):
        analyze_timeframe('5min')
        assert not mock_notify.called
    
    # Test with insufficient data
    with patch('bot.calculate_indicators', return_value=None):
        analyze_timeframe('5min')
        assert not mock_notify.called

@patch('bot.analyze_timeframe')
def test_job(mock_analyze):
    """Test main job execution"""
    from bot import job
    
    # Test normal execution
    job()
    assert mock_analyze.call_count == len(CONFIG['timeframe_priority'])
    
    # Test with exception
    mock_analyze.side_effect = Exception("Test error")
    with pytest.raises(Exception):
        job()

def test_image_generation_quality(indicators_data):
    """Special test for image generation issues"""
    test_cases = [
        ('normal', indicators_data, 'overbought'),
        ('empty_indicators', indicators_data.drop(['upper_band', 'lower_band', 'rsi'], axis=1), 'oversold'),
        ('partial_data', indicators_data.iloc[:10], 'overbought'),
        ('nan_values', indicators_data.mask(indicators_data > 100), 'oversold')
    ]
    
    for name, data, signal in test_cases:
        filename = None
        try:
            filename = generate_chart(data, '5min', signal)
            assert os.path.exists(filename), f"{name}: File not created"
            
            # Verify image can be read and has content
            img = plt.imread(filename)
            assert img.size > 0, f"{name}: Empty image"
            assert img.shape[2] == 4, f"{name}: Incorrect color channels"  # RGBA
            
            # Check for common image generation issues
            if name == 'normal':
                # Check if indicators are visible (sample center pixel)
                center_y = img.shape[0] // 2
                center_x = img.shape[1] // 2
                assert not np.all(img[center_y, center_x] == 1), f"{name}: Blank image area"
            
        except Exception as e:
            pytest.fail(f"Image generation failed for {name}: {str(e)}")
        finally:
            if filename and os.path.exists(filename):
                os.remove(filename)

def test_telegram_notification_images():
    """Test that generated images are valid for Telegram"""
    from bot import send_notification, generate_chart
    
    test_data = create_test_data()
    indicators_data = calculate_indicators(test_data)
    
    # Generate test chart
    filename = generate_chart(indicators_data, '5min', 'overbought')
    
    # Mock requests.post
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Test sending
        try:
            send_notification("Test message", filename)
            assert mock_post.called
            
            # Verify the file was sent correctly
            args, kwargs = mock_post.call_args
            assert 'files' in kwargs
            assert 'photo' in kwargs['files']
            
        except Exception as e:
            pytest.fail(f"Telegram notification failed: {str(e)}")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == "__main__":
    pytest.main(['-v', '--cov=bot', '--cov-report=html'])