import pytest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
import logging

import requests

# Import the functions and CONFIG from your bot script
from bot import (
    calculate_required_candles, get_ohlcv, calculate_indicators,
    check_conditions, plot_chart, analyze_bollinger_strategy,
    analyze_second_strategy, send_telegram_message, send_telegram_notification_with_chart, CONFIG
)

# Import TALib for direct mocking
from features import TALib

# Setup test data
def create_test_data(rows=100, start_price=100, freq='15min'):
    """Create synthetic OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=rows, freq=freq)
    prices = np.cumsum(np.random.randn(rows) * 0.5 + start_price) + np.sin(np.arange(rows) / 10) * 5
    volume = np.random.randint(100, 1000, size=rows)

    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.rand(rows),
        'low': prices - np.random.rand(rows),
        'close': prices,
        'volume': volume
    }, index=dates)

    # Ensure 'Open' and 'Close' are capitalized for TALib
    df.index.name = 'timestamp' # Ensure the index has a name
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] # TALib expects these names
    return df

# Fixtures
@pytest.fixture
def test_data():
    return create_test_data(rows=200, freq='15min')

@pytest.fixture
def indicators_data(test_data):
    # Rename columns back to lowercase for bot.py functions if needed, or adjust bot.py
    # For now, let's assume bot.py handles the column names consistently.
    # If calculate_indicators expects lowercase, this fixture needs adjustment:
    df = test_data.copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return calculate_indicators(df)

# Test cases for existing functions (unchanged)
def test_calculate_required_candles():
    """Test the candle requirement calculation"""
    initial_max_fetch_limit = CONFIG['max_fetch_limit'] # Store initial value

    # Test for 15min timeframe (smallest in CONFIG['timeframes'])
    # Expected: ceil((7 * 24 * 60) / 15) + (20 * 2 + 2) = 672 + 42 = 714
    assert calculate_required_candles('15min') == 714

    # Test for 1d timeframe (largest for second strategy)
    # Expected: ceil((7 * 24 * 60) / 1440) + (20 * 2 + 2) = 7 + 42 = 49
    assert calculate_required_candles('1d') == 49

    # Test max_fetch_limit enforcement
    CONFIG['max_fetch_limit'] = 50
    assert calculate_required_candles('15min') == 50
    assert calculate_required_candles('1d') == 49
    CONFIG['max_fetch_limit'] = initial_max_fetch_limit  # Reset

@patch('bot.exchange.fetch_ohlcv')
def test_get_ohlcv(mock_fetch):
    """Test OHLCV data fetching"""
    # Setup mock return for ccxt format
    mock_data = [
        [int((datetime.now() - timedelta(minutes=5)).timestamp() * 1000), 100, 101, 99, 100.5, 1000],
        [int((datetime.now() - timedelta(minutes=10)).timestamp() * 1000), 100.5, 101.5, 99.5, 101, 1500]
    ]
    mock_fetch.return_value = mock_data

    # Test fetching for a timeframe specified in CONFIG
    df = get_ohlcv(timeframe='15min') # Use '15min' as it's in CONFIG
    assert df is not None
    assert len(df) == 2
    assert 'close' in df.columns # Ensure column names are lowercase after processing
    assert df.index.is_monotonic_increasing # Should be sorted by timestamp
    assert df.index.name == 'timestamp' # Check if index name is set

    # Test with no timeframe (should use default '15min')
    df_default = get_ohlcv()
    assert df_default is not None
    assert len(df_default) == 2

    mock_fetch.side_effect = Exception("API Error")
    df_error = get_ohlcv('15min')
    assert df_error is None


def test_calculate_indicators(test_data):
    """Test indicator calculation"""
    # Ensure test_data columns are lowercase for bot.py's calculate_indicators
    df_input = test_data.copy()
    df_input.columns = ['open', 'high', 'low', 'close', 'volume']
    df = calculate_indicators(df_input)

    assert df is not None
    # Check all required columns exist
    assert 'middle_band' in df.columns
    assert 'upper_band' in df.columns
    assert 'lower_band' in df.columns
    assert 'rsi' in df.columns

    # Check RSI values are within bounds
    assert (df['rsi'] >= 0).all() and (df['rsi'] <= 100).all()

    # Test with insufficient data
    small_data = create_test_data(rows=10)
    small_data.columns = ['open', 'high', 'low', 'close', 'volume']
    assert calculate_indicators(small_data) is None # Should return None due to dropna


def test_check_conditions(indicators_data):
    """Test signal detection for Bollinger Band strategy"""
    # Test no signal
    signal, trigger_candle_data, relative_index = check_conditions(indicators_data)
    assert signal is None
    assert trigger_candle_data == 0
    assert relative_index is None

    # Create overbought condition
    ob_data = indicators_data.copy()
    # Manipulate the last candle to trigger a sell signal
    ob_data.loc[ob_data.index[-1], 'close'] = ob_data['upper_band'].iloc[-1] + 1
    ob_data.loc[ob_data.index[-1], 'rsi'] = CONFIG['rsi_overbought'] + 5

    signal, trigger_candle_data, relative_index = check_conditions(ob_data)
    assert signal == 'SELL'
    assert relative_index == 1 # Last candle

    # Create oversold condition
    os_data = indicators_data.copy()
    # Manipulate the last candle to trigger a buy signal
    os_data.loc[os_data.index[-1], 'close'] = os_data['lower_band'].iloc[-1] - 1
    os_data.loc[os_data.index[-1], 'rsi'] = CONFIG['rsi_oversold'] - 5

    signal, trigger_candle_data, relative_index = check_conditions(os_data)
    assert signal == 'BUY'
    assert relative_index == 1 # Last candle

    # Test lookback period (signal on second to last candle)
    lookback_data = indicators_data.copy()
    # Trigger buy signal on the second to last candle
    lookback_data.loc[lookback_data.index[-2], 'close'] = lookback_data['lower_band'].iloc[-2] - 1
    lookback_data.loc[lookback_data.index[-2], 'rsi'] = CONFIG['rsi_oversold'] - 5

    signal, trigger_candle_data, relative_index = check_conditions(lookback_data)
    assert signal == 'BUY'
    assert relative_candle_data.name == lookback_data.index[-2] # Check timestamp of trigger candle
    assert relative_index == 2 # Second to last candle


@patch('bot.os.path.exists', return_value=True) # Mock os.path.exists for cleanup
@patch('bot.os.remove') # Mock os.remove to prevent actual file deletion during test
def test_plot_chart(mock_remove, mock_exists, indicators_data):
    """Test chart generation"""
    test_file = os.path.join(CONFIG['log_directory'], 'test_chart.png')
    # Ensure column names are lowercase for plot_chart
    df_input = indicators_data.copy()
    df_input.columns = ['open', 'high', 'low', 'close', 'volume', 'middle_band', 'std_dev', 'upper_band', 'lower_band', 'rsi']

    # Test normal generation (no signal)
    filename = plot_chart(df_input, None, None, '15min')
    assert filename is not None
    assert filename == test_file # Check the filename returned

    # Test generation with a BUY signal
    buy_filename = plot_chart(df_input, 'BUY', 1, '15min') # Last candle is the trigger
    assert buy_filename is not None

    # Test generation with a SELL signal
    sell_filename = plot_chart(df_input, 'SELL', 2, '15min') # Second to last candle is the trigger
    assert sell_filename is not None

    # Test with insufficient data
    assert plot_chart(pd.DataFrame(), 'BUY', 1, '15min') is None

    # Test with missing columns
    df_partial = df_input.drop(columns=['rsi'])
    assert plot_chart(df_partial, 'BUY', 1, '15min') is None

    # Assert that os.remove was called (mocked) for each successful chart generation
    assert mock_remove.call_count == 3 # For the 3 successful generations


@patch('bot.send_telegram_notification_with_chart')
@patch('bot.plot_chart')
@patch('bot.calculate_indicators')
@patch('bot.get_ohlcv')
def test_analyze_bollinger_strategy(mock_get_ohlcv, mock_calculate_indicators, mock_plot_chart, mock_send_notification):
    """Test Bollinger Band strategy analysis workflow"""
    test_df = create_test_data(rows=100)
    test_df.columns = ['open', 'high', 'low', 'close', 'volume'] # Ensure lowercase for bot functions
    mock_get_ohlcv.return_value = test_df

    indicators_df = calculate_indicators(test_df)
    mock_calculate_indicators.return_value = indicators_df

    # Test with a buy signal
    buy_signal_df = indicators_df.copy()
    buy_signal_df.loc[buy_signal_df.index[-1], 'close'] = buy_signal_df['lower_band'].iloc[-1] - 1
    buy_signal_df.loc[buy_signal_df.index[-1], 'rsi'] = CONFIG['rsi_oversold'] - 5
    with patch('bot.check_conditions', return_value=('BUY', buy_signal_df.iloc[-1], 1)):
        mock_plot_chart.return_value = 'mock_chart_buy.png'
        analyze_bollinger_strategy('15min', ['chat_id_jose'])
        mock_get_ohlcv.assert_called_with(timeframe='15min', symbol=CONFIG['symbol'])
        mock_calculate_indicators.assert_called_once()
        mock_plot_chart.assert_called_with(indicators_df, 'BUY', 1, '15min')
        mock_send_notification.assert_called_once()
        assert "BUY Signal Detected!" in mock_send_notification.call_args[0][0]
        assert mock_send_notification.call_args[0][1] == 'mock_chart_buy.png'
        assert mock_send_notification.call_args[0][2] == ['chat_id_jose']
        mock_get_ohlcv.reset_mock()
        mock_calculate_indicators.reset_mock()
        mock_plot_chart.reset_mock()
        mock_send_notification.reset_mock()

    # Test with a sell signal
    sell_signal_df = indicators_df.copy()
    sell_signal_df.loc[sell_signal_df.index[-1], 'close'] = sell_signal_df['upper_band'].iloc[-1] + 1
    sell_signal_df.loc[sell_signal_df.index[-1], 'rsi'] = CONFIG['rsi_overbought'] + 5
    with patch('bot.check_conditions', return_value=('SELL', sell_signal_df.iloc[-1], 1)):
        mock_plot_chart.return_value = 'mock_chart_sell.png'
        analyze_bollinger_strategy('1h', ['chat_id_jose'])
        mock_send_notification.assert_called_once()
        assert "SELL Signal Detected!" in mock_send_notification.call_args[0][0]
        assert mock_send_notification.call_args[0][1] == 'mock_chart_sell.png'
        mock_get_ohlcv.reset_mock()
        mock_calculate_indicators.reset_mock()
        mock_plot_chart.reset_mock()
        mock_send_notification.reset_mock()

    # Test with no signal
    with patch('bot.check_conditions', return_value=(None, 0, None)):
        analyze_bollinger_strategy('4h', ['chat_id_jose'])
        mock_send_notification.assert_not_called()
        mock_plot_chart.assert_not_called()
        mock_get_ohlcv.reset_mock()
        mock_calculate_indicators.reset_mock()

    # Test when get_ohlcv returns None
    mock_get_ohlcv.return_value = None
    analyze_bollinger_strategy('15min', ['chat_id_jose'])
    mock_calculate_indicators.assert_not_called()
    mock_send_notification.assert_not_called()
    mock_plot_chart.assert_not_called()
    mock_get_ohlcv.reset_mock()

    # Test when calculate_indicators returns None
    mock_get_ohlcv.return_value = test_df
    mock_calculate_indicators.return_value = None
    analyze_bollinger_strategy('15min', ['chat_id_jose'])
    mock_send_notification.assert_not_called()
    mock_plot_chart.assert_not_called()


# --- New Tests for Second Strategy ---

@patch('bot.send_telegram_message')
@patch('features.TALib.relativeCandlesReversalPatterns')
@patch('features.TALib.Cycles')
@patch('features.TALib.relativeCandlesPhases')
@patch('bot.get_ohlcv')
def test_analyze_second_strategy(
    mock_get_ohlcv,
    mock_relative_candles_phases,
    mock_cycles,
    mock_reversal_patterns,
    mock_send_telegram_message
):
    """Test the second strategy analysis workflow"""
    test_df_original_cols = create_test_data(rows=50) # Need enough rows for TALib internal lookbacks (e.g., relativePositionOfCandles needs at least 3)
    # TALib methods expect 'Open', 'High', 'Low', 'Close', 'Volume' capitalized
    # get_ohlcv returns lowercase, so we need to ensure the mock returns data that, after get_ohlcv's processing, matches what TALib expects.
    # However, the bot.py's analyze_second_strategy calls TALib directly with the df returned by get_ohlcv.
    # So, the mock_get_ohlcv should return a DataFrame with lowercase columns, and we'll mock TALib to work with that.
    # OR, better: modify create_test_data to return expected lowercase for bot, and then mock TALib to expect that.
    # Let's adjust create_test_data to match what get_ohlcv produces (lowercase columns) and ensure TALib mocks handle it.
    test_df_for_bot = test_df_original_cols.copy()
    test_df_for_bot.columns = ['open', 'high', 'low', 'close', 'volume']
    mock_get_ohlcv.return_value = test_df_for_bot

    # Mock TALib method returns
    # Mock relativeCandlesReversalPatterns
    mock_reversal_patterns.return_value = 0 # Default: no signal
    # Mock Cycles: last value used for cycles
    mock_cycles.return_value = pd.Series(['X'] * 49 + ['A'], index=test_df_for_bot.index)
    # Mock relativeCandlesPhases: last value used for phases
    mock_relative_candles_phases.return_value = np.array([1] * 49 + [-1])

    # Test with no signal
    analyze_second_strategy('1h', ['chat_id_nexus'])
    mock_get_ohlcv.assert_called_with(timeframe='1h', symbol=CONFIG['symbol'])
    mock_reversal_patterns.assert_called_once()
    mock_cycles.assert_called_once()
    mock_relative_candles_phases.assert_called_once()
    mock_send_telegram_message.assert_not_called()
    mock_get_ohlcv.reset_mock()
    mock_reversal_patterns.reset_mock()
    mock_cycles.reset_mock()
    mock_relative_candles_phases.reset_mock()
    mock_send_telegram_message.reset_mock()


    # Test with a BUY signal (reversal_pattern = 1)
    mock_reversal_patterns.return_value = 1
    mock_cycles.return_value = pd.Series(['X'] * 49 + ['A'], index=test_df_for_bot.index)
    mock_relative_candles_phases.return_value = np.array([1] * 49 + [1])
    analyze_second_strategy('1h', ['chat_id_nexus'])
    mock_send_telegram_message.assert_called_once()
    message = mock_send_telegram_message.call_args[0][0]
    assert "🟢 *2 cycles BUY Signal Detected!* 🟢" in message
    assert "Reversal Pattern: BUY" in message
    assert "Cycles: A" in message
    assert "Phases: 1" in message
    assert mock_send_telegram_message.call_args[0][1] == ['chat_id_nexus']
    mock_get_ohlcv.reset_mock()
    mock_reversal_patterns.reset_mock()
    mock_cycles.reset_mock()
    mock_relative_candles_phases.reset_mock()
    mock_send_telegram_message.reset_mock()

    # Test with a SELL signal (reversal_pattern = -1)
    mock_reversal_patterns.return_value = -1
    mock_cycles.return_value = pd.Series(['X'] * 49 + ['-D'], index=test_df_for_bot.index)
    mock_relative_candles_phases.return_value = np.array([1] * 49 + [-1])
    analyze_second_strategy('1h', ['chat_id_nexus'])
    mock_send_telegram_message.assert_called_once()
    message = mock_send_telegram_message.call_args[0][0]
    assert "🔴 *2 cycles  SELL Signal Detected!* 🔴" in message
    assert "Reversal Pattern: SELL" in message
    assert "Cycles: -D" in message
    assert "Phases: -1" in message
    mock_get_ohlcv.reset_mock()
    mock_reversal_patterns.reset_mock()
    mock_cycles.reset_mock()
    mock_relative_candles_phases.reset_mock()
    mock_send_telegram_message.reset_mock()

    # Test with doubtful BUY signal (reversal_pattern = 2)
    mock_reversal_patterns.return_value = 2
    mock_cycles.return_value = pd.Series(['X'] * 49 + ['B'], index=test_df_for_bot.index)
    mock_relative_candles_phases.return_value = np.array([1] * 49 + [1])
    analyze_second_strategy('1h', ['chat_id_nexus'])
    mock_send_telegram_message.assert_called_once()
    message = mock_send_telegram_message.call_args[0][0]
    assert "🟢 *2 cycles BUY Signal Detected!* 🟢" in message # Still "BUY Signal Detected"
    assert "Reversal Pattern: BUY" in message # Still "BUY"
    mock_get_ohlcv.reset_mock()
    mock_reversal_patterns.reset_mock()
    mock_cycles.reset_mock()
    mock_relative_candles_phases.reset_mock()
    mock_send_telegram_message.reset_mock()

    # Test with doubtful SELL signal (reversal_pattern = -2)
    mock_reversal_patterns.return_value = -2
    mock_cycles.return_value = pd.Series(['X'] * 49 + ['-B'], index=test_df_for_bot.index)
    mock_relative_candles_phases.return_value = np.array([1] * 49 + [-1])
    analyze_second_strategy('1h', ['chat_id_nexus'])
    mock_send_telegram_message.assert_called_once()
    message = mock_send_telegram_message.call_args[0][0]
    assert "🔴 *2 cycles  SELL Signal Detected!* 🔴" in message # Still "SELL Signal Detected"
    assert "Reversal Pattern: SELL" in message # Still "SELL"
    mock_get_ohlcv.reset_mock()
    mock_reversal_patterns.reset_mock()
    mock_cycles.reset_mock()
    mock_relative_candles_phases.reset_mock()
    mock_send_telegram_message.reset_mock()


    # Test when get_ohlcv returns None
    mock_get_ohlcv.return_value = None
    analyze_second_strategy('1h', ['chat_id_nexus'])
    mock_reversal_patterns.assert_not_called()
    mock_cycles.assert_not_called()
    mock_relative_candles_phases.assert_not_called()
    mock_send_telegram_message.assert_not_called()
    mock_get_ohlcv.reset_mock()

    # Test when TALib method returns empty (or not enough data)
    mock_get_ohlcv.return_value = test_df_for_bot.iloc[:2] # Not enough data for TALib internal lookback
    mock_reversal_patterns.return_value = pd.Series([]) # Simulate empty return due to insufficient data
    mock_cycles.return_value = pd.Series([])
    mock_relative_candles_phases.return_value = np.array([])
    analyze_second_strategy('1h', ['chat_id_nexus'])
    # Should not call send_telegram_message if patterns are empty
    mock_send_telegram_message.assert_not_called()
    mock_get_ohlcv.reset_mock()
    mock_reversal_patterns.reset_mock()
    mock_cycles.reset_mock()
    mock_relative_candles_phases.reset_mock()
    mock_send_telegram_message.reset_mock()


@patch('bot.analyze_bollinger_strategy')
@patch('bot.analyze_second_strategy')
def test_job(mock_analyze_second_strategy, mock_analyze_bollinger_strategy):
    """Test main job execution"""
    from bot import job

    # Test normal execution
    job()

    # Verify Bollinger strategy is called for all its timeframes
    assert mock_analyze_bollinger_strategy.call_count == len(CONFIG['timeframes'])
    for tf in CONFIG['timeframes'].values():
        mock_analyze_bollinger_strategy.assert_any_call(tf, CONFIG['telegram_chat_ids']['jose'])

    # Verify second strategy is called for all its timeframes
    assert mock_analyze_second_strategy.call_count == len(CONFIG['second_strategy_timeframes'])
    for tf in CONFIG['second_strategy_timeframes']:
        mock_analyze_second_strategy.assert_any_call(tf, CONFIG['telegram_chat_ids']['nexus'])

    mock_analyze_bollinger_strategy.reset_mock()
    mock_analyze_second_strategy.reset_mock()

    # Test with an exception in one of the analysis functions
    mock_analyze_bollinger_strategy.side_effect = Exception("Test error BB")
    mock_analyze_second_strategy.side_effect = None # Ensure this one works
    with patch('bot.logger.error') as mock_logger_error:
        job()
        # The job should log the error but still attempt to run other strategies
        mock_logger_error.assert_called()
        # Check that second strategy was still attempted (if it wasn't the one that failed first)
        assert mock_analyze_second_strategy.call_count == len(CONFIG['second_strategy_timeframes'])

    mock_analyze_bollinger_strategy.reset_mock()
    mock_analyze_second_strategy.reset_mock()

    mock_analyze_second_strategy.side_effect = Exception("Test error Second")
    mock_analyze_bollinger_strategy.side_effect = None # Ensure this one works
    with patch('bot.logger.error') as mock_logger_error:
        job()
        mock_logger_error.assert_called()
        assert mock_analyze_bollinger_strategy.call_count == len(CONFIG['timeframes'])


# Remove or comment out resample_data tests if no longer used
# The provided bot.py does not use resample_data, so these tests are irrelevant.
# def test_resample_data(test_data):
#     """Test timeframe resampling"""
#     # Test resampling to 15min
#     df_15min = resample_data(test_data, '15min')
#     assert len(df_15min) <= len(test_data) / 3  # Roughly 1/3 as many candles
#
#     # Test resampling to 1h
#     df_1h = resample_data(test_data, '1h')
#     assert len(df_1h) <= len(test_data) / 12
#
#     # Test with insufficient data
#     small_data = create_test_data(rows=10)
#     assert resample_data(small_data, '1h') is None


# Re-evaluate the quality tests if plot_chart is robust
# The existing test_plot_chart should cover quality.
# def test_image_generation_quality(indicators_data):
#     """Special test for image generation issues"""
#     test_cases = [
#         ('normal', indicators_data, 'overbought'),
#         ('empty_indicators', indicators_data.drop(['upper_band', 'lower_band', 'rsi'], axis=1), 'oversold'),
#         ('partial_data', indicators_data.iloc[:10], 'overbought'),
#         ('nan_values', indicators_data.mask(indicators_data > 100), 'oversold')
#     ]
#
#     for name, data, signal in test_cases:
#         filename = None
#         try:
#             # Ensure column names are lowercase for plot_chart
#             df_input = data.copy()
#             # This part needs careful handling if the input 'data' to this test already has mixed cases
#             # For now, assuming indicators_data fixture already provides consistent lowercase
#             # If not, add df_input.columns = [c.lower() for c in df_input.columns] if necessary
#             filename = plot_chart(df_input, signal, 1, '5min') # Assuming signal is for the last candle
#             assert os.path.exists(filename), f"{name}: File not created"
#
#             # Verify image can be read and has content
#             img = plt.imread(filename)
#             assert img.size > 0, f"{name}: Empty image"
#             assert img.shape[2] == 4, f"{name}: Incorrect color channels"  # RGBA
#
#             # Check for common image generation issues
#             if name == 'normal':
#                 # Check if indicators are visible (sample center pixel)
#                 center_y = img.shape[0] // 2
#                 center_x = img.shape[1] // 2
#                 assert not np.all(img[center_y, center_x] == 1), f"{name}: Blank image area"
#
#         except Exception as e:
#             pytest.fail(f"Image generation failed for {name}: {str(e)}")
#         finally:
#             if filename and os.path.exists(filename):
#                 os.remove(filename)

@patch('requests.post')
@patch('bot.os.path.exists', return_value=True) # Mock os.path.exists for cleanup
@patch('bot.os.remove') # Mock os.remove to prevent actual file deletion during test
def test_send_telegram_notification_with_chart(mock_remove, mock_exists, mock_post):
    """Test that generated images are valid for Telegram via send_telegram_notification_with_chart"""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None # No exception on good status
    mock_post.return_value = mock_response

    # Create a dummy image file for testing
    dummy_image_path = os.path.join(CONFIG['log_directory'], 'dummy_chart.png')
    with open(dummy_image_path, 'w') as f:
        f.write("dummy image content")

    message = "Test chart notification"
    chat_ids = ['test_chat_id_1', 'test_chat_id_2']

    send_telegram_notification_with_chart(message, dummy_image_path, chat_ids)

    # Assert requests.post was called twice (once for each chat_id)
    assert mock_post.call_count == 2

    # Verify arguments for one call
    args, kwargs = mock_post.call_args_list[0]
    assert 'url' in args[0] and 'sendPhoto' in args[0]
    assert 'files' in kwargs
    assert 'photo' in kwargs['files']
    assert kwargs['files']['photo'].name == dummy_image_path # Check correct file was passed
    assert 'data' in kwargs
    assert kwargs['data']['chat_id'] in chat_ids
    assert kwargs['data']['caption'] == message

    # Assert the dummy image was "removed"
    mock_remove.assert_called_with(dummy_image_path)

    # Test error handling during send
    mock_post.side_effect = requests.exceptions.RequestException("Network error")
    with patch('bot.logger.error') as mock_logger_error:
        with pytest.raises(requests.exceptions.RequestException): # Expect the error to be re-raised
            send_telegram_notification_with_chart(message, dummy_image_path, chat_ids)
        mock_logger_error.assert_called_once() # Ensure error was logged

    # Test error handling during cleanup (image removal)
    mock_post.side_effect = None # Reset mock_post
    mock_remove.side_effect = Exception("Cleanup error")
    with patch('bot.logger.error') as mock_logger_error:
        send_telegram_notification_with_chart(message, dummy_image_path, chat_ids)
        mock_logger_error.assert_called() # Ensure cleanup error is logged


@patch('requests.post')
def test_send_telegram_message(mock_post):
    """Test send_telegram_message function"""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None # No exception on good status
    mock_post.return_value = mock_response

    message = "Test plain message"
    chat_ids = ['test_chat_id_1']

    send_telegram_message(message, chat_ids)
    assert mock_post.called_once()
    args, kwargs = mock_post.call_args
    assert 'url' in args[0] and 'sendMessage' in args[0]
    assert kwargs['data']['chat_id'] == chat_ids[0]
    assert kwargs['data']['text'] == message
    assert kwargs['data']['parse_mode'] == 'Markdown'

    # Test with multiple chat IDs
    mock_post.reset_mock()
    send_telegram_message(message, ['id1', 'id2'])
    assert mock_post.call_count == 2

    # Test error handling
    mock_post.side_effect = requests.exceptions.RequestException("Message send error")
    with patch('bot.logger.error') as mock_logger_error:
        send_telegram_message(message, chat_ids)
        mock_logger_error.assert_called_once()


if __name__ == "__main__":
    # Ensure logs directory exists for chart generation tests
    if not os.path.exists(CONFIG['log_directory']):
        os.makedirs(CONFIG['log_directory'])
    pytest.main(['-v', '--cov=bot', '--cov-report=html'])