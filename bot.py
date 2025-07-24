import logging
from logging.handlers import TimedRotatingFileHandler
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import mplfinance as mpf
from ccxt import binance
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
import os
import math
import json

# Import TALib from features.py
from features import TALib  # Using the TALib class from the uploaded features.py

# Configuration
CONFIG = {
    'telegram_token': '7697676714:AAEA8MQBUvW_FvgDsbPk1EH2Mm9Iow4hXFw',
    'telegram_chat_ids': {
        'jose': ['7781869973'],  # Jose's chat ID for Bollinger Band strategy
        'nexus': ['6654462171']  # Nexus's chat ID for Relative Candles strategy
    },
    'symbol': 'SOL/USDT',  # Apply both strategies to SOL/USDT
    'timeframes': {  # Timeframes for the Bollinger Band strategy
        '15min': '15min',
        '30min': '30min',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '6h': '6h',
        '12h': '12h',
        '1d': '1d',
    },
    'second_strategy_timeframes': ['15min','30min', '1h', '2h', '4h', '6h', '12h', '1d'],  # Timeframes for the second strategy
    'timeframe_priority': ['15min', '30min', '1h', '2h', '4h', '6h', '12h', '1d'],  # Order for data fetching for BB strategy
    'bb_period': 20,
    'bb_stddev': 2,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'log_directory': 'logs',
    'min_candles_for_analysis': 50,  # Minimum candles needed for reliable indicators
    'max_fetch_limit': 1000,  # Maximum candles to fetch in one request
    'required_history_days': 7,  # Days of history needed for largest timeframe
    'lookback_candles': 1,  # Number of past candles to check for a signal (including current)
    'state_file': 'signal_state.json'  # File to store signal states
}

# Initialize logging with date in filename
def setup_logging():
    """Configure daily rotating logs with date in filename"""
    if not os.path.exists(CONFIG['log_directory']):
        os.makedirs(CONFIG['log_directory'])

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Log file with start date in name
    start_date = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(CONFIG['log_directory'], f'trading_bot_{start_date}.log')

    file_handler = TimedRotatingFileHandler(
        log_file, when='midnight', interval=1, backupCount=3
    )
    file_handler.setFormatter(log_formatter)
    file_handler.suffix = "%Y%m%d.log"  # To keep dates in rotated filenames

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()
exchange = binance({'enableRateLimit': True})


def send_telegram_message(message, chat_ids_to_send):
    """Send simple text message to Telegram to specified chat IDs"""
    try:
        url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/sendMessage"
        for chat_id in chat_ids_to_send:
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, data=data)
            response.raise_for_status()
            logger.info(f"Telegram message sent successfully to chat ID: {chat_id}")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {str(e)}")


def calculate_required_candles(timeframe):
    """Calculate how many candles we need to fetch for a given timeframe"""
    # Convert timeframe to minutes
    timeframe_mins = {
        '15min': 15,
        '30min': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '12h': 720,
        '1d': 1440
    }.get(timeframe, 15)  # Default to 15 minutes if timeframe not found

    # Calculate required candles to cover required_history_days
    days_to_minutes = CONFIG['required_history_days'] * 24 * 60
    required_candles = math.ceil(days_to_minutes / timeframe_mins)

    # Add buffer for indicator calculation and lookback
    # Adjusted for the second strategy, assuming similar or more conservative min candles
    required_candles += max(CONFIG['bb_period'], CONFIG['rsi_period']) * 2 + CONFIG['lookback_candles']

    # Don't exceed exchange limits
    return min(required_candles, CONFIG['max_fetch_limit'])


def get_ohlcv(timeframe=None, since=None, symbol=None):
    """Fetch OHLCV data with smart fetching strategy"""
    try:
        ccxt_timeframes = {
            '15min': '15m',
            '30min': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '12h': '12h',
            '1d': '1d' 
        }

        # If no timeframe specified, use the smallest one (5min)
        if timeframe is None:
            timeframe = CONFIG['timeframe_priority'][0]

        tf = ccxt_timeframes[timeframe]

        # Use the symbol from CONFIG if not provided
        if symbol is None:
            symbol = CONFIG['symbol']

        # Calculate how many candles we need
        limit = calculate_required_candles(timeframe)

        # If since is not provided, calculate based on required history
        if since is None:
            timeframe_mins = {
                '15min': 15,
                '30min': 30,
                '1h': 60,
                '2h': 120,
                '4h': 240,
                '6h': 360,
                '12h': 720,
                '1d': 1440
            }
            # Fetch slightly more data to ensure enough for indicator calculations
            # The 'since' parameter needs to be in milliseconds
            since_ms = exchange.parse8601(
                (datetime.utcnow() - timedelta(minutes=timeframe_mins[timeframe] * limit)).isoformat())
        else:
            since_ms = since

        logger.info(f"Fetching {limit} candles for {symbol} {tf} from {datetime.utcfromtimestamp(since_ms / 1000)}...")
        ohlcv = exchange.fetch_ohlcv(symbol, tf, since=since_ms, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # Ensure numeric types
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(
            pd.to_numeric)
        logger.info(f"Fetched {len(df)} candles for {symbol} {tf}.")
        return df

    except Exception as e:
        logger.error(f"Error fetching OHLCV data for {symbol} {timeframe}: {str(e)}")
        return None


def calculate_indicators(df):
    """Calculate Bollinger Bands and RSI"""
    try:
        if df is None or df.empty:
            logger.warning("Empty DataFrame for indicator calculation.")
            return None

        # Bollinger Bands
        df['middle_band'] = df['close'].rolling(window=CONFIG['bb_period']).mean()
        df['std_dev'] = df['close'].rolling(window=CONFIG['bb_period']).std()
        df['upper_band'] = df['middle_band'] + (df['std_dev'] * CONFIG['bb_stddev'])
        df['lower_band'] = df['middle_band'] - (df['std_dev'] * CONFIG['bb_stddev'])

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=CONFIG['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=CONFIG['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Drop rows with NaN values (where indicators couldn't be calculated)
        df = df.dropna()

        if len(df) == 0:
            logger.warning("No valid data after indicator calculation")
            return None

        logger.debug("Calculated technical indicators")
        return df
    except Exception as e:
        logger.error(f"Failed to calculate indicators: {str(e)}")
        return None


def check_conditions(df):
    """Check for trading signals, including a lookback period"""
    try:
        if df is None or len(df) == 0:
            return None, None, None  # Added None for the candle index

        # Ensure enough candles for lookback
        if len(df) < CONFIG['lookback_candles']:
            logger.warning(f"Not enough data for lookback period ({len(df)} < {CONFIG['lookback_candles']})")
            return None, None, None

        # Iterate through the last 'lookback_candles'
        for i in range(1, CONFIG['lookback_candles'] + 1):
            if len(df) - i < 0:  # Avoid indexing errors for very small DFs
                break
            current_candle_data = df.iloc[-i]

            # Verify we have all required fields for this candle
            required_fields = ['open', 'high', 'low', 'close', 'upper_band', 'lower_band', 'rsi']
            if not all(field in current_candle_data.index for field in required_fields):
                logger.error(
                    f"Missing required fields for signal check in candle data: {current_candle_data.index.tolist()}")
                continue  # Skip this candle if data is incomplete

            # Bollinger Band Buy Signal: Close price crosses below lower band and RSI is oversold
            buy_signal = (
                    current_candle_data['close'] < current_candle_data['lower_band'] and
                    current_candle_data['rsi'] < CONFIG['rsi_oversold']
            )

            # Bollinger Band Sell Signal: Close price crosses above upper band and RSI is overbought
            sell_signal = (
                    current_candle_data['close'] > current_candle_data['upper_band'] and
                    current_candle_data['rsi'] > CONFIG['rsi_overbought']
            )

            if buy_signal:
                logger.info(f"Buy signal detected on {df.index[-i]}")
                return "BUY", current_candle_data, i  # Return signal type, candle data, and index from end
            elif sell_signal:
                logger.info(f"Sell signal detected on {df.index[-i]}")
                return "SELL", current_candle_data, i  # Return signal type, candle data, and index from end

        logger.info("No signal detected in the last lookback period.")
        return None, None, None
    except Exception as e:
        logger.error(f"Error checking conditions: {str(e)}")
        return None, None, None


def plot_chart(df, signal_type, trigger_candle_index, timeframe):
    """Generate and save candlestick chart with indicators and right padding, highlighting the trigger candle."""
    image_path = os.path.join(CONFIG['log_directory'], 'trade_signal.png')
    try:
        # Validate input data
        if df is None or df.empty:
            logger.error("Empty or None DataFrame provided")
            return None

        # Use only the last 100 candles, but ensure we have enough data
        display_candles = min(100, len(df))
        df_display = df.iloc[-display_candles:].copy()

        # Verify we have required columns
        required_cols = {'open', 'high', 'low', 'close', 'volume', 'upper_band', 'middle_band', 'lower_band', 'rsi'}
        if not required_cols.issubset(df_display.columns):
            logger.error(f"Missing required columns in DataFrame: {required_cols - set(df_display.columns)}")
            return None

        # Set up dark mode style
        mc = mpf.make_marketcolors(
            up='#3dc26f',  # green for up candles
            down='#e74c3c',  # red for down candles
            edge={'up': '#3dc26f', 'down': '#e74c3c'},
            wick={'up': '#3dc26f', 'down': '#e74c3c'},
            volume='#2eb82e',  # green for volume
            ohlc='#ffffff'  # white for ohlc lines
        )
        style = mpf.make_mpf_style(
            base_mpf_style='yahoo',
            marketcolors=mc,
            figcolor='#1a1a1a',  # background color
            bgcolor='#1a1a1a',  # panel background
            gridcolor='#2b2b2b',  # grid lines
            textcolor='#ffffff',  # text color
            facecolor='#1a1a1a',  # axes background
            edgecolor='#1a1a1a'  # borders
        )

        # Create addplots for indicators
        apds = [
            mpf.make_addplot(df_display['upper_band'], color='#0077B6', panel=0, linestyle='--'),
            mpf.make_addplot(df_display['middle_band'], color='#90E0EF', panel=0),
            mpf.make_addplot(df_display['lower_band'], color='#0077B6', panel=0, linestyle='--'),
            mpf.make_addplot(df_display['rsi'], panel=1, color='#FFD60A', ylim=(0, 100)),
            mpf.make_addplot(pd.Series(CONFIG['rsi_overbought'], index=df_display.index), panel=1, color='#e74c3c',
                             linestyle=':'),
            mpf.make_addplot(pd.Series(CONFIG['rsi_oversold'], index=df_display.index), panel=1, color='#3dc26f',
                             linestyle=':')
        ]

        # Highlight the trigger candle
        if signal_type and trigger_candle_index:
            # Calculate the actual index in the df_display
            highlight_index_in_display = len(df_display) - trigger_candle_index
            if highlight_index_in_display >= 0:
                highlight_color = '#FF4500' if signal_type == "SELL" else '#32CD32'  # Orange for sell, lime green for buy

                # Create a list of dictionaries for `alines`
                # We need to draw a vertical line at the exact trigger point
                # Use the timestamp of the trigger candle
                trigger_time = df_display.index[highlight_index_in_display]

                # Create a DataFrame for `vlines` to highlight the specific candle
                vlines_data = pd.DataFrame(index=[trigger_time], data={'val': 1})
                apds.append(
                    mpf.make_addplot(vlines_data['val'], type='vlines', panel=0, color=highlight_color, linestyle='-',
                                     width=0.7, alpha=0.7)
                )
                logger.info(
                    f"Highlighting trigger candle at index {highlight_index_in_display} (original index {df.index[-trigger_candle_index]})")

        fig, axes = mpf.plot(
            df_display,
            type='candle',
            style=style,
            title=f"{CONFIG['symbol']} {timeframe} - Bollinger Bands & RSI",
            ylabel='Price',
            ylabel_lower='Volume',
            volume=True,
            addplot=apds,
            returnfig=True,
            panel_ratios=(3, 1),
            figscale=1.5,
            # Adjust padding if needed, default is usually fine
            xrotation=0  # Prevents x-axis labels from rotating
        )

        # Add horizontal lines for RSI overbought/oversold levels on the RSI panel
        axes[1].text(df_display.index[0], CONFIG['rsi_overbought'] + 2, 'Overbought', color='#e74c3c',
                     verticalalignment='bottom')
        axes[1].text(df_display.index[0], CONFIG['rsi_oversold'] - 2, 'Oversold', color='#3dc26f',
                     verticalalignment='top')

        # Save the plot
        plt.savefig(image_path, bbox_inches='tight', dpi=100)
        plt.close(fig)  # Close the figure to free up memory
        logger.info(f"Chart saved to {image_path}")
        return image_path

    except Exception as e:
        logger.error(f"Error plotting chart: {str(e)}")
        return None


def send_telegram_notification_with_chart(message, image_path, chat_ids_to_send):
    """Send Telegram notification with chart to specified chat IDs"""
    try:
        url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/sendPhoto"
        for chat_id in chat_ids_to_send:
            files = {'photo': open(image_path, 'rb')}
            data = {'chat_id': chat_id, 'caption': message}
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            logger.info(f"Telegram notification sent successfully to chat ID: {chat_id}")
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {str(e)}")
        raise  # Re-raise to ensure the error is logged and handled
    finally:
        try:
            if os.path.exists(image_path):  # Check if file exists before trying to remove
                os.remove(image_path)
                logger.info(f"Removed chart image: {image_path}")
        except Exception as e:
            logger.error(f"Error removing chart image {image_path}: {str(e)}")


def get_next_candle_start(timeframe):
    """Calculate the exact UTC datetime when the next candle will start"""
    # Get candle duration in minutes
    timeframe_mins = {
        '15min': 15,
        '30min': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '12h': 720,
        '1d': 1440
    }
    
    if timeframe not in timeframe_mins:
        logger.error(f"Invalid timeframe: {timeframe}")
        return None
        
    total_minutes = timeframe_mins[timeframe]
    now = datetime.utcnow()
    
    if timeframe == '1d':
        # Daily candles start at 00:00 UTC
        next_start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        # Calculate current minute of day
        current_minute_of_day = now.hour * 60 + now.minute
        
        # Calculate next candle start minute of day
        next_minute_of_day = ((current_minute_of_day // total_minutes) + 1) * total_minutes
        
        if next_minute_of_day < 1440:  # 1440 minutes in a day
            # Same day
            next_start = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute_of_day)
        else:
            # Next day
            next_minute_of_day -= 1440
            next_start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute_of_day)
    
    return next_start


def load_signal_state():
    """Load signal state from file or initialize new state"""
    state = {
        'bollinger': {},
        'reversal': {}
    }
    
    # Initialize all timeframes
    for tf in CONFIG['timeframes'].values():
        state['bollinger'][tf] = {'last_signal': None, 'next_candle_start': None}
    for tf in CONFIG['second_strategy_timeframes']:
        state['reversal'][tf] = {'last_signal': None, 'next_candle_start': None}
    
    try:
        if os.path.exists(CONFIG['state_file']):
            with open(CONFIG['state_file'], 'r') as f:
                saved_state = json.load(f)
                
                # Merge saved state with initialized state
                for strategy in ['bollinger', 'reversal']:
                    for tf in saved_state.get(strategy, {}):
                        if tf in state[strategy]:
                            # Only update if we have valid data
                            state[strategy][tf] = saved_state[strategy][tf]
                            
                            # Convert next_candle_start from string to datetime if needed
                            if state[strategy][tf]['next_candle_start']:
                                try:
                                    state[strategy][tf]['next_candle_start'] = datetime.fromisoformat(
                                        state[strategy][tf]['next_candle_start']
                                    )
                                except Exception as e:
                                    logger.error(f"Error parsing datetime: {str(e)}")
                                    state[strategy][tf]['next_candle_start'] = None
    except Exception as e:
        logger.error(f"Error loading signal state: {str(e)}")
    
    return state


def save_signal_state(state):
    """Save signal state to file with datetime conversion"""
    try:
        # Create a copy for saving (convert datetime to ISO strings)
        save_state = {
            'bollinger': {},
            'reversal': {}
        }
        
        for strategy in ['bollinger', 'reversal']:
            for tf, data in state[strategy].items():
                save_state[strategy][tf] = data.copy()
                if save_state[strategy][tf]['next_candle_start'] and isinstance(
                    save_state[strategy][tf]['next_candle_start'], datetime
                ):
                    save_state[strategy][tf]['next_candle_start'] = save_state[strategy][tf]['next_candle_start'].isoformat()
        
        with open(CONFIG['state_file'], 'w') as f:
            json.dump(save_state, f, indent=2)
            logger.info("Signal state saved successfully")
    except Exception as e:
        logger.error(f"Error saving signal state: {str(e)}")


def analyze_bollinger_strategy(timeframe, chat_ids, state):
    """Analyzes data for Bollinger Band strategy with next-candle cooldown"""
    logger.info(f"Running Bollinger Band strategy for {CONFIG['symbol']} {timeframe}")
    df = get_ohlcv(timeframe=timeframe, symbol=CONFIG['symbol'])
    if df is None or df.empty:
        return state

    df = calculate_indicators(df)
    if df is None or df.empty:
        return state

    signal_type, trigger_candle_data, trigger_candle_relative_index = check_conditions(df)

    if signal_type:
        tf_state = state['bollinger'].get(timeframe, {'last_signal': None, 'next_candle_start': None})
        now = datetime.utcnow()
        
        # Check if we should send notification
        send_notification = False
        
        # Case 1: Different signal type
        if tf_state['last_signal'] != signal_type:
            send_notification = True
        # Case 2: Cooldown expired (next candle has started)
        elif tf_state['next_candle_start'] and now >= tf_state['next_candle_start']:
            send_notification = True
        # Case 3: No previous signal
        elif tf_state['last_signal'] is None:
            send_notification = True
        
        if send_notification:
            price = trigger_candle_data['close']
            rsi = trigger_candle_data['rsi']
            timestamp = trigger_candle_data.name
            
            # Generate chart
            chart_path = plot_chart(df, signal_type, trigger_candle_relative_index, timeframe)
            if chart_path:
                message = (
                    f"🚨 *{signal_type} Signal Detected!* 🚨\n"
                    f"• Symbol: {CONFIG['symbol']}\n"
                    f"• Timeframe: {timeframe}\n"
                    f"• Price: {price:.4f}\n"
                    f"• RSI: {rsi:.2f}\n"
                    f"• Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                send_telegram_notification_with_chart(message, chart_path, chat_ids)
            else:
                logger.error("Failed to generate chart for Bollinger Band strategy.")
                message = (
                    f"🚨 *{signal_type} Signal Detected!* 🚨\n"
                    f"• Symbol: {CONFIG['symbol']}\n"
                    f"• Timeframe: {timeframe}\n"
                    f"• Price: {price:.4f}\n"
                    f"• RSI: {rsi:.2f}\n"
                    f"• Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"_(Chart generation failed)_"
                )
                send_telegram_message(message, chat_ids)
            
            # Update state with new signal and next candle start time
            next_start = get_next_candle_start(timeframe)
            state['bollinger'][timeframe] = {
                'last_signal': signal_type,
                'next_candle_start': next_start
            }
            logger.info(f"Updated state for {timeframe}: signal={signal_type}, next_candle={next_start}")
    
    return state


def analyze_rc_strategy(timeframe, chat_ids, state):
    """Analyzes data for reversal strategy with next-candle cooldown"""
    logger.info(f"Running reversal strategy for {CONFIG['symbol']} {timeframe}")
    # Fetch data for the specified symbol and timeframe
    df = get_ohlcv(timeframe=timeframe, symbol=CONFIG['symbol'])
    if df is None or df.empty:
        return state

    try:
        # Apply technical analysis
        reversal_pattern = TALib.relativeCandlesReversalPatterns(df)
        cycles = TALib.Cycles(df)
        phases = TALib.relativeCandlesPhases(df)

        if reversal_pattern.empty or cycles.empty or phases.empty:
            return state

        # Get last values
        if len(reversal_pattern) == 0:
            return state
            
        latest_reversal_pattern = reversal_pattern.iloc[-1]
        latest_close_price = df['close'].iloc[-1]
        latest_timestamp = df.index[-1]
        
        signal_type = None
        signal_message = None
        
        if latest_reversal_pattern == 1:
            signal_type = "BUY"
            signal_message = (
                f"🟢 *Reversal BUY Signal Detected!* 🟢\n"
                f"• Symbol: {CONFIG['symbol']}\n"
                f"• Timeframe: {timeframe}\n"
                f"• Close Price: {latest_close_price:.4f}\n"
                f"• Time: {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        elif latest_reversal_pattern == -1:
            signal_type = "SELL"
            signal_message = (
                f"🔴 *Reversal SELL Signal Detected!* 🔴\n"
                f"• Symbol: {CONFIG['symbol']}\n"
                f"• Timeframe: {timeframe}\n"
                f"• Close Price: {latest_close_price:.4f}\n"
                f"• Time: {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        if signal_type and signal_message:
            tf_state = state['reversal'].get(timeframe, {'last_signal': None, 'next_candle_start': None})
            now = datetime.utcnow()
            
            # Check if we should send notification
            send_notification = False
            
            # Case 1: Different signal type
            if tf_state['last_signal'] != signal_type:
                send_notification = True
            # Case 2: Cooldown expired (next candle has started)
            elif tf_state['next_candle_start'] and now >= tf_state['next_candle_start']:
                send_notification = True
            # Case 3: No previous signal
            elif tf_state['last_signal'] is None:
                send_notification = True
            
            if send_notification:
                send_telegram_message(signal_message, chat_ids)
                # Update state with new signal and next candle start time
                next_start = get_next_candle_start(timeframe)
                state['reversal'][timeframe] = {
                    'last_signal': signal_type,
                    'next_candle_start': next_start
                }
                logger.info(f"Updated state for {timeframe}: signal={signal_type}, next_candle={next_start}")
    
    except Exception as e:
        logger.error(f"Error in reversal strategy: {str(e)}")
    
    return state


def job():
    """Main job with state persistence"""
    logger.info("Starting analysis job...")
    try:
        # Load state
        state = load_signal_state()
        
        # Run Bollinger Band Strategy
        for timeframe in CONFIG['timeframes'].values():
            state = analyze_bollinger_strategy(
                timeframe, 
                CONFIG['telegram_chat_ids']['jose'],
                state
            )
        
        # Run Reversal Strategy
        for timeframe in CONFIG['second_strategy_timeframes']:
            state = analyze_rc_strategy(
                timeframe, 
                CONFIG['telegram_chat_ids']['nexus'],
                state
            )
        
        # Save state
        save_signal_state(state)
        logger.info("Analysis job completed")
    except Exception as e:
        logger.error(f"Job execution failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    try:
        logger.info("Starting trading bot")

        # Send startup message
        start_msg = (
            f"🤖 *Trading Bot Started* 🤖\n"
            f"• Symbol: {CONFIG['symbol']}\n"
            f"• Bollinger Timeframes: {', '.join(CONFIG['timeframes'].keys())}\n"
            f"• Relative Candles Strategy Timeframes: {', '.join(CONFIG['second_strategy_timeframes'])}\n"
            f"• Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        send_telegram_message(start_msg, CONFIG['telegram_chat_ids']['jose'] + CONFIG['telegram_chat_ids'][
            'nexus'])  # Send startup to both

        # Initial run
        job()

        # Setup scheduler
        scheduler = BlockingScheduler()
        # Schedule the job to run every 5 minutes
        scheduler.add_job(job, 'interval', minutes=5)
        logger.info("Scheduler started - running every 5 minutes")

        scheduler.start()

    except KeyboardInterrupt:
        shutdown_msg = (
            f"🛑 *Trading Bot Stopped* 🛑\n"
            f"• Shutdown at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"• Last run: {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        )
        send_telegram_message(shutdown_msg, CONFIG['telegram_chat_ids']['jose'] + CONFIG['telegram_chat_ids'][
            'nexus'])  # Send shutdown to both
        logger.info("Bot stopped by user")

    except Exception as e:
        error_msg = (
            f"❌ *Trading Bot Crashed* ❌\n"
            f"• Error: {str(e)}\n"
            f"• Crash time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        send_telegram_message(error_msg, CONFIG['telegram_chat_ids']['jose'] + CONFIG['telegram_chat_ids'][
            'nexus'])  # Send crash alert to both
        logger.error(f"Bot crashed: {str(e)}", exc_info=True)