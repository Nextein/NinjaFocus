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

# Configuration
CONFIG = {
    'telegram_token': '7697676714:AAEA8MQBUvW_FvgDsbPk1EH2Mm9Iow4hXFw',
    'telegram_chat_id_nexus': '7781869973',
    'telegram_chat_id_jose': '6654462171',
    'symbol': 'SOL/USDT',
    'timeframes': {
        '15min': '15min',
        '30min': '30min',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '6h': '6h',
        '12h': '12h'
    },
    'timeframe_priority': ['15min', '30min', '1h', '2h', '4h', '6h', '12h'],  # Order for data fetching
    'bb_period': 20,
    'bb_stddev': 2,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'log_directory': 'logs',
    'min_candles_for_analysis': 50,  # Minimum candles needed for reliable indicators
    'max_fetch_limit': 1000,  # Maximum candles to fetch in one request
    'required_history_days': 7  # Days of history needed for largest timeframe
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

def send_telegram_message(message):
    """Send simple text message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/sendMessage"
        data = {
            'chat_id': CONFIG['telegram_chat_id_jose'],
            'text': message,
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, data=data)
        response.raise_for_status()
        logger.info("Telegram message sent successfully to jose")
        data = {
            'chat_id': CONFIG['telegram_chat_id_nexus'],
            'text': message,
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, data=data)
        response.raise_for_status()
        logger.info("Telegram message sent successfully to jose")
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
        '12h': 720
    }.get(timeframe, 15)  # Default to 15 minutes if timeframe not found
    
    # Calculate required candles to cover required_history_days
    days_to_minutes = CONFIG['required_history_days'] * 24 * 60
    required_candles = math.ceil(days_to_minutes / timeframe_mins)
    
    # Add buffer for indicator calculation
    required_candles += max(CONFIG['bb_period'], CONFIG['rsi_period']) * 2
    
    # Don't exceed exchange limits
    return min(required_candles, CONFIG['max_fetch_limit'])

def get_ohlcv(timeframe=None, since=None):
    """Fetch OHLCV data with smart fetching strategy"""
    try:
        ccxt_timeframes = {
            '15min': '15m',
            '30min': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '12h': '12h'
        }
        
        # If no timeframe specified, use the smallest one (5min)
        if timeframe is None:
            timeframe = CONFIG['timeframe_priority'][0]
        
        tf = ccxt_timeframes[timeframe]
        
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
                '12h': 720
            }.get(timeframe, 15)
            
            since = exchange.parse8601((datetime.now() - timedelta(
                minutes=timeframe_mins * limit
            )).strftime('%Y-%m-%d %H:%M:%S'))
        
        logger.info(f"Fetching {limit} candles for {timeframe} timeframe since {exchange.iso8601(since)}")
        
        ohlcv = exchange.fetch_ohlcv(CONFIG['symbol'], tf, since=since, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logger.info(f"Fetched {len(df)} OHLCV records for {timeframe} timeframe")
        return df.set_index('timestamp')
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV data for {timeframe}: {str(e)}")
        raise

def resample_data(df, timeframe):
    """Resample data to target timeframe using modern pandas syntax"""
    try:
        # Convert our timeframe labels to pandas resample codes
        resample_code = {
            '5min': '5min',
            '15min': '15min',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '12h': '12h'
        }[timeframe]
        
        resampled = df.resample(resample_code).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Ensure we have enough data for analysis
        if len(resampled) < CONFIG['min_candles_for_analysis']:
            logger.warning(f"Insufficient data for {timeframe} timeframe (only {len(resampled)} candles)")
            return None
            
        logger.debug(f"Resampled data to {timeframe} timeframe ({len(resampled)} candles)")
        return resampled
    except Exception as e:
        logger.error(f"Failed to resample data: {str(e)}")
        raise

def calculate_indicators(df):
    """Calculate Bollinger Bands and RSI with data validation"""
    try:
        if df is None or len(df) < CONFIG['bb_period']:
            logger.warning("Not enough data to calculate indicators")
            return None
            
        # Bollinger Bands
        df['middle_band'] = df['close'].rolling(CONFIG['bb_period']).mean()
        std = df['close'].rolling(CONFIG['bb_period']).std()
        df['upper_band'] = df['middle_band'] + std * CONFIG['bb_stddev']
        df['lower_band'] = df['middle_band'] - std * CONFIG['bb_stddev']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(CONFIG['rsi_period']).mean()
        avg_loss = loss.rolling(CONFIG['rsi_period']).mean()
        
        rs = avg_gain / avg_loss
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
    """Check for trading signals with safety checks"""
    try:
        if df is None or len(df) == 0:
            return None, 0
            
        last = df.iloc[-1]
        
        # Verify we have all required fields
        required_fields = ['close', 'upper_band', 'lower_band', 'rsi']
        if not all(field in last for field in required_fields):
            logger.warning("Missing required fields for condition checking")
            return None, 0
            
        price = last['close']
        bb_percentage = 0
        signal = None
        
        if price > last['upper_band']:# and last['rsi'] >= CONFIG['rsi_overbought']:
            bb_percentage = ((price - last['upper_band']) / last['upper_band']) * 100
            signal = 'overbought'
        elif price < last['lower_band']:# and last['rsi'] <= CONFIG['rsi_oversold']:
            bb_percentage = ((last['lower_band'] - price) / last['lower_band']) * 100
            signal = 'oversold'
        
        if signal:
            logger.info(f"Signal detected: {signal.upper()} (Price: {price:.4f}, RSI: {last['rsi']:.2f})")

        return signal, bb_percentage
    except Exception as e:
        logger.error(f"Failed to check conditions: {str(e)}")
        return None, 0

def generate_chart(df, timeframe, signal):
    """Generate dark mode candlestick chart with indicators and right padding"""
    try:
        # Validate input data
        if df is None or df.empty:
            logger.error("Empty or None DataFrame provided")
            return None

        # Use only the last 100 candles, but ensure we have enough data
        display_candles = min(100, len(df))
        df_display = df.iloc[-display_candles:].copy()
        
        # Verify we have required columns
        required_cols = {'open', 'high', 'low', 'close', 'volume', 
                        'upper_band', 'middle_band', 'lower_band', 'rsi'}
        if not required_cols.issubset(df_display.columns):
            logger.error(f"Missing required columns in DataFrame")
            return None

        # Set up dark mode style
        mc = mpf.make_marketcolors(
            up='#3dc26f',  # green for up candles
            down='#e74c3c',  # red for down candles
            edge={'up':'#3dc26f', 'down':'#e74c3c'},
            wick={'up':'#3dc26f', 'down':'#e74c3c'},
            volume='#2eb82e',  # green for volume
            ohlc='#ffffff'  # white for ohlc lines
        )

        style = mpf.make_mpf_style(
            base_mpf_style='nightclouds',
            marketcolors=mc,
            rc={
                'axes.labelcolor': '#e0e0e0',
                'axes.edgecolor': '#404040',
                'xtick.color': '#b0b0b0',
                'ytick.color': '#b0b0b0',
                'figure.facecolor': '#121212',
                'axes.facecolor': '#1e1e1e',
                'grid.color': '#303030',
                'grid.alpha': 0.3,
                'grid.linestyle': '--',

            }
        )

        # Create plots for indicators
        apds = [
            mpf.make_addplot(df_display[['upper_band', 'middle_band', 'lower_band']], 
                          color='#4d94ff', panel=0, alpha=0.7),
            mpf.make_addplot(df_display['volume'], 
                          type='bar', color='#ddd', alpha=0.4, panel=1, ylabel='Volume', y_on_right=True),
            mpf.make_addplot(df_display['rsi'], 
                          color='#b967ff', panel=1, ylabel='RSI', y_on_right=False)
            
        ]

        # Generate filename with timestamp
        filename = f'{datetime.now().strftime("%Y%m%d_%H%M")}_{timeframe}.png'
        
        # Calculate right padding (5% of total candles)
        padding = int(len(df_display) * 0.05)
        
        # Create the plot with proper configuration
        fig, axes = mpf.plot(
            df_display,
            type='candle',
            addplot=apds,
            style=style,
            volume=False,
            title=f'\n{CONFIG["symbol"]} {timeframe} - {signal}',
            figratio=(12, 8),
            panel_ratios=(6,1),
            returnfig=True,
            xlim=(0, len(df_display) + padding),  # Add right padding
            tight_layout=True,
            scale_padding={'left':0.1, 'top':0.5, 'right':0.1, 'bottom':0.1}
        )
        
        # Adjust layout and save
        fig.savefig(
            filename,
            dpi=100,
            facecolor=fig.get_facecolor(),
            bbox_inches='tight',
            pad_inches=0.5
        )
        plt.close(fig)
        
        logger.info(f"Successfully generated chart: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Chart generation failed: {str(e)}", exc_info=True)
        return None

def send_notification(message, image_path):
    """Send Telegram notification with chart"""
    try:
        url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/sendPhoto"
        files = {'photo': open(image_path, 'rb')}
        data = {'chat_id': CONFIG['telegram_chat_id_jose'], 'caption': message}
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        logger.info("Telegram notification sent successfully")
        data = {'chat_id': CONFIG['telegram_chat_id_nexus'], 'caption': message}
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        logger.info("Telegram notification sent successfully")
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {str(e)}")
        raise
    finally:
        try:
            os.remove(image_path)
            logger.debug(f"Removed temporary chart file: {image_path}")
        except:
            pass

def analyze_timeframe(timeframe):
    """Process individual timeframe with proper data fetching"""
    try:
        # First try to fetch data for the target timeframe directly
        df = get_ohlcv(timeframe)
        df = calculate_indicators(df)
        
        # If we don't have enough data, try to fetch higher timeframe data
        if df is None or len(df) < CONFIG['min_candles_for_analysis']:
            current_idx = CONFIG['timeframe_priority'].index(timeframe)
            if current_idx > 0:
                higher_tf = CONFIG['timeframe_priority'][current_idx - 1]
                logger.info(f"Not enough data for {timeframe}, trying higher timeframe {higher_tf}")
                df = get_ohlcv(higher_tf)
                df = resample_data(df, timeframe)
                df = calculate_indicators(df)
        
        if df is None or len(df) < CONFIG['min_candles_for_analysis']:
            logger.warning(f"Still insufficient data for {timeframe} after fallback")
            return
            
        signal, bb_pct = check_conditions(df)
        
        if signal:
            message = (f"🚨 {CONFIG['symbol']} {timeframe} {signal.upper()} 🚨\n"
                      f"Price is {abs(bb_pct):.2f}% {'above' if signal == 'overbought' else 'below'} BB\n"
                      f"Current Price: {df.iloc[-1]['close']:.4f}\n"
                      f"RSI: {df.iloc[-1]['rsi']:.2f}")
            
            send_notification(message, chart_path)
            chart_path = generate_chart(df, timeframe, signal)
    except Exception as e:
        logger.error(f"Error analyzing {timeframe} timeframe: {str(e)}")

def job():
    """Main job execution"""
    try:
        logger.info("Starting analysis job")
        
        for timeframe in CONFIG['timeframe_priority']:
            analyze_timeframe(timeframe)
            
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
            f"• Timeframes: {', '.join(CONFIG['timeframes'].keys())}\n"
            f"• Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        send_telegram_message(start_msg)
        
        # Initial run
        job()
        
        # Setup scheduler
        scheduler = BlockingScheduler()
        scheduler.add_job(job, 'interval', minutes=5)
        logger.info("Scheduler started - running every 5 minutes")
        
        scheduler.start()
        
    except KeyboardInterrupt:
        shutdown_msg = (
            f"🛑 *Trading Bot Stopped* 🛑\n"
            f"• Shutdown at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"• Last run: {datetime.now().strftime('%H:%M:%S')}"
        )
        send_telegram_message(shutdown_msg)
        logger.info("Bot stopped by user")
        
    except Exception as e:
        error_msg = (
            f"❌ *Trading Bot Crashed* ❌\n"
            f"• Error: {str(e)}\n"
            f"• Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        send_telegram_message(error_msg)
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        
    finally:
        logging.shutdown()