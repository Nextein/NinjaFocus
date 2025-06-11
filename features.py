# =====================================================
#               Author: Marc Goulding
#               gouldingmarc@gmail.com
# =====================================================
import time


import numpy as np
import pandas as pd


import pandas_ta as pdta
import talib as ta
import pandas_ta


from pprint import pprint
from scipy import stats, signal


from plutus.lab.fxcharts import ha_candlesticks, renko

# TODO - tulipindicators.org
# TODO - Renko Wyckoff     https://www.tradingview.com/script/9BKOIhdl-Numbers-Renko/
# https://mboxwave.com/indicators
# pandas_ta
# Heikin Ashi fast: https://stackoverflow.com/questions/40613480/heiken-ashi-using-pandas-python
# https://github.com/kylejusticemagnuson/pyti
# https://python.stockindicators.dev/indicators/

# Forex Factory forum
# https://www.incrediblecharts.com/
# https://www.best-metatrader-indicators.com/
# thinkorswim platform


class TALib:
    """
    Technical Analysis library.
    """
    def __init__(self):
        self.wyckoff = [
            self.VolumeProfile,
            self.MoneyFlowMultiplier,
            self.MoneyFlowVolume,
            self.ADL,
            self.POC,
            self.OBV
        ]

        self.trend = [
            self.EMA,
            self.MACD,
            self.WMA
        ]

    @classmethod
    def printAll(self):
        raise Exception("ERROR - TALib class printAll() not implemented")

    @classmethod
    def group(self, data, window) -> pd.Series:
        try:
            g = data.rolling(window).sum()
        except:
            g = pd.Series(data).rolling(window).sum()
        return g

    @classmethod
    def EMA(self, data, period, apply_to='Close', **args):
        """
        Exponential Moving Average.

        This feature has an unstable period. For more, see:
        https://ta-lib.org/d_api/ta_setunstableperiod.html

        args:
            data
            period
            apply_to
        """
        if apply_to is None:
            ema = ta.EMA(data, timeperiod=period)
            return ema
        else:
            try:
                ema = ta.EMA(data[apply_to], timeperiod=period)
            except KeyError:
                ema = ta.EMA(data['Close'], timeperiod=period)
            return ema

    @classmethod
    def MAWI(self, data, fastperiod, slowperiod, **args) -> pd.Series:
        """ Moving Average Width Indicator

        args:
            data: OHLC data
            fastperiod: fast EMA period
            slowperiod: slowEMa period
            
        """
        fema = self.EMA(data, period=fastperiod, apply_to=None)
        sema = self.EMA(data, period=slowperiod, apply_to=None)
        mawi = fema - sema
        return mawi

    @classmethod
    def HighLowIndex(self, data, period):
        """
        High Low Index.
        High Low Index = Today's High - X day's ago Low
        """
        hli = data['High'] - data['Low'].shift(period)
        return hli

    @classmethod
    def MidPrice(self, data, period):
        """
        MidPrice.

        args:
            data
            period
        """
        midprice = ta.MIDPRICE(data['High'], data['Low'], timeperiod=period)
        return midprice

    @classmethod
    def ROC(self, data, **args):
        roc = (data['Close'] / data['Close'].shift(args['lag']) - 1)*100
        return roc.to_frame()

    @classmethod
    def Volume(self, data, **args):
        return data['Volume']

    @classmethod
    def FractalHighs(self, data, **args):
        """
        Bill Williams FractalHighs.

        TODO - Not implemented for options different than 3 or 5

        args:
            data
            window
        """

        if args['window'] % 2 != 1:
            args['window'] += 1

        data.at[:, 'FractalLows'] = None

        if args['window'] == 5:
            for i in range(int((args['window']-1)/2), int(data.shape[0]-(args['window']-1)/2)):
                if (data['High'].iloc[i] > data['High'].iloc[i-2]
                        and data['High'].iloc[i] > data['High'].iloc[i-1]
                        and data['High'].iloc[i] > data['High'].iloc[i+1]
                        and data['High'].iloc[i] > data['High'].iloc[i+2]):

                    data.at[i, 'FractalLows'] = data['High'].iloc[i]
        else:
            for i in range(int((args['window']-1)/2), int(data.shape[0]-(args['window']-1)/2)):
                if (
                        data['High'].iloc[i] > data['High'].iloc[i-1]
                        and data['High'].iloc[i] > data['High'].iloc[i+1]):

                    data.at[i, 'FractalLows'] = data['High'].iloc[i]
        return data['FractalLows']

    @classmethod
    def Peaks(self, x, y, **args):
        """ Finds peaks in a signal. """
        peaks,_ = signal.find_peaks(y)
        pkx = x[peaks]
        pky = y[peaks]

        return pkx, pky

    @classmethod
    def FractalLows(self, data, **args):
        """ Bill Williams FractalLows.

        TODO - Not implemented for options different than 3 or 5

        """

        if args['window'] % 2 != 1:
            args['window'] += 1

        data['FractalLows'] = None

        if args['window'] == 5:
            for i in range(int((args['window']-1)/2), int(data.shape[0]-(args['window']-1)/2)):
                if (data['Low'].iloc[i] < data['Low'].iloc[i-2]
                        and data['Low'].iloc[i] < data['Low'].iloc[i-1]
                        and data['Low'].iloc[i] < data['Low'].iloc[i+1]
                        and data['Low'].iloc[i] < data['Low'].iloc[i+2]):

                    data.at[i, 'FractalLows'] = data['Low'].iloc[i]
        else:
            for i in range(int((args['window']-1)/2), int(data.shape[0]-(args['window']-1)/2)):
                if (
                        data['Low'].iloc[i] < data['Low'].iloc[i-1]
                        and data['Low'].iloc[i] < data['Low'].iloc[i+1]):

                    data.at[i, 'FractalLows'] = data['Low'].iloc[i]
        return data['FractalLows']

    @classmethod
    def DEMA(self, data, **args):
        """
        Double Exponential Moving Average.

        args:
            data
            period
            apply_to
        """
        try:
            dema = ta.DEMA(data[args['apply_to']], timeperiod=args['period'])
        except KeyError:
            dema = ta.DEMA(data['Close'], timeperiod=args['period'])
        return dema

    @classmethod
    def TEMA(self, data, **args):
        """
        Triple Exponential Moving Average.

        args:
            data
            period
            apply_to
        """
        try:
            tema = ta.TEMA(data[args['apply_to']], timeperiod=args['period'])
        except KeyError:
            tema = ta.TEMA(data['Close'], timeperiod=args['period'])
        return tema

    @classmethod
    def TRIMA(self, data, **args):
        """
        Triangular Exponential Moving Average.

        args:
            data
            period
            apply_to
        """
        try:
            trima = ta.TRIMA(data[args['apply_to']], timeperiod=args['period'])
        except KeyError:
            trima = ta.TRIMA(data['Close'], timeperiod=args['period'])
        return trima

    @classmethod
    def SMA(self, data, period, apply_to='Close', **args):
        """
        Simple Moving Average

        args:
            data
            period
            apply_to
        """
        if apply_to is None:
            sma = ta.SMA(data, timeperiod=period)
        else:
            try:
                sma = ta.SMA(data[apply_to], timeperiod=period)
            except KeyError:
                sma = ta.SMA(data['Close'], timeperiod=period)
            return sma

        return sma

    @classmethod
    def WMA(self, data, **args):
        """
        Weighting Moving Average.

        args:
            data
            period
            apply_to
        """
        try:
            wma = ta.WMA(data[args['apply_to']], timeperiod=args['period'])
        except KeyError:
            wma = ta.WMA(data['Close'], timeperiod=args['period'])
        return wma

    # @classmethod
    # def FibMA(self, data, length:int, **args):
    #     """ Fibonnaci MA
    #         length: minimum value is 3
    #     """
    #     indexes = [1,2]
    #     for i in range(length-2):
    #         indexes.append(indexes[-1]+indexes[-2])

    @classmethod
    def MidPoint(self, data, **args):
        """
        MidPoint over period.

        args:
            data
            period
            apply_to
        """
        try:
            midpoint = ta.MIDPOINT(data[args['apply_to']], timeperiod=args['period'])
        except KeyError:
            midpoint = ta.MIDPOINT(data['Close'], timeperiod=args['period'])
        return midpoint

    @classmethod
    def MAVP(self, data, **args):
        """
        Moving Average with Variable Period.

        args:
            data
            minperiod
            maxperiod
        """
        try:
            mavp = ta.MAVP(data[args['apply_to']], periods=args['period'], minperiod=args['minperiod'], maxperiod=args['maxperiod'], matype=0)
        except KeyError:
            mavp = ta.MAVP(data['Close'], periods=args['period'], minperiod=args['minperiod'], maxperiod=args['maxperiod'], matype=0)
        return mavp

    @classmethod
    def MACDHistogram(self, data, fastperiod=12, slowperiod=26, signalperiod=9, **args):
        """
        Moving Average Convergence/Divergence Histogram.

        args:
            data
            fastperiod
            slowperiod
            signalperiod

        """
        try:
            macd, macdsignal, macdhist = ta.MACD(data['Close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        except:
            macd, macdsignal, macdhist = ta.MACD(data, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        return macdhist

    @classmethod
    def MACDLine(self, data, fastperiod=12, slowperiod=26, signalperiod=9, **args):
        """
        Moving Average Convergence/Divergence line.

        args:
            data
            fastperiod
            slowperiod
            signalperiod

        """
        try:
            macd, macdsignal, macdhist = ta.MACD(data['Close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        except:
            macd, macdsignal, macdhist = ta.MACD(data, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        return macd

    @classmethod
    def MACDSignal(self, data, fastperiod=12, slowperiod=26, signalperiod=9, **args):
        """
        Moving Average Convergence/Divergence Signal line.

        args:
            data
            fastperiod
            slowperiod
            signalperiod

        """
        try:
            macd, macdsignal, macdhist = ta.MACD(data['Close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        except:
            macd, macdsignal, macdhist = ta.MACD(data, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        return macdsignal

    @classmethod
    def MACD(self, data, fastperiod=12, slowperiod=26, signalperiod=9, **args):
        """
        Moving Average Convergence/Divergence.

        Returns all 3 parts of MACD indicator.

        args:
            data
            fastperiod
            slowperiod
            signalperiod

        """
        try:
            macd, macdsignal, macdhist = ta.MACD(data['Close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        except:
            macd, macdsignal, macdhist = ta.MACD(data, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

        return macd, macdsignal, macdhist

    @classmethod
    def KAMA(self, data, **args):
        """
        Kaufman Adaptive Moving Average.
        args:
            data
            period
            apply_to
        """
        try:
            kama = ta.KAMA(data[args['apply_to']], timeperiod=args['period'])
        except KeyError:
            kama = ta.KAMA(data['Close'], timeperiod=args['period'])
        return kama

    @classmethod
    def HT_Trendline(self, data, **args):
        """
        Hilbert Transform - Instantaneous Trendline.
        
        args:
            data
            apply_to
        """

        try:
            trendline = ta.HT_TRENDLINE(data[args['apply_to']])
        except KeyError:
            trendline = ta.HT_TRENDLINE(data['Close'])
        return trendline

    @classmethod
    def SAR(self, data, **args):
        """
        Parabolic SAR.

        args:
            data
            acceleration
            maximum
        """
        sar = ta.SAR(data['High'], data['Low'], acceleration=args['acceleration'], maximum=args['maximum'])
        return sar

    @classmethod
    def T3(self, data, **args):
        """
        Triple Exponential Moving Average (T3).

        args:
            data
            period
            apply_to
            vfactor
        """
        try:
            t3 = ta.T3(data[args['apply_to']], timeperiod=args['period'], vfactor=args['vfactor'])
        except KeyError:
            t3 = ta.T3(data['Close'], timeperiod=args['period'], vfactor=args['vfactor'])
        return t3

    @classmethod
    def ADX(self, data, period=14, **args):
        """
        Average Directional Movement Index.

        args:
            data
            period
        """
        adx = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=period)
        return adx

    @classmethod
    def ADXR(self, data, **args):
        """
        Average Directional Movement Index Rating.

        args:
            data
            period
        """
        adxr = ta.ADXR(data['High'], data['Low'], data['Close'], timeperiod=args['period'])
        return adxr

    @classmethod
    def APO(self, data, **args):
        """
        Absolute Price Oscillator.

        args:
            data
            fastperiod
            slowperiod
            matype
        """
        try:
            apo = ta.APO(data['Close'], fastperiod=args['fastperiod'], slowperiod=args['slowperiod'], matype=args['matype'])
        except KeyError:
            apo = ta.APO(data['Close'], fastperiod=args['fastperiod'], slowperiod=args['slowperiod'], matype=0)
        return apo

    @classmethod
    def AroonUp(self, data, **args):
        """
        Aroon - Up line.

        args:
            data
            period
        """
        down, up = ta.AROON(data['High'], data['Low'], timeperiod=args['period'])
        return up

    @classmethod
    def AroonDown(self, data, **args):
        """
        Aroon - Down line.

        args:
            data
            period
        """
        down, up = ta.AROON(data['High'], data['Low'], timeperiod=args['period'])
        return down

    @classmethod
    def AroonOsc(self, data, **args):
        """
        Aroon - Down line.

        args:
            data
            period
        """
        aroon = ta.AROONOSC(data['High'], data['Low'], timeperiod=args['period'])
        return aroon

    @classmethod
    def BOP(self, data, **args):
        """
        Balance of Power.
        """
        bop = ta.BOP(data['Open'], data['High'], data['Low'], data['Close'])
        return bop

    @classmethod
    def CCI(self, data, **args):
        """
        Commodity Channel Index

        args:
            data
            period
        """
        cci = ta.CCI(data['High'], data['Low'], data['Close'], timeperiod=args['period'])
        return cci

    @classmethod
    def CMO(self, data, **args):
        """
        Chande Momentum Oscillator

        args:
            data
            period
            apply_to
        """
        try:
            cmo = ta.CMO(data['apply_to'], timeperiod=args['period'])
        except KeyError:
            cmo = ta.CMO(data['Close'], timeperiod=args['period'])
        return cmo

    @classmethod
    def DX(self, data, **args):
        """
        Directional Movement Index.

        args:
            data
            period
        """
        dx = ta.DX(data['High'], data['Low'], data['Close'], timeperiod=args['period'])
        return dx

    @classmethod
    def DMIPlus(self, data, period=14, **args):
        """
        Directional Movement Index. Positive.
        args:
            data
            period
        """
        plus_di = ta.MINUS_DI(data['High'], data['Low'], data['Close'], timeperiod=period)
        return plus_di

    @classmethod
    def DMIMinus(self, data, period, **args):
        """
        Directional Movement Index. Negative.
        args:
            data
            period
        """
        minus_di = ta.MINUS_DI(data['High'], data['Low'], data['Close'], timeperiod=period)
        return minus_di

    @classmethod
    def MFI(self, data, **args):
        """
        Money Flow Index

        args:
            data
            period
        """
        mfi = ta.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=args['period'])
        return mfi

    @classmethod
    def Momentum(self, data, **args):
        """
        Momentum.

        args:
            data
            period
            apply_to
        """
        try:
            mom = ta.MOM(data['apply_to'], timeperiod=args['period'])
        except KeyError:
            mom = ta.MOM(data['Close'], timeperiod=args['period'])
        return mom

    @classmethod
    def PPO(self, data, **args):
        """
        Percentage Price Oscillator

        args:
            data
            apply_to
            fast
            slow
        """
        try:
            ppo = ta.PPO(data[args['apply_to']], fastperiod=args['fast'], slowperiod=args['slow'], matype=0)
        except KeyError:
            ppo = ta.PPO(data['Close'], fastperiod=args['fast'], slowperiod=args['slow'], matype=0)
        return ppo

    @classmethod
    def RSI(self, data, **args):
        """
        Relative Strength Index.
        args:
            data
            period
            apply_to
        """
        try:
            rsi = ta.RSI(data[args['apply_to']], timeperiod=args['period'])
        except KeyError:
            rsi = ta.RSI(data['Close'], timeperiod=args['period'])
        return rsi

    @classmethod
    def UltimateOsc(self, data, **args):
        """
        Ultimate Oscillator.

        args:
            data
            period1
            period2
            period3
        """
        osc = ta.ULTSOC(data['High'], data['Low'], data['Close'], timeperiod1=args['period1'], timeperiod2=args['period2'], timeperiod3=args['period3'])
        return osc

    @classmethod
    def WilliamsR(self, data, **args):
        """
        Williams' %R.

        args:
            data
            period
        """
        will = ta.WILLR(data['High'], data['Low'], data['Close'], timeperiod=args['period'])
        return will

    @classmethod
    def ChaikinAD(self, data, **args):
        """
        Chaikin Accumulation Distribution Line (ADL).

        args:
            data
        """
        ad = ta.AD(data['High'], data['Low'], data['Close'], data['Volume'])
        return ad

    @classmethod
    def ChaikinADOsc(self, data, fast=20, slow=50, **args):
        """
        Chaikin ADL Oscillator.

        args:
            data
            fast: fast MACD period
            slow: slow MACD period
        """
        adosc = ta.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], fastperiod=fast, slowperiod=slow)
        return adosc

    @classmethod
    def ChaikinMoneyFlow(self, data, period=20, **args):
        """
        Chaikin Money Flow (CMF)
        """
        cmf = self.MoneyFlowVolume(data).rolling(period).mean() / data['Volume'].rolling(period).mean()
        return cmf

    @classmethod
    def OBV(self, data, **args):
        """
        On Balance Volume.

        args:
            data
        """
        obv = ta.OBV(data['Close'], data['Volume'])
        return obv

    @classmethod
    def OBVMACD(self, data, **args):
        """
        MACD applied to OBV.
        args:
            data
            sl
        """
        obv = self.OBV(data)
        hist = self.MACDHistogram(obv, fastperiod=3, slowperiod=7, signalperiod=9)

        return hist

    @classmethod
    def AveragePrice(self, data, **args):
        """
        Average Price.
        """
        avg = ta.AVGPRICE(data['Open'], data['High'], data['Low'], data['Close'])
        return avg

    @classmethod
    def MedianPrice(self, data, **args):
        """
        Median Price.
        """
        med = ta.MEDPRICE(data['High'], data['Low'])
        return med

    @classmethod
    def TypicalPrice(self, data, **args):
        typ = ta.TYPPRICE(data['High'], data['Low'], data['Close'])
        return typ

    @classmethod
    def WeightedClosePrice(self, data, **args):
        """
        Weighted Close Price.
        """
        wcp = ta.WCLPRICE(data['High'], data['Low'], data['Close'])
        return wcp

    @classmethod
    def ATR(self, data, period):
        """
        Average True Range.

        args:
            data
            period
        """
        atr = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=period)
        return atr

    @classmethod
    def NATR(self, data, **args):
        """
        Normalised Average True Range.

        args:
            data
            period
        """
        natr = ta.NATR(data['High'], data['Low'], data['Close'], timeperiod=args['period'])
        return natr

    @classmethod
    def TRANGE(self, data, **args):
        """
        True Range
        """
        data['prev close'] = data['close'].shift(1)
        data['high-low'] = data['high'] - data['Llw']
        data['high-pc'] = abs(data['high'] - data['prev close'])
        data['low-pc'] = abs(data['low'] - data['prev close'])

        # True Range
        tr = data[['high-low', 'high-pc', 'low-pc']].max(axis=1)
        return tr

    @classmethod
    def ROCR(self, data, **args):
        """
        Rate Of Change Ratio.

        args:
            data
            period
        """
        rocr = ta.ROCR(data['Close'], timeperiod=args['period'])
        return rocr

    @classmethod
    def HT_DCP(self, data, **args):
        """
        Hilbert Transform - Dominant Cycle Period.

        args:
            data
            apply_to
        """
        try:
            dcp = ta.HT_DCPERIOD(data[args['apply_to']])
        except KeyError:
            dcp = ta.HT_DCPERIOD(data['Close'])
        return dcp

    @classmethod
    def HT_DCPh(self, data, **args):
        """
        Hilbert Transform - Dominant Cycle Phase.
        
        args:
            data
            apply_to
        """
        try:
            dcph = ta.HT_DCPHASE(data[args['apply_to']])
        except KeyError:
            dcph = ta.HT_DCPHASE(data['Close'])
        return dcph

    @classmethod
    def HT_PhasorRealComponent(self, data, **args):
        """
        Hilbert Transform - Phasor Real Component.

        args:
            data
            apply_to
        """
        try:
            real, im = ta.HT_PHASOR(data[args['apply_to']])
        except KeyError:
            real, im = ta.HT_PHASOR(data['Close'])
        return real

    @classmethod
    def HT_PhasorImaginaryComponent(self, data, **args):
        """
        Hilbert Transform - Phasor Real Component.

        args:
            data
            apply_to
        """
        try:
            real, im = ta.HT_PHASOR(data[args['apply_to']])
        except KeyError:
            real, im = ta.HT_PHASOR(data['Close'])
        return im

    @classmethod
    def HT_TrendMode(self, data, **args):
        """
        Hilbert Transform - Trend Mode.

        args:
            data
            apply_to
        """
        try:
            tm = ta.HT_TRENDMODE(data[args['apply_to']])
        except KeyError:
            tm = ta.HT_TRENDMODE(data['Close'])
        return tm

    @classmethod
    def TRIX(self, data, **args):
        """
        Rate of Change of a Triple Exponentially Smoothed Moving Average.

        Shows the percent rate of change of a triple exponentially smoothed moving
        average.

        http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix


        args:
            data
            period
            apply_to
        """
        try:
            data['EMA'] = self.EMA(data, period=args['period'], apply_to=args['apply_to'])
        except KeyError:
            data['EMA'] = self.EMA(data, period=args['period'], apply_to='Close')
        data['2EMA'] = self.EMA(data, period=args['period'], apply_to='EMA')
        ema3 = self.EMA(data, period=args['period'], apply_to='2EMA')
        trix = (ema3 - ema3.shift(1, fill_value=ema3.mean())) * 100 / ema3.shift(1, fill_value=ema3.mean())
        return trix

    @classmethod
    def MI(self, data, **args):
        """
        Mass Index.

        It uses the high-low range to identify trend reversals based on range
        expansions. It identifies range bulges that can foreshadow a reversal of
        the current trend.

        http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index

        Args:
            data
            slowperiod
            fastperiod
        """
        data['Amplitude'] = data['High']-data['Low']
        data['1EMA'] = self.EMA(data, period=args['slowperiod'], apply_to='Amplitude')
        data['2EMA'] = self.EMA(data, period=args['slowperiod'], apply_to='1EMA')
        mass = data['1EMA'] / data['2EMA']
        mass = mass.rolling(args['fastperiod'], min_periods=args['fastperiod']).sum()
        return mass

    @classmethod
    def DPO(self, data, period=20, **args):
        """
        Detrended Price Oscillator.

        Is an indicator designed to remove trend from price and make it easier to
        identify cycles.

        http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci

        Args:
            data
            period
            apply_to

        Returns:
            pandas.Series: New feature generated.
        """
        try:
            dpo = data[args['apply_to']].shift(int(0.5*period+1), fill_value=data[args['apply_to']].mean()) - data[args['apply_to']].rolling(period, min_periods=period).mean()
        except KeyError:
            dpo = data['Close'].shift(int(0.5*period+1), fill_value=data['Close'].mean()) - data['Close'].rolling(period, min_periods=period).mean()
        return dpo

    @classmethod
    def KijunSen(self, data, **args):
        """
        Ichimoku Kinko Hyo - Kijun Sen (baseline).

        args: none
        """
        kijun = 0.5 * (
                data['High'].rolling(26, min_periods=26).max() +
                data['Low'].rolling(26, min_periods=26).min())
        return kijun

    @classmethod
    def TenkanSen(self, data, **args):
        """
        Ichimoku Kinko Hyo - Tenkan Sen (baseline).

        args: none
        """
        tenkan = 0.5 * (
                data['High'].rolling(9, min_periods=9).max() +
                data['Low'].rolling(9, min_periods=9).min())
        return tenkan
    
    @classmethod
    def Senkou(self, data, span='A', **args):
        """
        Ichimoku Kinko Hyo - Senkan Span A/B.

        args: none
        """
        if span == 'A':
            senkou_span_a = ((self.TenkanSen(data) + self.KijunSen(data)) / 2).shift(26)
            return senkou_span_a
        elif span == 'B':
            senkou_span_b = ((data['High'].rolling(52).max() + data['Low'].rolling(52).min()) / 2).shift(26)
            return senkou_span_b
        elif span == 'AB':
            senkou_span_a = ((self.TenkanSen(data) + self.KijunSen(data)) / 2).shift(26)
            senkou_span_b = ((data['High'].rolling(52).max() + data['Low'].rolling(52).min()) / 2).shift(26)
            return senkou_span_a, senkou_span_b
        else:
            raise Exception("Senkou Span can only have span='A' or span='B'")

    @classmethod
    def HeikinAshi(self, data, **args):
        """
        Heikin Ashi candles.

        args:
            data
        """
        # OLD IMPLEMENTATION (slower)
        # ha = data.copy()
        # for i in range(ha.shape[0]):
        #     if i > 0:
        #       ha.loc[ha.index[i],'Open'] = (data.iloc[i-1]['Open'] + data.iloc[i-1]['Close'])/2
          
        #     ha.loc[ha.index[i],'Close'] = (data.iloc[i]['Open'] + data.iloc[i]['Close'] + data.iloc[i]['Low'] +  data.iloc[i]['High'])/4
        # return ha

        # NEW IMPLEMENTATION
        ha = data.copy()
        ha['Close']=(data['Open']+ data['High']+ data['Low']+data['Close'])/4

        idx = ha.index.name
        ha.reset_index(inplace=True)

        for i in range(0, len(ha)):
            if i > 0:
                ha._set_value(i, 'Open', ((ha._get_value(i - 1, 'Open') + ha._get_value(i - 1, 'Close')) / 2))
            else:
                ha._set_value(i, 'Open', ((ha._get_value(i, 'Open') + ha._get_value(i, 'Close')) / 2))

        if idx:
            ha.set_index(idx, inplace=True)

        # ha['original_High'] = data['High']
        # ha['original_Low'] = data['Low']
        # ha['High']=ha[['Open','Close','original_High']].max(axis=1)
        # ha['Low']=ha[['Open','Close','original_Low']].min(axis=1)
        return ha

    @classmethod
    def Renko(self, data, **args):
        """ Renko charts """
        if args.get('fixed'):
            # renko with fixed value
            try:
                df = renko(data, fixed=args['fixed'], custom=args['custom'])
            except KeyError:
                df = renko(data, fixed=args['fixed'])
        elif args.get('atr'):
            # renko based on average true range
            bricks_atr = renko(data,
                               atr=args['atr'])
            df = pd.DataFrame(bricks_atr)
        return df

    @classmethod
    def Colour(self, data, zerocross=False):
        "Returns 1 or -1 based on candle colour."
        if zerocross:
            c = np.zeros(data.shape[0])
            c[data >= 0] = 1
            c[data < 0] = -1
        else:
            c = np.zeros(data.shape[0])
            c[data['Open'] > data['Close']] = -1
            c[data['Open'] <= data['Close']] = 1
        return c

    @classmethod
    def BB(self, data, period, dev=2):
        """
        Bollinger Bands.

        args:
            data: pandas DataFrame containing OHLC
            period: lookback period
            dev: deviation

        returns:
            lower, middle, upper band

        raises:
            None
        """
        std = data['Close'].rolling(window=period).std()
        middle = data['Close'].rolling(window=period).mean()
        lower = middle - dev*std
        upper = middle + dev*std

        return lower, middle, upper

    @classmethod
    def KC(self, data, period, dev=1.5):
        """
        Keltner Channels. (1960)

        Same as Bollinger Bands but instead of standard deviations of Close
        price it uses standard deviations of the ATR.

        args:
            data: pandas DataFrame containing OHLC
            period: lookback period
            dev: number of deviations away from middle band.

        returns:
            lower, middle, upper channel
        """
        atr = self.ATR(data, period)
        middle = data['Close'].rolling(window=period).mean()
        lower = middle - dev*atr
        upper = middle + dev*atr

        return lower, middle, upper

    @classmethod
    def TTM(self, data, bbperiod, kcperiod, **args):
        """
        Returns TTM squeeze indicator from hackingthemarkets on YouTube.
        True represents a squeeze.
        False represents that there is no squeeze.

        args:
            data: OHLC data
            bbperiod: lookback period for Bollinger Bands
            kcperiod: lookback period for Keltner Channels
            bbdev: deviations multiple for BB
            kcdev: deviations multiple for KC

        returns:
            list of booleans representing TTM squeeze for each candle

        raises:
            None
        """
        # Lower, middle and upper bands for BB and KC indicators
        try:
            bbl, bbm, bbu = self.BB(data, bbperiod, args['bbdev'])
        except KeyError:
            bbl, bbm, bbu = self.BB(data, bbperiod)

        try:
            kcl, kcm, kcu = self.KC(data, kcperiod, args['kcdev'])
        except KeyError:
            kcl, kcm, kcu = self.KC(data, kcperiod)

        # Check squeeze:
        ttm = (kcu > bbu) & (kcl < bbl)
        return ttm

    @classmethod
    def volumeHasSpike(self, data, n=30, spike=2):
        """
        Checks if currency pairs have had a spike in volume in the last n candles
        """
        if data.iloc[-n:]['Volume'].max() > spike*data.iloc[:-n]['Volume'].max():
            return True
        else:
            return False

    @classmethod
    def relativePositionOfCandles(self, data):
        """
        Tag candles with a state between:

        up, down, reverse-up, reverse-down, reverse-up2, reverse-down2, indecision, indecision2
        or undefined. ( U-D-RU-RD-RU2-RD2-I-I2-X )

        States are defined based on position relative to previous candlestick (Higher Highs or Lower Lows etc).
        """
        state = ['X' for i in range(data.shape[0])]

        # Identify state for each candle based on previous candle's state
        for i in range(2, data.shape[0]):
            if state[i-1] == 'X':
                if self.HH(data, i) and self.HL(data, i):
                    state[i] = 'U'
                elif self.LH(data, i) and self.LL(data, i):
                    state[i] = 'D'
                elif self.LH(data, i) and self.HL(data, i):
                    if self.LH(data, i-1) and self.HL(data, i-1):
                        state[i] = 'I2'
                    else:
                        state[i] = 'I'
                elif self.HH(data, i) and self.LL(data, i):
                    if self.greenCandle(data, i):
                        state[i] = 'RU2'
                    elif self.redCandle(data, i):
                        state[i] = 'RD2'
            elif state[i-1] == 'U':
                if self.HH(data, i) and self.HL(data, i):
                    state[i] = 'U'
                elif self.LH(data, i) and self.LL(data, i):
                    state[i] = 'RD'
                elif self.LH(data, i) and self.HL(data, i):
                    if self.LH(data, i-1) and self.HL(data, i-1):
                        state[i] = 'I2'
                    else:
                        state[i] = 'I'
                elif self.HH(data, i) and self.LL(data, i):
                    state[i] = 'RU'
            elif state[i-1] == 'D':
                if self.HH(data, i) and self.HL(data, i):
                    state[i] = 'RU'
                elif self.LH(data, i) and self.LL(data, i):
                    state[i] = 'D'
                elif self.LH(data, i) and self.HL(data, i):
                    if self.LH(data, i-1) and self.HL(data, i-1):
                        state[i] = 'I2'
                    else:
                        state[i] = 'I'
                elif self.HH(data, i) and self.LL(data, i):
                    state[i] = 'RU'
            elif state[i-1] == 'RU' or state[i-1] == 'RU2':
                if self.HH(data, i) and self.HL(data, i):
                    state[i] = 'U'
                elif self.LH(data, i) and self.LL(data, i):
                    state[i] = 'RD'
                elif self.LH(data, i) and self.HL(data, i):
                    if self.LH(data, i-1) and self.HL(data, i-1):
                        state[i] = 'I2'
                    else:
                        state[i] = 'I'
                elif self.HH(data, i) and self.LL(data, i):
                    if self.greenCandle(data, i):
                        state[i] = 'RU2'
                    elif self.redCandle(data, i):
                        state[i] = 'RD2'
            elif state[i-1] == 'RD' or state[i-1] == 'RD2':
                if self.HH(data, i) and self.HL(data, i):
                    state[i] = 'RU'
                elif self.LH(data, i) and self.LL(data, i):
                    state[i] = 'D'
                elif self.LH(data, i) and self.HL(data, i):
                    if self.LH(data, i-1) and self.HL(data, i-1):
                        state[i] = 'I'
                    else:
                        state[i] = 'I'
                elif self.HH(data, i) and self.LL(data, i):
                    if self.greenCandle(data, i):
                        state[i] = 'RU2'
                    elif self.redCandle(data, i):
                        state[i] = 'RD2'
            elif state[i-1] == 'I':
                if self.HH(data, i) and self.HL(data, i):
                    state[i] = 'RU'
                elif self.LH(data, i) and self.LL(data, i):
                    state[i] = 'RD'
                elif self.LH(data, i) and self.HL(data, i):
                    state[i] = 'I2'
                elif self.HH(data, i) and self.LL(data, i):
                    if self.greenCandle(data, i):
                        state[i] = 'RU2'
                    elif self.redCandle(data, i):
                        state[i] = 'RD2'
            elif state[i-1] == 'I2':
                if self.HH(data, i) and self.HL(data, i):
                    state[i] = 'RU'
                elif self.LH(data, i) and self.LL(data, i):
                    state[i] = 'RD'
                elif self.LH(data, i) and self.HL(data, i):
                    state[i] = 'I2'
                elif self.HH(data, i) and self.LL(data, i):
                    if self.greenCandle(data, i):
                        state[i] = 'RU2'
                    elif self.redCandle(data, i):
                        state[i] = 'RD2'
            else:
                print(f"Strategy FSM in unkown state: {state[i]}")
                exit()
        return state

    @classmethod
    def relativeCandlesReversalPatterns(self, data):
        """
        possible values: [-2, -1, 1, 2]

        returns: (only returns one value for last candle)
             0 when there is no pattern
             1 when there is a buy sequence,
             2 when there is a doubtful buy sequence,
            -1 when there is a sell sequence,
            -2 when there is a doubtful sell sequence,

        """
        tags = self.relativePositionOfCandles(data.iloc[-5:])
        state2, state1, state0 = tags[-3], tags[-2], tags[-1]
        buy_sequences = [
            state1 == 'D'  and state0 == 'RU',
            state2 == 'D'  and state1 == 'I' and state0 == 'RU',
            state1 == 'RD' and state0 == 'RU',
            state2 == 'RD' and state1 == 'I' and state0 == 'RU',
        ]

        buy_doubtful_sequences = [
            # state1 == 'D'  and state0 == 'RU2',
            state1 == 'D'  and state0 == 'RD2',
            # state2 == 'D'  and state1 == 'I' and state0 == 'RU2',
            state2 == 'D'  and state1 == 'I' and state0 == 'RD2',
            # state1 == 'RD' and state0 == 'RU2',
            state1 == 'RD' and state0 == 'RD2',
            # state2 == 'RD' and state1 == 'I' and state0 == 'RU2',
            state2 == 'RD' and state1 == 'I' and state0 == 'RD2',
        ]

        sell_sequences = [
            state1 == 'U'  and state0 == 'RD',
            state2 == 'U'  and state1 == 'I' and state0 == 'RD',
            state1 == 'RU' and state0 == 'RD',
            state2 == 'RU' and state1 == 'I' and state0 == 'RD',
        ]

        sell_doubtful_sequences = [
            # state1 == 'U'  and state0 == 'RD2',
            state1 == 'U'  and state0 == 'RU2',
            # state2 == 'U'  and state1 == 'I' and state0 == 'RD2',
            state2 == 'U'  and state1 == 'I' and state0 == 'RU2',
            # state1 == 'RU' and state0 == 'RD2',
            state1 == 'RU' and state0 == 'RU2',
            # state2 == 'RU' and state1 == 'I' and state0 == 'RD2',
            state2 == 'RU' and state1 == 'I' and state0 == 'RU2',
        ]

        if any(buy_sequences):
            return 1
        elif any(buy_doubtful_sequences):
            return 2
        elif any(sell_sequences):
            return -1
        elif any(sell_doubtful_sequences):
            return -2
        else:
            return 0

    @classmethod
    def relativeCandlesPhases(self, data, **args):
        """
        Direction Phases based on relative candles.
        args:
            data
        """
        tags = self.relativePositionOfCandles(data)
        phase = np.zeros(data.shape[0])
        
        phase[0] = 1 if self.greenCandle(data,0) else -1
        for i in range(1, 3):
            phase[i] = phase[i-1]

        for i in range(3,data.shape[0]):
            # up_sequences = [
            #     tags[i-2]=='RU' and tags[i-1]=='I' and tags[i]=='RU',
            #     tags[i-2]=='RU' and tags[i-1]=='I' and tags[i]=='RU2',
            #     tags[i-1]=='RU' and tags[i]=='U',
            #     tags[i-1]=='RU' and tags[i]=='RU2',
            #     tags[i-2]=='RU2' and tags[i-1]=='I' and tags[i]=='RU',
            #     tags[i-2]=='RU2' and tags[i-1]=='I' and tags[i]=='RU2',
            #     tags[i-1]=='RU2' and tags[i]=='U',
            #     tags[i-1]=='RU2' and tags[i]=='RU2',
            # ]
            # down_sequences = [
            #     tags[i-2]=='RD' and tags[i-1]=='I' and tags[i]=='RD',
            #     tags[i-2]=='RD' and tags[i-1]=='I' and tags[i]=='RD2',
            #     tags[i-1]=='RD' and tags[i]=='D',
            #     tags[i-1]=='RD' and tags[i]=='RD2',
            #     tags[i-2]=='RD2' and tags[i-1]=='I' and tags[i]=='RD',
            #     tags[i-2]=='RD2' and tags[i-1]=='I' and tags[i]=='RD2',
            #     tags[i-1]=='RD2' and tags[i]=='D',
            #     tags[i-1]=='RD2' and tags[i]=='RD2',
            # ]
            state2, state1, state0 = tags[i-2], tags[i-1], tags[i]
            up_sequences = [
                state1 == 'D'  and state0 == 'RU',
                state2 == 'D'  and state1 == 'I' and state0 == 'RU',
                state1 == 'D'  and state0 == 'RU2',
                state2 == 'D'  and state1 == 'I' and state0 == 'RU2',
                state1 == 'RD' and state0 == 'RU',
                state2 == 'RD' and state1 == 'I' and state0 == 'RU',
                state1 == 'RD' and state0 == 'RU2',
                state2 == 'RD' and state1 == 'I' and state0 == 'RU2',
                state1 == 'RD2' and state0 == 'RU',
                state2 == 'RD2' and state1 == 'I' and state0 == 'RU',
                state1 == 'RD2' and state0 == 'RU2',
                state2 == 'RD2' and state1 == 'I' and state0 == 'RU2',
                state2 == 'I' and state1 == 'I2' and state0 == 'RU',
                state2 == 'I' and state1 == 'I2' and state0 == 'RU2',

            ]

            down_sequences = [
                state1 == 'U'  and state0 == 'RD',
                state2 == 'U'  and state1 == 'I' and state0 == 'RD',
                state1 == 'U'  and state0 == 'RD2',
                state2 == 'U'  and state1 == 'I' and state0 == 'RD2',
                state1 == 'RU' and state0 == 'RD',
                state2 == 'RU' and state1 == 'I' and state0 == 'RD',
                state1 == 'RU' and state0 == 'RD2',
                state2 == 'RU' and state1 == 'I' and state0 == 'RD2',
                state1 == 'RU2' and state0 == 'RD',
                state2 == 'RU2' and state1 == 'I' and state0 == 'RD',
                state1 == 'RU2' and state0 == 'RD2',
                state2 == 'RU2' and state1 == 'I' and state0 == 'RD2',
                state2 == 'I' and state1 == 'I2' and state0 == 'RD',
                state2 == 'I' and state1 == 'I2' and state0 == 'RD2',
            ]

            # Check phase up sequence:
            if any(up_sequences):
                phase[i] = 1
            # Check phase down sequence:
            elif any(down_sequences):
                phase[i] = -1
            else:
                phase[i] = phase[i-1]
        return phase

    @classmethod
    def phaseChanges(self, data, nphases=4):
        """
        Returns indexes where value changes on an indicator (data).

        args:
            data: array containing a discrete set of values (your indicator)
            nphases: Set to -1 if you want the phase changes of entire data.
        """
        indexes = []
        subtotal = 0

        for i in np.arange(len(data)-2, 0, -1):
            if data[i] != data[i+1]:
                indexes.append(i+1)
                subtotal += 1
                if subtotal == nphases:
                    break

        return [0] + indexes[::-1]

    @classmethod
    def Cycles(self, data) -> pd.Series:
        """ Cycles Indicator by Marc Goulding.

        Cycles:   A    B    CC    C    D
                 -A   -B   -CC   -C   -D
        """
        possible_states = [
             "A",    "B",    "CC",    "C",    "D",
            "-A",   "-B",   "-CC",   "-C",   "-D", "X"  #  X = unknown. Only used at start
        ]

        phases = self.relativeCandlesPhases(data)

        current_state = "X"

        cycles = pd.Series(
            np.zeros(data.shape[0])
        ).astype(str)


        for i in range(0, data.shape[0]):
            if current_state == "A":
                if phases[i] == 1:
                    current_state = "A"
                    cycles[i] = current_state
                elif phases[i] == -1:
                    current_state = "B"
                    cycles[i] = current_state

            elif current_state == "B":
                phaseIndexes = self.phaseChanges(phases[:i], nphases = 2)
                minA = data.iloc[phaseIndexes[0]:phaseIndexes[1]]['Low'].min()
                try:
                    minB = data.iloc[phaseIndexes[1]:i+1]['Low'].min()
                except IndexError:
                    minB = data.iloc[phaseIndexes[1]:]['Low'].min()

                if minA < minB:
                    if phases[i] == 1:
                        current_state = "CC"
                        cycles[i] = current_state
                    elif phases[i] == -1:
                        current_state = "B"
                        cycles[i] = current_state
                else:
                    current_state = "-A"
                    cycles[i] = current_state

            elif current_state == "CC":
                phaseIndexes = self.phaseChanges(phases[:i], nphases = 3)
                try:
                    maxCC = data.iloc[phaseIndexes[-1]:i+1]['High'].max()
                except IndexError:
                    maxCC = data.iloc[phaseIndexes[-1]:]['High'].max()
                maxAB = data.iloc[phaseIndexes[0]:phaseIndexes[-1]]['High'].max()
                if maxCC > maxAB:
                    current_state = "C"
                    cycles[i] = current_state
                else:
                    if phases[i] == 1:
                        current_state = "CC"
                        cycles[i] = current_state
                    elif phases[i] == -1:
                        current_state = "-CC"
                        cycles[i] = current_state

            elif current_state == "C":
                if phases[i] == 1:
                    current_state = "C"
                    cycles[i] = current_state
                elif phases[i] == -1:
                    current_state = "D"
                    cycles[i] = current_state

            elif current_state == "D":
                phaseIndexes = self.phaseChanges(phases[:i], nphases = 3)
                minBC = data.iloc[phaseIndexes[0]:phaseIndexes[-1]]['Low'].min()
                try:
                    minD = data.iloc[phaseIndexes[-1]:i+1]['Low'].min()
                except IndexError:
                    minD = data.iloc[phaseIndexes[-1]:]['Low'].min()
                if minBC < minD:
                    if phases[i] == 1:
                        current_state = "CC"
                        cycles[i] = current_state
                    elif phases[i] == -1:
                        current_state = "D"
                        cycles[i] = current_state
                else:
                    current_state = "-A"
                    cycles[i] = current_state

            elif current_state == "-A":
                if phases[i] == 1:
                    current_state = "-B"
                    cycles[i] = current_state
                elif phases[i] == -1:
                    current_state = "-A"
                    cycles[i] = current_state

            elif current_state == "-B":
                phaseIndexes = self.phaseChanges(phases[:i], nphases = 2)
                try:
                    maxB = data.iloc[phaseIndexes[1]:i+1]['High'].max()
                except IndexError:
                    maxB = data.iloc[phaseIndexes[1]:]['High'].max()
                maxA = data.iloc[phaseIndexes[0]:phaseIndexes[1]]['High'].max()
                if maxA < maxB:
                    if phases[i] == 1:
                        current_state = "-B"
                        cycles[i] = current_state
                    elif phases[i] == -1:
                        current_state = "A"
                        cycles[i] = current_state
                else:
                    current_state = "-CC"
                    cycles[i] = current_state

            elif current_state == "-CC":
                phaseIndexes = self.phaseChanges(phases[:i], nphases = 3)
                try:
                    minCC = data.iloc[phaseIndexes[-1]:i+1]['Low'].min()
                except IndexError:
                    minCC = data.iloc[phaseIndexes[-1]:]['Low'].min()
                minAB = data.iloc[phaseIndexes[0]:phaseIndexes[-1]]['Low'].min()
                if minCC < minAB:
                    current_state = "-C"
                    cycles[i] = current_state
                else:
                    if phases[i] == 1:
                        current_state = "CC"
                        cycles[i] = current_state
                    elif phases[i] == -1:
                        current_state = "-CC"
                        cycles[i] = current_state

            elif current_state == "-C":
                if phases[i] == 1:
                    current_state = "-D"
                    cycles[i] = current_state
                elif phases[i] == -1:
                    current_state = "-C"
                    cycles[i] = current_state

            elif current_state == "-D":
                phaseIndexes = self.phaseChanges(phases[:i], nphases = 3)
                try:
                    maxD = data.iloc[phaseIndexes[-1]:i+1]['High'].max()
                except IndexError:
                    maxD = data.iloc[phaseIndexes[-1]:]['High'].max()
                maxBC = data.iloc[phaseIndexes[0]:phaseIndexes[-1]]['High'].max()

                if maxBC > maxD:
                    if phases[i] == 1:
                        current_state = "-D"
                        cycles[i] = current_state
                    elif phases[i] == -1:
                        current_state = "-CC"
                        cycles[i] = current_state
                else:
                    current_state = "A"
                    cycles[i] = current_state

            elif current_state == "X":

                if phases[i] == 1:
                    current_state = "A"
                    cycles[i] = current_state
                elif phases[i] == -1:
                    current_state = "-A"
                    cycles[i] = current_state

        return cycles

    @classmethod
    def trendingCycles(self, data):
        """
        Checks if there are 2 cycles down or 2 cycles up.
        
        Returns -1, 0 or 1 corresponding to downtrend, nothing and uptrend respectively.

        Only gives cycles during a phase 2 in the trend.
        If next phase 1 in trend has begun it will return False until phase 2 begins.
        """
        class Phase:
            def __init__(self):
                self.min = -1
                self.max = -1
            def updateMax(high):
                if high>self.max:
                    self.max = high
            def updateMin(low):
                if low<self.min:
                    self.min = low

        phases = self.phases(data)

        i = data.shape[0]-1
        state = 4  # FSM state to read last 4 phases in data
        phase = [Phase() for i in range(4)]
        phase[state-1].max = data.iloc[i]['High']
        phase[state-1].min = data.iloc[i]['Low']

        while state > 0:
            i -= 1
            if phases[i] == phases[i+1]:
                phase[state-1].updateMax(data.iloc[i]['High'])
                phase[state-1].updateMin(data.iloc[i]['Low'])
            else:
                state -= 1
                if state == 0:
                    break
                phase[state-1].max = data.iloc[i]['High']
                phase[state-1].min = data.iloc[i]['Low']

        up_conditions = [
            # Last phase is an uptrend phase 2 going down
            phases[-1] == -1,
            # Higher Highs
            max(phase[0].max, phase[1].max) < max(phase[2].max, phase[3].max),
            # Higher Lows
            min(phase[0].min, phase[1].min) < min(phase[2].min, phase[3].min),
        ]
        down_conditions = [
            # Last phase is a downtrend phase 2 going up
            phases[-1] == 1,
            # Lower Highs
            max(phase[0].max, phase[1].max) > max(phase[2].max, phase[3].max),
            # Lower Lows
            min(phase[0].min, phase[1].min) > min(phase[2].min, phase[3].min),
        ]

        # Check 2 cycles up
        if all(up_conditions):
            return -1
        # Check 2 cycles down
        elif all(down_conditions):
            return 1
        else:
            return 0

    @classmethod
    def HH(self, data, i) -> bool:
        """ Last 2 candles make a Higher High """
        return self.in_order(data.iloc[i]['High'], data.iloc[i-1]['High'], is_value=3)

    @classmethod
    def HL(self, data, i) -> bool:
        """ Last 2 candles make a Higher Low """
        return self.in_order(data.iloc[i]['Low'], data.iloc[i-1]['Low'], is_value=3)

    @classmethod
    def LL(self, data, i) -> bool:
        """ Last 2 candles make a Lower Low """
        return self.in_order(data.iloc[i-1]['Low'], data.iloc[i]['Low'], is_value=3)

    @classmethod
    def LH(self, data, i) -> bool:
        """ Last 2 candles make a Lower High """
        return self.in_order(data.iloc[i-1]['High'], data.iloc[i]['High'], is_value=3)

    @classmethod
    def in_order(self, data1, data2, is_value=0) -> bool:
        """ Check if last candle in data1 is above last candle in data2"""
        if not is_value:
            return data1.iloc[-1] > data2.iloc[-1]
        elif is_value == 1:
            return data1 > data2.iloc[-1]
        elif is_value == 2:
            return data1.iloc[-1] > data2
        elif is_value == 3:
            return data1 > data2

    @classmethod
    def greenCandle(self, data, i=-1) -> bool:
        """ Green Candle """
        return data.iloc[i]['Close'] > data.iloc[i]['Open']

    @classmethod
    def redCandle(self, data, i=-1) -> bool:
        """ Red Candle """
        return data.iloc[i]['Close'] < data.iloc[i]['Open']

    @classmethod
    def superTrend(self, data, period=14, multiplier=2, boolean: bool = True, **args):
        """ Super Trend indicator from Part Time Larry (YouTube) """
        supertrend = pandas_ta.supertrend(data['High'], data['Low'], data['Close'], period=period, multiplier=multiplier)
        if boolean:
            return supertrend.iloc[:, 1]
        else:
            return supertrend.iloc[:, 0]

    @classmethod
    def VolumeCluster(self, data, period=3, apply_to=None):
        aggregated_volumes = data['Volume'].rolling(period).sum()
        return aggregated_volumes

    @classmethod
    def ElliotWaveOscillator(self, data, shortperiod=5, longperiod=34, apply_to='Close', **args):
        """
        Elliot Wave Oscillator
        args:
            data
            shortperiod
            longperiod
            apply_to: default is 'Close'
        """
        try:
            ewo = self.SMA(data, period=shortperiod, apply_to=args['apply_to']) - self.SMA(data, period=longperiod, apply_to=args['apply_to'])
        except KeyError:
            ewo = self.SMA(data, period=shortperiod, apply_to='Close') - self.SMA(data, period=longperiod, apply_to='Close')
        return ewo

    @classmethod
    def LinearRegressionSlope(self, data, period, **args):
        """
        Linear Regression Slope
        
        args:
            data
            period: lookback window

        returns:
            slope: Slope of linear regression applied to window of size period

        """
        slope = ta.LINEARREG(data, period)
        return slope

    @classmethod
    def Squeeze(self, data, period=20, **args):
        """
        Squeeze Indicator from TradingView by LazyBear.
 
        source: https://www.tradingview.com/script/nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/

                val = linreg(data['Close']  -  np.mean(np.mean(data.loc[-lengthKC:,'High'].max(), data.loc[-lengthKC:,'Low']).max(),self.SMA(close,period=lengthKC)), 
                             lengthKC,0)

        args:
            data
            period: Refers to KClength in original tradingview indicator.

        returns:
            squeeze: pandas DataFrame with 1 column containing values of indicator
        """


        highest_high = data['High'].rolling(period).max()
        lowest_low = data['Low'].rolling(period).min()
        try:
            squeeze = data['Close'] - ta.LINEARREG(
                ((highest_high+lowest_low)/2 + self.SMA(data, period=period))/2,
                period
            )
        except Exception as e:
            # NOTE - Lookback period of data probably not large enough to calculate squeeze.
            # Returning zeros instead.
            # Exceptions mainly occur during multiple timeframe backtests
            squeeze = pd.Series(np.zeros(data.shape[0]))
        return squeeze

    @classmethod
    def EMASqueeze(self, data, sqz_period=9, emaperiod=25, **args):
        """ Squeeze applied to an EMA
        args:
            data: OHLCV data
            sqz_period: Squeeze period
            emaperiod: EMA period
        """
        ema = self.EMA(data, period=emaperiod)
        highest_high = ema.rolling(sqz_period).max()
        lowest_low = ema.rolling(sqz_period).min()
        sqz = data['Close'] - ta.LINEARREG(
            ((highest_high+lowest_low)/2 + self.EMA(data, period=emaperiod))/2,
            sqz_period
        )
        return sqz

    @classmethod
    def VolumeProfile(self, data, period=-1, num_samples=200, kde_factor=0.05, xaxis=False, group=1, **args):
        """
        Volume Profile

        args:
            data
            period: lookback window of indicator. A good value is 150.
            num_samples: nbins in volume profile histogram
            kde_factor: Kernel Density Estimator factor. Applied over Volume Profile to then find maxima. 
            xaxis: convert into a line with volume over time looking at close price.
            group: used only when xaxis=True. Group histogram into less nbins.

        retuns:
            prices
            volumes
        """
        if xaxis:
            vps = np.zeros(data.shape[0])
            for i in range(period, data.shape[0]):
                prices, volumes = self.VolumeProfile(data.iloc[i-period:i], -1, num_samples, kde_factor, xaxis=False)
                try:
                    vps[i] = volumes[prices <= data.iloc[i]['Close']][-1]
                except:
                    vps[i] = volumes[prices > data.iloc[i]['Close']][0]

            if group > 1:
                vps = self.group(vps, group)
            return vps
        else:
            if period == -1:
                kde = stats.gaussian_kde(data['Close'],weights=data['Volume'],bw_method=kde_factor)
                prices = np.linspace(data['Close'].min(),data['Close'].max(),num_samples)
                volumes = kde(prices)
            else:
                kde = stats.gaussian_kde(data.iloc[-period:]['Close'],weights=data.iloc[-period:]['Volume'],bw_method=kde_factor)
                prices = np.linspace(data.iloc[-period:]['Close'].min(),data.iloc[-period:]['Close'].max(),num_samples)
                volumes = kde(prices)
            return prices, volumes

    @classmethod
    def POC(self, data, period=150, num_samples=200, kde_factor=0.05, single_value=False, **args) -> np.array:
        """ Volume POC.
            Made by Superchiqui

        args:
            data
            period: 
            num_samples: nbins in volume profile histogram
            kde_factor: Kernel Density Estimator factor. Applied over Volume Profile to then find maxima. 
            single_value: True to obtain only last value. False to obtain numpy list.

        """
        if single_value:
            prices, kdy = self.VolumeProfile(data, period, num_samples, kde_factor)
            kdylist = kdy.tolist()
            pocindex = kdylist.index(max(kdylist))
            xpoc = prices[pocindex]
            
            return xpoc
        else:
            pocs = np.zeros(data.shape[0])
            for i in range(period, data.shape[0]):
                pocs[i] = self.POC(data.iloc[i-period:i], single_value=True)
            return pocs

    @classmethod
    def DistanceToPOC(self, data, period=150, num_samples=200, kde_factor=0.05, single_value=False, **args):
        """
        Current VPVR's distance to POC in %

        distance = VPVR / POC * 100
        """
        pass
    
    @classmethod
    def RVI(self, data, period, **args):
        """ Relative Volatility Index
        args:
            data
            period
        """
        avggains = data.rolling(period).std()[data['Open']<data['Close']]
        avglosses = data.rolling(period).std()[data['Open']>data['Close']]
        rvi = 0
        raise Exception("RVI feature not implemented in features.py")
        return rvi

    @classmethod
    def WeisWavesVolume(self, data, **args) -> pd.Series:
        """
        Weis Waves Volume Indicator.
        Cumulative volume resetting count when phase changes.
        A phase changes when there are two consecutive higher closes or two consecutive lower closes.
        """
        indexes = self.phaseChanges(self.RelativeCandlesPhases(data), nphases=-1)
        wwv = data['Volume'].copy()
        indexes = [0] + indexes + [data.shape[0]]
        for i in range(len(indexes)-1):
            wwv.iloc[indexes[i]:indexes[i+1]] = wwv.iloc[indexes[i]:indexes[i+1]].cumsum()
        return wwv

    @classmethod
    def MoneyFlowMultiplier(self, data, **args) -> pd.DataFrame:
        """
        Money Flow multiplier.

        source: https://www.tradingsim.com/day-trading/accumulation-distribution-indicator
        """
        MFM = ((data['Close']-data['Low']) - (data['High']-data['Close'])) / (data['High']-data['Low'])
        return MFM

    @classmethod
    def MoneyFlowVolume(self, data, **args) -> pd.DataFrame:
        """
        Money Flow Volume.

        Used with Wyckoff.

        source: https://www.tradingsim.com/day-trading/accumulation-distribution-indicator
        """
        MFV = self.MoneyFlowMultiplier(data) * data['Volume']
        return MFV

    @classmethod
    def ConvergenceDivergence(self, data, indicatordata, single_value=True, window=150, **args):
        """ Check 0-cross Oscillator for convergence or divergence on last 2 bells above or below 0.

            Only looks at last 4 cycles.

            args:
                data: price data
                indicatordata: 0-cross indicator array
                single_value: True returns 1 value. False returns a pandas with values for each datapoint.
                window: lookback window when single_value=False

            returns:
                1  -> Bullish divergence
                -1 -> Bearish divergence
                0  -> No divergence
        """
        if single_value:
            phases = self.Colour(indicatordata, zerocross=True)
            idx = self.phaseChanges(phases, nphases=4)
            if phases[-1] == 1:
                try:
                    # Find local maxima
                    maxpriceB = data.iloc[idx[-4]:idx[-3]]['High'].max()
                    maxpriceD = data.iloc[idx[-1]:]['High'].max()
                    maxB = indicatordata.iloc[idx[-4]:idx[-3]].max()
                    maxD = indicatordata.iloc[idx[-1]:].max()

                    # Check Divergence
                    if (maxpriceB<maxpriceD and maxB>maxD):
                        return -1
                    elif (maxpriceB>maxpriceD and maxB<maxD):
                        return -1
                    else:
                        return 0
                except IndexError:
                    return 0
            else:
                try:
                    # Find local minima
                    minpriceB = data.iloc[idx[1]:idx[2]]['Low'].min()
                    minpriceD = data.iloc[idx[-1]:]['Low'].min()
                    minB = indicatordata.iloc[idx[1]:idx[2]].min()
                    minD = indicatordata.iloc[idx[-1]:].min()

                    # Check Divergence
                    if (minpriceB<minpriceD and minB>minD):
                        return 1
                    elif (minpriceB>minpriceD and minB<minD):
                        return 1
                    else:
                        return 0
                except IndexError:
                    return 0
        else:
            divergences = np.zeros(data.shape[0])
            for i in range(window, data.shape[0]):
                subdata = data[i-window:i]
                subindicatordata = indicatordata[i-window:i]

                phases = self.Colour(subindicatordata, zerocross=True)
                idx = self.phaseChanges(phases, nphases=4)
                # print('idx:', idx)
                if phases[-1] == 1:
                    try:
                        # Find local maxima
                        maxpriceB = subdata.iloc[idx[-4]:idx[-3]]['High'].max()
                        maxpriceD = subdata.iloc[idx[-1]:]['High'].max()
                        maxB = subindicatordata.iloc[idx[-4]:idx[-3]].max()
                        maxD = subindicatordata.iloc[idx[-1]:].max()

                        # Check Divergence
                        if (maxpriceB<maxpriceD and maxB>maxD):
                            divergences[i] = -1
                        elif (maxpriceB>maxpriceD and maxB<maxD):
                            divergences[i] = -1
                        else:
                            divergences[i] = 0
                    except IndexError:
                        divergences[i] = 0
                else:
                    try:
                        # Find local minima
                        minpriceB = subdata.iloc[idx[1]:idx[2]]['Low'].min()
                        minpriceD = subdata.iloc[idx[-1]:]['Low'].min()
                        minB = subindicatordata.iloc[idx[1]:idx[2]].min()
                        minD = subindicatordata.iloc[idx[-1]:].min()

                        # Check Divergence
                        if (minpriceB<minpriceD and minB>minD):
                            divergences[i] = 1
                        elif (minpriceB>minpriceD and minB<minD):
                            divergences[i] = 1
                        else:
                            divergences[i] = 0
                    except IndexError:
                        divergences[i] = 0
            return divergences

    @classmethod
    def PivotPoints(self, data, **args):  
        """ Pivot Points.
            
            Best used on daily timeframe.

            args:
                data
        """
        PP = (data['High'] + data['Low'] + data['Close']) / 3
        R1 = 2 * PP - data['Low']
        S1 = 2 * PP - data['High']
        R2 = PP + data['High'] - data['Low'] 
        S2 = PP - data['High'] + data['Low']
        R3 = data['High'] + 2 * (PP - data['Low'])
        S3 = data['Low'] - 2 * (data['High'] - PP)
        psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
        PSR = pd.DataFrame(psr)
        return PSR

    @classmethod
    def EOM(self, data, period, ma=True, **args):
        """
        Ease of Movement.
        
        args:
            data
            period
            ma: bool. Toggles between returning EOM moving average or simply EOM.
        """ 
        eom = (data['High'].diff(1) + data['Low'].diff(1)) * (data['High'] - data['Low']) / (2 * data['Volume'])  
        if ma:
            eom_ma = pd.Series(pd.rolling_mean(eom, period))  
            return eom_ma
        else:
            return eom

    @classmethod
    def COPP(self, data, period, **args):  
        """
        Coppock Curve  
        """
        M = data['Close'].diff(int(period * 11 / 10) - 1)  
        N = data['Close'].shift(int(period * 11 / 10) - 1)  
        ROC1 = M / N  
        M = data['Close'].diff(int(period * 14 / 10) - 1)  
        N = data['Close'].shift(int(period * 14 / 10) - 1)  
        ROC2 = M / N  
        copp = pd.Series(pd.ewma(ROC1 + ROC2, span = period, min_periods = period))  
        return copp

    @classmethod
    def Stochastic(self, data, kperiod=14, dperiod=3, **args):
        """
        Stochastic Oscillator.

        args:
            data
            kperiod
            dperiod
        returns:
            % k
            % d
        """
        # data.ta.stoch(high='high', low='low', k=kperiod, d=dperiod, append=True)
        # stochk = data[data.columns[-2]]
        # stochd = data[data.columns[-1]]
        # return stochk, stochd
    
        high_roll = data["High"].rolling(kperiod).max()
        low_roll = data["Low"].rolling(kperiod).min()
        
        # Fast stochastic indicator
        num = data["Close"] - low_roll
        denom = high_roll - low_roll
        stochk = (num / denom) * 100
        
        # Slow stochastic indicator
        stochd = stochk.rolling(dperiod).mean()
        
        return stochk, stochd

    @classmethod
    def MACDdiv(self, data: pd.DataFrame, fastperiod=12, slowperiod=26, signalperiod=9, **args):
        """
        MACD Convergence Divergence indicator based on https://usethinkscript.com/threads/macd-divergence-indicator-for-thinkorswim.35/

        Assumes moving average is exponential.
        TODO - Allow for any type of moving average.

        args:
            fasperiod
            slowperiod
            signalperiod

        returns:

        """
        dif = self.MACDHistogram
        raise Exception("MACDdiv feature not implemented in features.py")
        return None
    
    @classmethod
    def ZigZag2(self, data: pd.DataFrame, period: int, **args) -> list:
        """
        ZigZag indicator.

        Translated from https://www.tradingview.com/script/mRbjBGdL-Double-Zig-Zag-with-HHLL/

        args:
            data
            period
        returns:
            zigzagpattern: ['H' or 'L', high/low price, index in dataframe]
        """

        highest_high = data['High'].rolling(period).max()
        lowest_low = data['Low'].rolling(period).min()
        new_high = data['High'] >= highest_high
        new_low = data['Low'] <= lowest_low
        new_high = new_high.astype(int)
        new_low = new_low.astype(int)

        direction = new_low.copy() * 0
        changed = new_low.copy() * 0

        zigzag_pattern = []
        for idx in range(1, len(data)):
            if new_high.iloc[idx] and not new_low.iloc[idx]:
                if direction.iloc[idx-1] != 1:
                    direction.iloc[idx] = 1
                    changed.iloc[idx] = 1
                elif direction.iloc[idx-1] == 1:
                    changed.iloc[idx] = 0
            elif not new_high.iloc[idx] and new_low.iloc[idx]:
                if direction.iloc[idx-1] != -1:
                    direction.iloc[idx] = -1
                    changed.iloc[idx] = True
                elif direction.iloc[idx-1] == -1:
                    changed.iloc[idx] = False

            if new_high.iloc[idx] or new_low.iloc[idx]:
                if changed.iloc[idx-1] or len(zigzag_pattern)==0:
                    if direction.iloc[idx-1] == 1:
                        # [H/L, open time, price, index]
                        pat = ['H', data['open time'].iloc[idx], data['High'].iloc[idx], idx]
                        zigzag_pattern.append(pat)
                    elif direction.iloc[idx-1] == -1:
                        pat = ['L', data['open time'].iloc[idx], data['Low'].iloc[idx], idx]
                        zigzag_pattern.append(pat)
                else:
                    if direction.iloc[idx-1] == 1 and data['High'].iloc[idx] > zigzag_pattern[-1][1]:
                        pat = ['H', data['open time'].iloc[idx], data['High'].iloc[idx], idx]
                        zigzag_pattern[-1] = pat
                    elif direction.iloc[idx-1] == -1 and data['Low'].iloc[idx] < zigzag_pattern[-1][1]:
                        pat = ['L', data['open time'].iloc[idx], data['Low'].iloc[idx], idx]
                        zigzag_pattern[-1] = pat
        return zigzag_pattern

    @classmethod
    def ZigZag(self, data: pd.DataFrame, period: int, **args):
        """
        ZigZag indicator.

        Original PineScript author: HeWhoMustNotBeNamed on TradingView

        args:
            data
            period

        returns:
            zzpivots: list containing pivot prices
            zzindex: list containing pivot index in data
            zzdirs: list containing pivot direction (1 or -1)
            zztimes: list containing pivot candle's open time
        """
        max_pivots = 1000
        highest_high = data['High'].rolling(period).max()
        lowest_low = data['Low'].rolling(period).min()
        new_high = data['High'] >= highest_high
        new_low = data['Low'] <= lowest_low
        phigh = new_high.copy()
        plow = new_low.copy()

        
        zzpivots, zzindex, zzdirs, zztimes = [],[],[],[]

        phigh.loc[new_high==True] = data['High'][new_high]
        plow.loc[new_low==True] = data['Low'][new_low]


        direction = new_low.copy() * 0
        changed = new_low.copy() * 0


        pivots = []
        for idx in range(1, len(data)):
            if new_high.iat[idx] and not new_low.iat[idx]:
                direction.iat[idx] = 1
            elif not new_high.iat[idx] and new_low.iat[idx]:
                direction.iat[idx] = -1
            else:
                direction.iat[idx] = direction.iat[idx-1]
            
            if direction.iat[idx] == direction.iat[idx-1]:
                changed.iat[idx] = 0
            else:
                changed.iat[idx] = 1


            # if phigh or plow
            if new_high.iat[idx] or new_low.iat[idx]:
                # value = dir == 1 ? phigh : plow
                # newDir = dir
                newpivot = phigh.iat[idx] if (direction.iat[idx] == 1) else plow.iat[idx]
                newidx = idx
                newtime = data['open time'].iat[idx]
                newdir = direction.iat[idx]
                # if not dirchanged and array.size(zigzagpivots) >= 1
                if direction.iat[idx] == direction.iat[idx-1] and len(zzpivots) >= 1:
                    # pivot = array.shift(zigzagpivots)
                    pivot = zzpivots.pop()
                    # pivotbar = array.shift(zigzagpivotbars)
                    pivot_index = zzindex.pop()
                    # pivotdir = array.shift(zigzagpivotdirs)
                    pivot_dir = zzdirs.pop()
                    pivot_time = zztimes.pop()
                    # useNewValues = value * pivotdir < pivot * pivotdir
                    # bar := useNewValues ? pivotbar : idx
                    if newpivot * pivot_dir < pivot * pivot_dir:
                        newpivot = pivot
                        newidx = pivot_index
                        newtime = pivot_time
                # bar

                # if array.size(zigzagpivots) >= 2
                if len(zzpivots) >= 2:
                    # LastPoint = array.get(zigzagpivots, 1)
                    lastpoint = zzpivots[-1]
                    # newDir :=  ? dir * 2 : dir
                    if direction.iat[idx] * newpivot > direction.iat[idx] * lastpoint:
                        newdir = direction.iat[idx]*2
                    else:
                        newdir = direction.iat[idx]

                    # newDir

                # array.unshift(zigzagpivots, value=value)
                zzpivots.append(newpivot)
                # array.unshift(zigzagpivotbars, idx)
                zzindex.append(newidx)
                # array.unshift(zigzagpivotdirs, newDir)
                zzdirs.append(newdir)

                zztimes.append(newtime)

                # if array.size(zigzagpivots) > max_pivot_size
                if len(zzpivots) > max_pivots:
                    # array.pop(zigzagpivots)
                    zzpivots.pop()
                    # array.pop(zigzagpivotbars)
                    zzindex.pop()
                    # array.pop(zigzagpivotdirs)
                    zzdirs.pop()
                    zztimes.pop()
            
                        
        # pprint(zzpivots)
        # pprint(zzindex)
        # pprint(zzdirs)
        # pprint(zztimes)
        # exit()
        return zzpivots, zzindex, zzdirs, zztimes

    @classmethod
    def HarmonicPatterns(self, data: pd.DataFrame, period: int, waitForConfirmation=False, only: bool=False, **args):
        """
        Harmonic Patterns

        Patterns available:
            ABCD
            AB=CD
            ABCDExt
            Gartley
            Crab
            DeepCrab
            Bat
            Butterfly
            Cypher
            3 Dives
            5-0
            Shark
            Double Top/Bottom
        args:
            data: OHLCV candlesticks as pandas DataFrame
            period
            only: if True, if only detects those we've explicitly passed as True
            crab=True
        returns:
            pattern_found: True/False
            pattern: String containing name of pattern / None
            direction: 1/-1 bullish or bearish
        """
        abcdClassic = args.get('abcdClassic', not only)
        abEQcd = args.get('abEQcd', not only)
        abcdExt = args.get('abcdExt', not only)
        gartley = args.get('gartley', not only)
        crab = args.get('crab', not only)
        deepCrab = args.get('deepCrab', not only)
        bat = args.get('bat', not only)
        butterfly = args.get('butterfly', not only)
        shark = args.get('shark', not only)
        cypher = args.get('cypher', not only)
        threeDrives = args.get('threeDrives', not only)
        fiveZero = args.get('fiveZero', not only)
        doubleBottomTop = args.get('doubleBottomTop', not only)

        if waitForConfirmation:
            data = data.iloc[:-1]
        
        start=0
        zzpivots, zzbars, zzdirs, zztimes = self.ZigZag(data, period)

        errorPercent = 10
        err_min = (100 - errorPercent) / 100
        err_max = (100 + errorPercent) / 100

        pattern_found = False
        pattern = None
        direction = None


        if len(zzpivots) >= 6+start:
            d = zzpivots[-1-start]
            c = zzpivots[-2-start]
            b = zzpivots[-3-start]
            a = zzpivots[-4-start]
            x = zzpivots[-5-start]
            y = zzpivots[-6-start]

            d_bar = zzbars[-1-start]
            c_bar = zzbars[-2-start]
            b_bar = zzbars[-3-start]
            a_bar = zzbars[-4-start]
            x_bar = zzbars[-5-start]
            y_bar = zzbars[-6-start]

            d_dir = zzdirs[-1-start]
            c_dir = zzdirs[-2-start]
            b_dir = zzdirs[-3-start]
            a_dir = zzdirs[-4-start]
            x_dir = zzdirs[-5-start]
            y_dir = zzdirs[-6-start]

            d_time = zztimes[-1-start]

            highpoint = max(x, a, b, c, d)
            lowpoint = min(x, a, b, c, d)
            direction = 1 if (d>c) else -1

            xab = abs(b-a)/abs(x-a)
            abc = abs(c-b)/abs(a-b)
            bcd = abs(d-c)/abs(b-c)
            xad = abs(d-a)/abs(x-a)
            yxa = abs(a-x)/abs(y-x)

            # Check zigzag appeared on this candle. If not wait until next candle
            if d_time != data['open time'].iat[-1]:
                return pattern_found, pattern, direction
            

            time_ratio = abs(c_bar-d_bar)/abs(a_bar-b_bar)
            price_ratio = abs(c-d)/abs(a-b)
            if a < b and a < c and c < b and c < d and a < d and b < d:
                abcdDirection = 1 
            elif a > b and a > c and c > b and c > d and a > d and b > d:
                abcdDirection = -1
            else:
                abcdDirection = 0
            
            risk = abs(b-d)
            reward = abs(c-d)
            riskreward = risk*100 / (risk + reward)

            if b < highpoint and b > lowpoint:
                # gartley
                if gartley and xab >= 0.618 * err_min and xab <= 0.618 * err_max and abc >= 0.382 * err_min and abc <= 0.886 * err_max and (bcd >= 1.272 * err_min and bcd <= 1.618 * err_max or xad >= 0.786 * err_min and xad <= 0.786 * err_max):
                    pattern = "gartley"
                    pattern_found = True
                # Crab
                if crab and xab >= 0.382 * err_min and xab <= 0.618 * err_max and abc >= 0.382 * err_min and abc <= 0.886 * err_max and (bcd >= 2.24 * err_min and bcd <= 3.618 * err_max or xad >= 1.618 * err_min and xad <= 1.618 * err_max):
                    pattern = "crab"
                    pattern_found = True
                # Deep Crab
                if deepCrab and xab >= 0.886 * err_min and xab <= 0.886 * err_max and abc >= 0.382 * err_min and abc <= 0.886 * err_max and (bcd >= 2.00 * err_min and bcd <= 3.618 * err_max or xad >= 1.618 * err_min and xad <= 1.618 * err_max):
                    pattern = "deepCrab"
                    pattern_found = True
                # Bat
                if bat and xab >= 0.382 * err_min and xab <= 0.50 * err_max and abc >= 0.382 * err_min and abc <= 0.886 * err_max and (bcd >= 1.618 * err_min and bcd <= 2.618 * err_max or xad >= 0.886 * err_min and xad <= 0.886 * err_max):
                    pattern = "bat"
                    pattern_found = True
                #Butterfly
                if butterfly and xab >= 0.786 * err_min and xab <= 0.786 * err_max and abc >= 0.382 * err_min and abc <= 0.886 * err_max and (bcd >= 1.618 * err_min and bcd <= 2.618 * err_max or xad >= 1.272 * err_min and xad <= 1.618 * err_max):
                    pattern = "butterfly"
                    pattern_found = True
                #Shark
                if shark and abc >= 1.13 * err_min and abc <= 1.618 * err_max and bcd >= 1.618 * err_min and bcd <= 2.24 * err_max and xad >= 0.886 * err_min and xad <= 1.13 * err_max:
                    pattern = "shark"
                    pattern_found = True
                #Cypher
                if cypher and xab >= 0.382 * err_min and xab <= 0.618 * err_max and abc >= 1.13 * err_min and abc <= 1.414 * err_max and (bcd >= 1.272 * err_min and bcd <= 2.00 * err_max or xad >= 0.786 * err_min and xad <= 0.786 * err_max):
                    pattern = "cypher"
                    pattern_found = True
            #3 drive
            if threeDrives and yxa >= 0.618 * err_min and yxa <= 0.618 * err_max and xab >= 1.27 * err_min and xab <= 1.618 * err_max and abc >= 0.618 * err_min and abc <= 0.618 * err_max and bcd >= 1.27 * err_min and bcd <= 1.618 * err_max:
                pattern = "threeDrives"
                pattern_found = True
            #5-0
            if fiveZero and xab >= 1.13 * err_min and xab <= 1.618 * err_max and abc >= 1.618 * err_min and abc <= 2.24 * err_max and bcd >= 0.5 * err_min and bcd <= 0.5 * err_max:
                pattern = "fiveZero"
                pattern_found = True
            #ABCD Classic
            if abcdClassic and abc >= 0.618 * err_min and abc <= 0.786 * err_max and bcd >= 1.272 * err_min and bcd <= 1.618 * err_max and abcdDirection != 0:
                pattern = "abcdClassic"
                pattern_found = True
            #AB=CD
            if abEQcd and time_ratio >= err_min and time_ratio <= err_max and price_ratio >= err_min and price_ratio <= err_max and abcdDirection != 0:
                pattern = "abEQcd"
                pattern_found = True
            #ABCD Ext
            if abcdExt and price_ratio >= 1.272 * err_min and price_ratio <= 1.618 * err_max and abc >= 0.618 * err_min and abc <= 0.786 * err_max and abcdDirection != 0:
                pattern = "abcdExt"
                pattern_found = True
            #Double Top/Bottom
            if (doubleBottomTop
                        and (d_dir == 1 and b_dir == 2 and c_dir == -1 
                        or d_dir == -1 and b_dir == -2 and c_dir == 1)
                    and riskPerReward < MaxRiskPerReward):
                pattern = "doubleBottomTop"
                pattern_found = True

        return pattern_found, pattern, direction

    @classmethod
    def SupportResistance(self, data, period=30, srtype="sr", **args):
        """
        Support and Resistance Lookback Based.

        source: https://www.tradingview.com/script/58YNCo7F-Support-and-Resistance-Lookback-based/

        args:
            data
            period
            srtype: can be sr, s or r
        returns:
            support
            resistance
        """
        returns = abs(data['Close'].pct_change()) * 100
        avgret100 = returns.mean() / 100

        srDist = 0.5 * avgret100

        # //----------------------------Support and Resistance Levels--------------------------------

        arrayHighs = np.ones(5) * data['Close'].iat[-1]
        arrayLows = np.ones(5) * data['Close'].iat[-1]

        srhigh = np.zeros(data.shape[0])
        srlow = np.zeros(data.shape[0])

        def f_round(x):
            """ Rounds x to increases/decreases of around 1 """
            decimals = round(2 - np.log10(x))
            p = np.power(10., decimals)
            return round(abs(x) * p) / p

        # //Peak-Valleys & Flat Levels
        highback = data['High'].shift(4).rolling(period).max()
        lowback = data['Low'].shift(4).rolling(period).min()

        for i in range(5, data.shape[0]):
            if (
                        highback.iat[i] > data['High'].iat[i]
                    and highback.iat[i] > data['High'].iat[i-1]
                    and highback.iat[i] > data['High'].iat[i-2]
                    and highback.iat[i] > data['High'].iat[i-3]
                    and highback.iat[i] > highback.iat[i-1]
                    and abs(1 - highback.iat[i] / arrayHighs[-1]) > srDist
                    and abs(1 - highback.iat[i] / arrayHighs[-2]) > srDist
                    and abs(1 - highback.iat[i] / arrayLows[-1]) > srDist
                    and abs(1 - highback.iat[i] / arrayLows[-2]) > srDist):
                srhigh[i] = f_round(data['High'].iat[i-4])
                arrayHighs = np.append(arrayHighs, srhigh[i])

            if (
                        lowback.iat[i] < data['Low'].iat[i]
                    and lowback.iat[i] < data['Low'].iat[i-1]
                    and lowback.iat[i] < data['Low'].iat[i-2]
                    and lowback.iat[i] < data['Low'].iat[i-3]
                    and lowback.iat[i] < lowback.iat[i-1]
                    and abs(1 - lowback.iat[i] / arrayLows[-1]) > srDist
                    and abs(1 - lowback.iat[i] / arrayLows[-2]) > srDist
                    and abs(1 - lowback.iat[i] / arrayHighs[-1]) > srDist
                    and abs(1 - lowback.iat[i] / arrayHighs[-2]) > srDist):
                srlow[i] = f_round(data['Low'].iat[i-4])
                arrayLows = np.append(arrayLows, srlow[i])

        def fill_zeros_with_last(arr):
            prev = np.arange(len(arr))
            prev[arr == 0] = 0
            prev = np.maximum.accumulate(prev)
            return arr[prev]

        srhigh = fill_zeros_with_last(srhigh)
        srlow = fill_zeros_with_last(srlow)

        if srtype == "s":
            return srlow
        elif srtype == "r":
            return srhigh
        else:
            return srlow, srhigh

    @classmethod
    def FibonacciRetracement(self, data, period, ratios=[0.236, 0.382, 0.500, 0.618, 0.786, 1.272, 1.414, 1.618], **args):
        """ Fibonacci Retracement """
        shownlevels = len(ratios)

                    
        #     diff = array.get(zigzag, 4) - array.get(zigzag, 2)
        #     stopit = false
        #     for x in range(0, len(fibo_ratios) - 1)
        #         if stopit and x > shownlevels
        #             break
        #         fibolevel =  array.get(zigzag, 2) + diff * array.get(fibo_ratios, x), 
        #         # Append fibolevel to list of levels for that candle

        #         if (dir == 1 and array.get(zigzag, 2) + diff * array.get(fibo_ratios, x) > array.get(zigzag, 0)) or
        #            (dir == -1 and array.get(zigzag, 2) + diff * array.get(fibo_ratios, x) < array.get(zigzag, 0))
        #             stopit = true















talib = {
    'ADX': TALib.ADX,
    'ADXR': TALib.ADXR,
    'APO': TALib.APO,
    'Average True Range': TALib.ATR,
    'ATR': TALib.ATR,
    'ADL': TALib.ChaikinAD,
    'AroonUp': TALib.AroonUp,
    'AroonDown': TALib.AroonDown,
    'AroonOsc': TALib.AroonOsc,
    'AveragePrice': TALib.AveragePrice,
    'ADLOscillator': TALib.ChaikinADOsc,
    "BollingerBands": TALib.BB,
    'BOP': TALib.BOP,
    'CMO': TALib.CMO,
    'CCI': TALib.CCI,
    'ChaikinMoneyFlow': TALib.ChaikinMoneyFlow,
    'ChaikinA/D': TALib.ChaikinAD,
    'ChaikinA/DOsc': TALib.ChaikinADOsc,
    'ConvergenceDivergence': TALib.ConvergenceDivergence,
    'DX': TALib.DX,
    'DMI+': TALib.DMIPlus,
    'DMI-': TALib.DMIMinus,
    'DEMA': TALib.DEMA,
    "DPO": TALib.DPO,
    'EMA': TALib.EMA,
    "ElliotWaveOscillator": TALib.ElliotWaveOscillator,
    "EWO": TALib.ElliotWaveOscillator,
    "EaseOfMovement": TALib.EOM,
    "EOM": TALib.EOM,
    'FractalHighs': TALib.FractalHighs,
    'FractalLows': TALib.FractalLows,
    'HarmonicPatterns': TALib.HarmonicPatterns,
    'HT_DominantCyclePeriod': TALib.HT_DCP,
    'HT_DominantCyclePhase': TALib.HT_DCPh,
    'HT_PhasorRealComponent': TALib.HT_PhasorRealComponent,
    'HT_PhasorImaginaryComponent': TALib.HT_PhasorImaginaryComponent,
    'HT_TrendMode': TALib.HT_TrendMode,
    'HT_Trendline': TALib.HT_Trendline,
    "HeikinAshi": TALib.HeikinAshi,
    "KeltnerChannel": TALib.KC,
    "KijunSen": TALib.KijunSen,
    'KAMA': TALib.KAMA,
    'MACDHistogram': TALib.MACDHistogram,
    'MAVP': TALib.MAVP,
    'MACDLine': TALib.MACD,
    'MACDSignal': TALib.MACDSignal,
    'MACD': TALib.MACD,
    'MidPoint': TALib. MidPoint,
    'MidPrice': TALib.MidPrice,
    "MassIndex": TALib.MI,
    "MI": TALib.MI,
    'MFI': TALib.MFI,
    'Momentum': TALib.Momentum,
    'MedianPrice': TALib.MedianPrice,
    'NATR': TALib.NATR,
    'OBV': TALib.OBV,
    'OBVMACD': TALib.OBVMACD,
    'ParabolicSAR': TALib.SAR,
    'POC': TALib.POC,
    'PPO': TALib.PPO,
    'PivotPoints': TALib.PivotPoints,
    'ROCR': TALib.ROCR,
    'ROC': TALib.ROC,
    'RSI': TALib.RSI,
    "Renko": TALib.Renko,
    "renko": TALib.Renko,
    "RelativePositionOfCandles": TALib.relativePositionOfCandles,
    "Squeeze": TALib.Squeeze,
    "SupportResistance": TALib.SupportResistance,
    'SAR': TALib.SAR,
    'LinearRegressionSlope': TALib.LinearRegressionSlope,
    'Slope': TALib.LinearRegressionSlope,
    'SMA': TALib.SMA,
    "SpikeinVolume": TALib.volumeHasSpike,
    "Stochastic": TALib.Stochastic,
    "SuperTrend": TALib.superTrend,
    'T3': TALib.T3,
    'TRIMA': TALib.TRIMA,
    'TypicalPrice': TALib.TypicalPrice,
    'TRANGE': TALib.TRANGE,
    'True Range': TALib.TRANGE,
    'TRIX': TALib.TRIX,
    "TenkanSen": TALib.TenkanSen,
    "TTM": TALib.TTM,
    'TEMA': TALib.TEMA,
    'UltimateOsc': TALib.UltimateOsc,
    'Volume': TALib.Volume,
    'VolumeProfile': TALib.VolumeProfile,
    "VolumeSpike": TALib.volumeHasSpike,
    "VolumeHas Spike": TALib.volumeHasSpike,
    "VolumeCluster": TALib.VolumeCluster,
    'WilliamsR': TALib.WilliamsR,
    'WeightedClosePrice': TALib.WeightedClosePrice,
    'WMA': TALib.WMA,
    "WWV": TALib.WeisWavesVolume,
    "EMASqueeze": TALib.EMASqueeze,
    "ZigZag": TALib.ZigZag,
"": None}

# pprint(talib)


if __name__ == '__main__':
    print("features.py can only be imported, not run!")
    pprint(talib.keys())
    exit(1)
