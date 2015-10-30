#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import division

from collections import defaultdict

import pandas as pd
import numpy as np

import warnings


def map_transaction(txn):
    """
    Maps a single transaction row to a dictionary.

    Parameters
    ----------
    txn : pd.DataFrame
        A single transaction object to convert to a dictionary.

    Returns
    -------
    dict
        Mapped transaction.
    """
    # sid can either be just a single value or a SID descriptor
    if isinstance(txn['sid'], dict):
        sid = txn['sid']['sid']
        symbol = txn['sid']['symbol']
    else:
        sid = txn['sid']
        symbol = None

    return {'sid': sid,
            'symbol': symbol,
            'price': txn['price'],
            'order_id': txn['order_id'],
            'amount': txn['amount'],
            'commission': txn['commission'],
            'dt': txn['dt']}


def make_transaction_frame(transactions):
    """
    Formats a transaction DataFrame.

    Parameters
    ----------
    transactions : pd.DataFrame
        Contains improperly formatted transactional data.

    Returns
    -------
    df : pd.DataFrame
        Daily transaction volume and dollar ammount.
         - See full explanation in tears.create_full_tear_sheet.
    """

    transaction_list = []
    for dt in transactions.index:
        txns = transactions.loc[dt]
        if len(txns) == 0:
            continue

        for txn in txns:
            txn = map_transaction(txn)
            transaction_list.append(txn)
    df = pd.DataFrame(sorted(transaction_list, key=lambda x: x['dt']))
    df['txn_dollars'] = -df['amount'] * df['price']

    df.index = list(map(pd.Timestamp, df.dt.values))
    return df


def get_txn_vol(transactions):
    """Extract daily transaction data from set of transaction objects.

    Parameters
    ----------
    transactions : pd.DataFrame
        Time series containing one row per symbol (and potentially
        duplicate datetime indices) and columns for amount and
        price.

    Returns
    -------
    pd.DataFrame
        Daily transaction volume and number of shares.
         - See full explanation in tears.create_full_tear_sheet.
    """
    transactions.index = transactions.index.normalize()

    amounts = transactions.amount.abs()
    prices = transactions.price
    values = amounts * prices
    daily_amounts = amounts.groupby(amounts.index).sum()
    daily_values = values.groupby(values.index).sum()
    daily_amounts.name = "txn_shares"
    daily_values.name = "txn_volume"
    return pd.concat([daily_values, daily_amounts], axis=1)


def adjust_returns_for_slippage(returns, turnover, slippage_bps):
    """Apply a slippage penalty for every dollar traded.

    Parameters
    ----------
    returns : pd.Series
        Time series of daily returns.
    turnover: pd.Series
        Time series of daily total of buys and sells
        divided by portfolio value.
            - See txn.get_turnover.
    slippage_bps: int/float
        Basis points of slippage to apply.

    Returns
    -------
    pd.Series
        Time series of daily returns, adjusted for slippage.
    """
    slippage = 0.0001 * slippage_bps
    # Only include returns in the period where the algo traded.
    trim_returns = returns.loc[turnover.index]
    return trim_returns - turnover * slippage


def create_txn_profits(transactions):
    """
    Compute per-trade profits.

    Generates a new transactions DataFrame with a profits column

    Parameters
    ----------
    transactions : pd.DataFrame
        Daily transaction volume and number of shares.
         - See full explanation in tears.create_full_tear_sheet.

    Returns
    -------
    profits_dts : pd.DataFrame
        DataFrame containing transactions and their profits, datetimes,
        amounts, current prices, prior prices, and symbols.
    """

    txn_descr = defaultdict(list)

    for symbol, transactions_sym in transactions.groupby('symbol'):
        transactions_sym = transactions_sym.reset_index()

        for i, (amount, price, dt) in transactions_sym.iloc[1:][['amount', 'price', 'date_time_utc']].iterrows():
            prev_amount, prev_price, prev_dt = transactions_sym.loc[
                i - 1, ['amount', 'price', 'date_time_utc']]
            profit = (price - prev_price) * -amount
            txn_descr['profits'].append(profit)
            txn_descr['dts'].append(dt - prev_dt)
            txn_descr['amounts'].append(amount)
            txn_descr['prices'].append(price)
            txn_descr['prev_prices'].append(prev_price)
            txn_descr['symbols'].append(symbol)

    profits_dts = pd.DataFrame(txn_descr)

    return profits_dts


def extract_roundtrips(transactions):
    transactions_split = split_trades(transactions)

    txn_descr = defaultdict(list)

    for sym, trans_sym in transactions_split.groupby('symbol'):
        amount_cumsum = trans_sym.amount.cumsum()
        assert np.all(np.abs(np.diff(np.sign(amount_cumsum)))
                      != 2), "Not all trades end on 0."
        exit_idx = np.where(amount_cumsum == 0)[0] + 1

        for trade_start, trade_end in zip(exit_idx, exit_idx[1:]):
            txn = trans_sym.iloc[trade_start:trade_end]

            if len(txn) == 0:
                continue

            pnl = txn.txn_dollars.sum()

            assert txn.amount.sum() == 0

            txn_descr['pnl'].append(pnl)
            txn_descr['symbol'].append(sym)

            txn_descr['duration'].append(txn.index[-1] - txn.index[0])

            txn_descr['open'].append(txn.index[0])
            txn_descr['close'].append(txn.index[-1])
            txn_descr['long'].append(txn.amount.iloc[0] > 0)

    if len(txn_descr) == 0:
        warnings.warn('No round-trip trades found. Aborting.', UserWarning)
        return pd.DataFrame([])

    txn_descr = pd.DataFrame(txn_descr)
    txn_descr['profitable'] = txn_descr.pnl > 0
    txn_descr['dollar_volume'] = txn_descr.pnl.abs()

    return txn_descr


def split_trades(transactions):
    trans_split = []
    for sym, trans_sym in transactions.groupby('symbol'):
        trans_sym.sort()

        while True:
            cum_amount = trans_sym.amount.cumsum()
            sign_flip = np.where(np.abs(np.diff(np.sign(cum_amount))) == 2)[0]

            if len(sign_flip) == 0:
                break  # all sign flips are converted

            sign_flip = sign_flip[0] + 2

            txn = trans_sym.iloc[:sign_flip]

            left_over_txn_amount = txn.amount.sum()

            assert left_over_txn_amount != 0

            split_txn_1 = txn.iloc[[-1]].copy()
            split_txn_2 = txn.iloc[[-1]].copy()

            split_txn_1['amount'] -= left_over_txn_amount
            split_txn_2['amount'] = left_over_txn_amount

            # Recompute rest of transaction
            split_txn_1['txn_dollars'] = - \
                split_txn_1['amount'] * split_txn_1['price']
            split_txn_2['txn_dollars'] = - \
                split_txn_2['amount'] * split_txn_2['price']

            # Delay 2nd trade by a second to avoid overlapping indices
            split_txn_2.index += pd.Timedelta(seconds=1)

            assert split_txn_1.amount.iloc[
                0] + split_txn_2.amount.iloc[0] == txn.iloc[-1].amount
            assert trans_sym.iloc[
                :sign_flip - 1].amount.sum() + split_txn_1.amount.iloc[0] == 0

            # Recreate transactions so far with split transaction
            trans_sym = pd.concat([trans_sym.iloc[
                                  :sign_flip - 1], split_txn_1, split_txn_2, trans_sym.iloc[sign_flip:]])

        assert np.all(np.abs(np.diff(np.sign(trans_sym.amount.cumsum()))) != 2)
        trans_split.append(trans_sym)

    transactions_split = pd.concat(trans_split)

    return transactions_split


def apply_sector_mappings_to_trades(trades, sector_mappings):
    sector_trades = trades.copy()
    sector_trades.symbol = sector_trades.symbol.apply(lambda x: sector_mappings.get(x, np.nan))
    sector_trades = sector_trades.dropna(axis=0)

    return sector_trades


def get_turnover(transactions, positions, period=None, average=True):
    """
    Portfolio Turnover Rate:

    Value of purchases and sales divided
    by the average portfolio value for the period.

    If no period is provided the period is one time step.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Contains transactions data.
        - See full explanation in tears.create_full_tear_sheet
    positions : pd.DataFrame
        Contains daily position values including cash
        - See full explanation in tears.create_full_tear_sheet
    period : str, optional
        Takes the same arguments as df.resample.
    average : bool
        if True, return the average of purchases and sales divided
        by portfolio value. If False, return the sum of
        purchases and sales divided by portfolio value.

    Returns
    -------
    turnover_rate : pd.Series
        timeseries of portfolio turnover rates.
    """

    traded_value = transactions.txn_volume
    portfolio_value = positions.sum(axis=1)
    if period is not None:
        traded_value = traded_value.resample(period, how='sum')
        portfolio_value = portfolio_value.resample(period, how='mean')
    # traded_value contains the summed value from buys and sells;
    # this is divided by 2.0 to get the average of the two.
    turnover = traded_value / 2.0 if average else traded_value
    turnover_rate = turnover / portfolio_value

    turnover_rate = turnover_rate.fillna(0)
    return turnover_rate
