#!/usr/bin/env python3
"""Run strategy and output results."""
import sys
import os
sys.path.insert(0, '/Users/vatsashah/prediction-market-challenge')
from orderbook_pm_challenge.cli import main

strategy = sys.argv[1] if len(sys.argv) > 1 else '/Users/vatsashah/prediction-market-strategy/strategy_v12.py'
n = sys.argv[2] if len(sys.argv) > 2 else '100'
args = ['run', strategy, '--simulations', n]
if '--json' in sys.argv:
    args.append('--json')
main(args)
