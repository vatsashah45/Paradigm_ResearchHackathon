import json, sys

data = json.load(sys.stdin)
results = data['simulation_results']

# Analyze initial competitor spread vs spread_ticks
for r in sorted(results, key=lambda x: x['total_edge'])[:10]:
    regime = r['regime']
    print(f"edge={r['total_edge']:.3f} spread_ticks={regime['competitor_spread_ticks']} fills={r['fill_count']} init_p={regime['initial_probability']:.3f} retail_rate={regime['retail_arrival_rate']:.3f}")

print("---")
for r in sorted(results, key=lambda x: x['total_edge'])[-10:]:
    regime = r['regime']
    print(f"edge={r['total_edge']:.3f} spread_ticks={regime['competitor_spread_ticks']} fills={r['fill_count']} init_p={regime['initial_probability']:.3f} retail_rate={regime['retail_arrival_rate']:.3f}")

# Show spread_ticks vs edge
from collections import defaultdict
by_spread = defaultdict(list)
for r in results:
    by_spread[r['regime']['competitor_spread_ticks']].append(r['total_edge'])

print("\n--- Edge by competitor spread_ticks ---")
for s in sorted(by_spread.keys()):
    edges = by_spread[s]
    print(f"spread_ticks={s}: n={len(edges)} mean={sum(edges)/len(edges):.3f} min={min(edges):.3f} max={max(edges):.3f}")
