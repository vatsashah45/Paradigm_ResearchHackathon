import json, sys

data = json.load(sys.stdin)
results = data['simulation_results']
edges = [r['total_edge'] for r in results]
retail = [r['retail_edge'] for r in results]
arb = [r['arb_edge'] for r in results]
fills = [r['fill_count'] for r in results]

print(f'N: {len(edges)}')
print(f'Mean edge: {sum(edges)/len(edges):.4f}')
print(f'Median edge: {sorted(edges)[len(edges)//2]:.4f}')
print(f'Min/Max edge: {min(edges):.4f} / {max(edges):.4f}')
print(f'Mean retail: {sum(retail)/len(retail):.4f}')
print(f'Mean arb: {sum(arb)/len(arb):.4f}')
print(f'Mean fills: {sum(fills)/len(fills):.1f}')
print(f'Positive edge: {sum(1 for e in edges if e > 0)}/{len(edges)}')
print(f'Zero fills: {sum(1 for f in fills if f == 0)}')
print()
for r in sorted(results, key=lambda x: x['total_edge'])[:5]:
    regime = r['regime']
    print(f"  Worst: edge={r['total_edge']:.3f} retail={r['retail_edge']:.3f} arb={r['arb_edge']:.3f} fills={r['fill_count']} spread={regime['competitor_spread_ticks']} inv={r['max_abs_inventory']:.1f} init_p={regime['initial_probability']:.2f}")
print()
for r in sorted(results, key=lambda x: x['total_edge'])[-5:]:
    regime = r['regime']
    print(f"  Best:  edge={r['total_edge']:.3f} retail={r['retail_edge']:.3f} arb={r['arb_edge']:.3f} fills={r['fill_count']} spread={regime['competitor_spread_ticks']} inv={r['max_abs_inventory']:.1f} init_p={regime['initial_probability']:.2f}")
