import sqlite3, datetime, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
db = sqlite3.connect('data/trades.db')
c = db.cursor()
cutoff = 1774840000
c.execute('SELECT id, timestamp, slug, side, entry_price, size_usd, delta, edge, time_remaining_sec, oracle_age_sec, outcome, pnl FROM trades WHERE timestamp >= ? ORDER BY id', (cutoff,))
rows = c.fetchall()
total_pnl = 0; wins = 0; losses = 0
by_asset = {}; by_tf = {}

for r in rows:
    tid, ts, slug, side, entry, size, delta, edge, t_rem, oracle_age, outcome, pnl = r
    dt = datetime.datetime.fromtimestamp(ts)
    parts = slug.split('-')
    asset = parts[0].upper()
    tf = parts[2] if len(parts) > 2 else '?'
    total_pnl += pnl
    if outcome == 'won':
        wins += 1
    elif outcome == 'lost':
        losses += 1

    key_a = asset
    if key_a not in by_asset:
        by_asset[key_a] = {'w': 0, 'l': 0, 'pnl': 0}
    by_asset[key_a]['pnl'] += pnl
    if outcome == 'won':
        by_asset[key_a]['w'] += 1
    elif outcome == 'lost':
        by_asset[key_a]['l'] += 1

    key_t = f'{asset} {tf}'
    if key_t not in by_tf:
        by_tf[key_t] = {'w': 0, 'l': 0, 'pnl': 0}
    by_tf[key_t]['pnl'] += pnl
    if outcome == 'won':
        by_tf[key_t]['w'] += 1
    elif outcome == 'lost':
        by_tf[key_t]['l'] += 1

    flags = []
    if entry < 0.35:
        flags.append('EXTREME_LOW')
    if entry > 0.82:
        flags.append('EXTREME_HIGH')
    if abs(delta) < 0.002:
        flags.append('LOW_DELTA')
    if edge < 0.03:
        flags.append('LOW_EDGE')
    if oracle_age > 40:
        flags.append('STALE_CL')
    flag_str = ' [' + ','.join(flags) + ']' if flags else ''
    mark = ' X' if outcome == 'lost' else '  '
    print(f'#{tid:2d} {dt.strftime("%H:%M")} {asset:3s} {tf:3s} {side:3s} @{entry:.2f} ${size:7.1f} d={delta:+.5f} edge={edge:.1%} t={t_rem:5.0f}s cl={oracle_age:4.1f}s {outcome:4s} {pnl:+8.1f}{mark}{flag_str}')

print(f'\n===== BILAN =====')
n = wins + losses
print(f'Trades: {n} | W:{wins} L:{losses} | WR: {wins/n*100:.0f}%')
print(f'PnL total: ${total_pnl:+.2f}')
avg_win = sum(r[11] for r in rows if r[10] == 'won') / max(wins, 1)
avg_loss = sum(r[11] for r in rows if r[10] == 'lost') / max(losses, 1)
print(f'Avg win: ${avg_win:+.1f} | Avg loss: ${avg_loss:+.1f}')

print(f'\nPar asset:')
for k, v in sorted(by_asset.items()):
    t = v['w'] + v['l']
    wr = v['w'] / t * 100 if t > 0 else 0
    print(f'  {k}: {v["w"]}W/{v["l"]}L ({wr:.0f}%) PnL=${v["pnl"]:+.1f}')

print(f'\nPar asset+TF:')
for k, v in sorted(by_tf.items()):
    t = v['w'] + v['l']
    wr = v['w'] / t * 100 if t > 0 else 0
    print(f'  {k}: {v["w"]}W/{v["l"]}L ({wr:.0f}%) PnL=${v["pnl"]:+.1f}')

# Anomaly detection
print(f'\n===== ANOMALIES =====')
for r in rows:
    tid, ts, slug, side, entry, size, delta, edge, t_rem, oracle_age, outcome, pnl = r
    parts = slug.split('-')
    asset = parts[0].upper()
    tf = parts[2] if len(parts) > 2 else '?'
    issues = []
    if entry < 0.35:
        issues.append(f'entry @{entry:.2f} tres bas (<0.35) = longshot')
    if entry > 0.82:
        issues.append(f'entry @{entry:.2f} tres haut (>0.82) = paye cher pour peu de gain')
    if abs(delta) < 0.0015:
        issues.append(f'delta={delta:+.5f} trop faible')
    if edge < 0.025:
        issues.append(f'edge={edge:.1%} tres faible')
    if oracle_age > 40:
        issues.append(f'oracle age={oracle_age:.0f}s stale')
    if t_rem < 50:
        issues.append(f't_rem={t_rem:.0f}s tres court')
    if issues:
        dt = datetime.datetime.fromtimestamp(ts)
        print(f'  #{tid} {dt.strftime("%H:%M")} {asset} {tf} {outcome}: {"; ".join(issues)}')

db.close()
