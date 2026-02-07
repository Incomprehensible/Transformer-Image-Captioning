#!/usr/bin/env python3
"""Plot selected training metrics vs epoch on a single x-y plot.

Usage:
  python3 scripts/plot_metrics_over_epochs.py -i experiments/.../training_results.json -o results/plots_epochs -m train_loss test_loss test_acc

By default all metrics found in the JSON are plotted.
"""
import argparse
import json
import os
import sys
from itertools import cycle


def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    return {k: list(v) for k, v in data.items()}


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def _normalize_series(y, method):
    import math
    if method == 'none':
        return list(y)
    arr = list(y)
    if not arr:
        return arr
    if method == 'minmax':
        mn = min(arr)
        mx = max(arr)
        if mx == mn:
            return [0.0 for _ in arr]
        return [(v - mn) / (mx - mn) for v in arr]
    if method == 'zscore':
        mean = sum(arr) / len(arr)
        var = sum((v - mean) ** 2 for v in arr) / len(arr)
        std = math.sqrt(var)
        if std == 0:
            return [0.0 for _ in arr]
        return [(v - mean) / std for v in arr]
    if method == 'max':
        mx = max(abs(v) for v in arr) or 1.0
        return [v / mx for v in arr]
    return list(arr)


def plot_metrics(metrics, chosen, outdir, filename, linewidth=2.0, normalize='none',
                 mark_best=False, mark_style='*', mark_size=120, annotate_mark=False,
                 theme='default'):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib required. Install it and re-run.")
        return 2

    try:
        import seaborn as sns
        use_seaborn = True
    except Exception:
        sns = None
        use_seaborn = False

    if theme == 'dark':
        try:
            plt.style.use('dark_background')
        except Exception:
            pass
        if use_seaborn and hasattr(sns, 'set_theme'):
            sns.set_theme(context='notebook', style='darkgrid', palette='muted')
        plt.rcParams.update({'figure.facecolor': '#222222', 'axes.facecolor': '#222222',
                             'axes.edgecolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white',
                             'text.color': 'white', 'axes.labelcolor': 'white', 'grid.color': '#444444'})
    elif theme == 'minimal':
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass
        if use_seaborn and hasattr(sns, 'set_theme'):
            sns.set_theme(context='notebook', style='white', palette='muted')
    else:
        if use_seaborn and hasattr(sns, 'set_theme'):
            sns.set_theme(context='notebook', style='whitegrid', palette='muted')
        else:
            try:
                plt.style.use('ggplot')
            except Exception:
                pass

    keys = sorted(metrics.keys())
    if chosen:
        sel = [k for k in chosen if k in metrics]
        missing = [k for k in chosen if k not in metrics]
        if missing:
            print(f"Warning: these metrics not found and will be skipped: {missing}")
    else:
        sel = keys

    if not sel:
        print("No metrics to plot.")
        return 1

    lengths = [len(metrics[k]) for k in sel]
    min_len = min(lengths)
    if any(l != min_len for l in lengths):
        print(f"Warning: metric series have different lengths; truncating to {min_len} epochs.")

    epochs = list(range(1, min_len + 1))

    plt.figure(figsize=(11, 6))
    prop = plt.rcParams.get('axes.prop_cycle')
    if prop is not None and hasattr(prop, 'by_key'):
        colors = prop.by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
    else:
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    color_cycle = cycle(colors)
    style_cycle = cycle(['-', '--', '-.', ':'])

    for key in sel:
        orig_y = metrics[key][:min_len]
        y = list(orig_y)
        if normalize and normalize != 'none':
            y = _normalize_series(y, normalize)
        c = next(color_cycle)
        ls = next(style_cycle)
        # place markers sparsely to avoid clutter
        markevery = max(1, int(min_len / 10))
        plt.plot(epochs, y, label=key, color=c, linestyle=ls, linewidth=linewidth,
             marker='o', markersize=6, markevery=markevery, alpha=0.95)

        if mark_best:
            name = key.lower()
            # decide whether higher is better
            higher_is_better = any(token in name for token in ('bleu', 'meteor', 'acc', 'accuracy', 'f1'))
            lower_is_better = any(token in name for token in ('loss', 'perplex', 'perp'))
            # default: higher is better for acc/score metrics, lower better for loss/perplexity
            if lower_is_better:
                best_idx = int(min(range(len(orig_y)), key=lambda i: orig_y[i]))
            elif higher_is_better:
                best_idx = int(max(range(len(orig_y)), key=lambda i: orig_y[i]))
            else:
                # fallback: mark max
                best_idx = int(max(range(len(orig_y)), key=lambda i: orig_y[i]))

            # value for plotting (use normalized if applied so marker sits on the drawn curve)
            plot_val = y[best_idx] if (normalize and normalize != 'none') else orig_y[best_idx]
            ex = epochs[best_idx]
            plt.scatter([ex], [plot_val], marker=mark_style, s=mark_size, color=c, edgecolor='k', zorder=5)
            if annotate_mark:
                val_label = orig_y[best_idx]
                txt = f"{ex}: {val_label:.4g}"
                plt.annotate(txt, xy=(ex, plot_val), xytext=(5, 5), textcoords='offset points', fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    plt.xlabel('Epoch', fontsize=12)
    plt.xticks(epochs)
    plt.grid(alpha=0.25)
    leg = plt.legend(loc='best', frameon=True)
    if leg:
        leg.get_frame().set_alpha(0.9)
    plt.tight_layout()

    outpath = os.path.join(outdir, filename)
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved combined plot to {outpath}")
    return 0


def parse_metrics_arg(arg_list):
    if not arg_list:
        return []
    parts = []
    for a in arg_list:
        parts.extend([p.strip() for p in a.split(',') if p.strip()])
    return parts


def main():
    p = argparse.ArgumentParser(description='Plot metrics vs epoch on one plot')
    p.add_argument('-i', '--input', required=True, help='Path to training_results.json')
    p.add_argument('-o', '--outdir', default='plots', help='Output directory')
    p.add_argument('-m', '--metrics', nargs='*', help='Metrics to plot (space or comma separated). Default: all')
    p.add_argument('-n', '--name', default='metrics_over_epochs.png', help='Output filename')
    p.add_argument('--normalize', default='none', choices=['none', 'minmax', 'zscore', 'max'],
                   help='Normalization method for metric values before plotting')
    p.add_argument('--mark-best', action='store_true', help='Mark per-metric best epoch (min for losses, max for scores)')
    p.add_argument('--mark-style', default='*', help='Matplotlib marker style for best points')
    p.add_argument('--mark-size', type=int, default=120, help='Marker size for best points')
    p.add_argument('--annotate-mark', action='store_true', help='Annotate marked best points with epoch and value')
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(2)

    metrics = load_results(args.input)
    chosen = parse_metrics_arg(args.metrics)
    ensure_outdir(args.outdir)
    p.add_argument('--theme', default='default', choices=['default', 'dark', 'minimal'],
                   help='Plot theme preset')

    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(2)

    metrics = load_results(args.input)
    chosen = parse_metrics_arg(args.metrics)
    ensure_outdir(args.outdir)
    rc = plot_metrics(metrics, chosen, args.outdir, args.name,
                      normalize=args.normalize,
                      mark_best=args.mark_best,
                      mark_style=args.mark_style,
                      mark_size=args.mark_size,
                      annotate_mark=args.annotate_mark,
                      theme=args.theme)
    if rc == 0:
        print('Done.')
    sys.exit(rc)


if __name__ == '__main__':
    main()
