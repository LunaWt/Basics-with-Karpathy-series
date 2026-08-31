"""L05 diagnostics: how initialization and BatchNorm change a deep tanh stack.

Three conditions on an identical 8-layer 200x200 stack:
  bad   -- W ~ U(-1, 1), no scaling      (what bengio_mlp.py started with)
  init  -- W ~ randn / sqrt(fan_in)      (the fan-in fix)
  bn    -- fan-in init + BatchNorm       (statistics forced every forward)

Records, per layer: activation std, saturation fraction, and the std of the
gradient arriving at that layer's activations.

Run:  python bigram/l05_diagnostics.py
"""

import matplotlib.pyplot as plt
import torch

DEPTH, DIM, N = 8, 200, 4096
SEED = 42
CONDITIONS = ('bad', 'init', 'bn')
COLORS = {'bad': '#d62728', 'init': '#ff7f0e', 'bn': '#2ca02c'}
LABELS = {
    'bad': 'U(-1,1), no scaling',
    'init': 'randn / sqrt(fan_in)',
    'bn': 'fan-in init + BatchNorm',
}


def run(mode: str) -> dict[str, list[float]]:
    torch.manual_seed(SEED)
    x = torch.randn(N, DIM) * 1.095  # same input scale as C[X] in bengio_mlp.py
    x.requires_grad_(True)  # so the graph is built and gradients reach every layer
    acts = []

    for _ in range(DEPTH):
        if mode == 'bad':
            W = torch.empty(DIM, DIM).uniform_(-1.0, 1.0)
        else:
            W = torch.randn(DIM, DIM) / DIM**0.5

        hpreact = x @ W
        if mode == 'bn':
            mean = hpreact.mean(dim=0, keepdim=True)
            std = hpreact.std(dim=0, keepdim=True)
            hpreact = (hpreact - mean) / (std + 1e-5)  # gain=1, bias=0 at init

        x = torch.tanh(hpreact)
        x.retain_grad()
        acts.append(x)

    # any scalar will do -- we only care about the relative scale per layer
    acts[-1].pow(2).mean().backward()

    return {
        'std': [a.std().item() for a in acts],
        'sat': [(a.abs() > 0.99).float().mean().item() for a in acts],
        'gstd': [a.grad.std().item() for a in acts],
    }


results = {mode: run(mode) for mode in CONDITIONS}

for mode in CONDITIONS:
    r = results[mode]
    print(f'\n{LABELS[mode]}')
    print(f'  {"layer":>5} {"act std":>9} {"sat %":>8} {"grad std":>11}')
    for i in range(DEPTH):
        print(f'  {i + 1:>5} {r["std"][i]:>9.4f} {r["sat"][i] * 100:>7.2f}% {r["gstd"][i]:>11.3e}')

layers = range(1, DEPTH + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for mode in CONDITIONS:
    style = dict(color=COLORS[mode], label=LABELS[mode], marker='o', markersize=4)
    axes[0].plot(layers, results[mode]['std'], **style)
    axes[1].plot(layers, [s * 100 for s in results[mode]['sat']], **style)
    # absolute gradient scale is set by the arbitrary loss, so only the shape
    # across depth is meaningful -- normalise each curve to its own last layer
    ref = results[mode]['gstd'][-1]
    axes[2].plot(layers, [g / ref for g in results[mode]['gstd']], **style)

axes[0].set_title('Forward: activation std')
axes[0].set_ylabel('std of tanh output')
axes[0].axhline(1.0, color='gray', ls=':', lw=1)

axes[1].set_title('Forward: saturated units  (|h| > 0.99)')
axes[1].set_ylabel('% of units')

axes[2].set_title('Backward: gradient growth toward layer 1')
axes[2].set_ylabel('grad std, relative to last layer')
axes[2].set_yscale('log')
axes[2].axhline(1.0, color='gray', ls=':', lw=1)

for ax in axes:
    ax.set_xlabel('layer')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle(f'L05 diagnostics: {DEPTH} tanh layers, {DIM}x{DIM}, seed {SEED}', fontsize=11)
fig.tight_layout()
fig.savefig('bigram/l05_diagnostics.png', dpi=130)
print('\nsaved -> bigram/l05_diagnostics.png')
