"""
Generate training loss visualization from SFT run
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Training losses from modal run
training_data = [
    {"step": 1, "loss": 3.1566, "lr": 0.0, "epoch": 0.18},
    {"step": 2, "loss": 3.323, "lr": 0.00006, "epoch": 0.35},
    {"step": 3, "loss": 3.1044, "lr": 0.00012, "epoch": 0.53},
    {"step": 4, "loss": 2.0761, "lr": 0.00018, "epoch": 0.70},
    {"step": 5, "loss": 1.2604, "lr": 0.00024, "epoch": 0.88},
    {"step": 6, "loss": 1.1167, "lr": 0.0003, "epoch": 1.0},
]

steps = [d['step'] for d in training_data]
losses = [d['loss'] for d in training_data]
lrs = [d['lr'] for d in training_data]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Loss curve
ax1.plot(steps, losses, 'b-o', linewidth=2, markersize=8, label='Training Loss')
ax1.set_xlabel('Training Step', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('PsychoHistory SFT Training Loss\n(1 Epoch, 91 Cases, LoRA Rank 64)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Add loss values as text
for i, (step, loss) in enumerate(zip(steps, losses)):
    ax1.text(step, loss + 0.1, f'{loss:.3f}', ha='center', va='bottom', fontsize=9)

# Add improvement annotation
improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
ax1.text(0.98, 0.95, f'Improvement: {improvement:.1f}%\n(3.157 → 1.117)',
         transform=ax1.transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
         fontsize=10)

# Plot 2: Learning rate schedule
ax2.plot(steps, lrs, 'r-o', linewidth=2, markersize=8, label='Learning Rate')
ax2.set_xlabel('Training Step', fontsize=12)
ax2.set_ylabel('Learning Rate', fontsize=12)
ax2.set_title('Learning Rate Schedule (Linear Warmup)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()

# Save figure
output_path = 'training/results/sft_training_loss.png'
import os
os.makedirs('training/results', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"✅ Training loss graph saved to: {output_path}")
print(f"\n📊 Summary Statistics:")
print(f"   Initial loss: {losses[0]:.3f}")
print(f"   Final loss: {losses[-1]:.3f}")
print(f"   Improvement: {improvement:.1f}%")
print(f"   Training steps: {len(steps)}")
print(f"   Final LR: {lrs[-1]:.6f}")
