"""
Generate comprehensive training and eval loss visualization
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Complete training data from H100 run (3 epochs, 70 train / 21 val)
training_steps = [
    {"step": 1, "train_loss": 4.5251, "eval_loss": None, "epoch": 0.23, "lr": 0.0},
    {"step": 2, "train_loss": 4.3927, "eval_loss": 4.2866, "epoch": 0.46, "lr": 0.00006},
    {"step": 3, "train_loss": 4.2259, "eval_loss": None, "epoch": 0.69, "lr": 0.00012},
    {"step": 4, "train_loss": 3.3187, "eval_loss": 2.1578, "epoch": 0.91, "lr": 0.00018},
    {"step": 5, "train_loss": 2.1223, "eval_loss": None, "epoch": 1.0, "lr": 0.00024},
    {"step": 6, "train_loss": 1.5322, "eval_loss": 1.4366, "epoch": 1.23, "lr": 0.0003},
    {"step": 7, "train_loss": 1.405, "eval_loss": None, "epoch": 1.46, "lr": 0.00027},
    {"step": 8, "train_loss": 1.4137, "eval_loss": 1.3629, "epoch": 1.69, "lr": 0.00024},
    {"step": 9, "train_loss": 1.3695, "eval_loss": None, "epoch": 1.91, "lr": 0.00021},
    {"step": 10, "train_loss": 1.3031, "eval_loss": 1.3103, "epoch": 2.0, "lr": 0.00018},
    {"step": 11, "train_loss": 1.2597, "eval_loss": None, "epoch": 2.23, "lr": 0.00015},
    {"step": 12, "train_loss": 1.2548, "eval_loss": 1.2775, "epoch": 2.46, "lr": 0.00012},
    {"step": 13, "train_loss": 1.3091, "eval_loss": None, "epoch": 2.69, "lr": 0.00009},
    {"step": 14, "train_loss": 1.2686, "eval_loss": 1.2587, "epoch": 2.91, "lr": 0.00006},
    {"step": 15, "train_loss": 1.262, "eval_loss": None, "epoch": 3.0, "lr": 0.00003},
]

steps = [d['step'] for d in training_steps]
train_losses = [d['train_loss'] for d in training_steps]
eval_steps = [d['step'] for d in training_steps if d['eval_loss'] is not None]
eval_losses = [d['eval_loss'] for d in training_steps if d['eval_loss'] is not None]

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# Plot both train and eval loss
ax.plot(steps, train_losses, 'b-o', linewidth=2.5, markersize=8, label='Train Loss', alpha=0.8)
ax.plot(eval_steps, eval_losses, 'r-s', linewidth=2.5, markersize=10, label='Eval Loss', alpha=0.8)

ax.set_xlabel('Training Step', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax.set_title('PsychoHistory SFT Training: Train vs Eval Loss\n' +
             '(3 Epochs, 70 Train / 21 Val, LoRA Rank 64, H100 GPU)',
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='upper right')

# Add improvement annotations
train_improvement = ((train_losses[0] - train_losses[-1]) / train_losses[0]) * 100
eval_improvement = ((eval_losses[0] - eval_losses[-1]) / eval_losses[0]) * 100

ax.text(0.02, 0.98,
        f'Train Loss:\n  Initial: {train_losses[0]:.3f}\n  Final: {train_losses[-1]:.3f}\n  Improvement: {train_improvement:.1f}%',
        transform=ax.transAxes, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        fontsize=11)

ax.text(0.02, 0.65,
        f'Eval Loss:\n  Initial: {eval_losses[0]:.3f}\n  Final: {eval_losses[-1]:.3f}\n  Improvement: {eval_improvement:.1f}%',
        transform=ax.transAxes, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
        fontsize=11)

# Add final metrics box
ax.text(0.98, 0.02,
        f'Final Metrics:\n  Train: {train_losses[-1]:.3f}\n  Eval: {eval_losses[-1]:.3f}\n  Gap: {abs(train_losses[-1] - eval_losses[-1]):.3f}',
        transform=ax.transAxes, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
        fontsize=11, fontweight='bold')

plt.tight_layout()

# Save figure
output_path = 'training/results/sft_full_training_curves.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"✅ Full training curves saved to: {output_path}")
print(f"\n📊 Final Results:")
print(f"   Train Loss: {train_losses[0]:.3f} → {train_losses[-1]:.3f} ({train_improvement:.1f}% improvement)")
print(f"   Eval Loss:  {eval_losses[0]:.3f} → {eval_losses[-1]:.3f} ({eval_improvement:.1f}% improvement)")
print(f"   Train/Eval Gap: {abs(train_losses[-1] - eval_losses[-1]):.3f} (minimal overfitting!)")
print(f"   Total Steps: {len(steps)}")
print(f"   Training Time: 9min 36sec")
