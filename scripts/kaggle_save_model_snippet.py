# ============================================================================
# KAGGLE MODEL SAVING FIX
# Add this cell at the END of your training notebook (after attack.attack())
# ============================================================================

import torch
import os

print("\n" + "="*80)
print("SAVING TRAINED MODEL FOR KAGGLE OUTPUT")
print("="*80)

# For Stateful Projan-2, use this filename:
model_filename = 'stateful_projan2_trained_model.pth'

# For Projan-2, use this filename instead:
# model_filename = 'projan2_trained_model.pth'

# Save model state dict to /kaggle/working
save_path = f'/kaggle/working/{model_filename}'
torch.save(model.state_dict(), save_path)

# Verify save
file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
print(f"\n✅ Model successfully saved!")
print(f"   Path: {save_path}")
print(f"   Size: {file_size_mb:.2f} MB")
print(f"   Expected: ~0.3-0.5 MB for MNIST Net model")

if file_size_mb > 1.0:
    print(f"\n⚠️  WARNING: File is larger than expected!")
    print(f"   This might include optimizer state or other data.")
    print(f"   Consider using model.state_dict() instead of full model.")

print("\n💡 Next Steps:")
print("   1. Run this notebook to completion")
print("   2. Go to 'Output' tab and click 'Save Version'")
print("   3. In defense notebook, add this output as a dataset")
print("   4. Use path: /kaggle/input/<output-name>/{}")
print("="*80 + "\n")
