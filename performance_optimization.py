import torch
from torch.utils.data import DataLoader
import time

# Function to measure inference time
def evaluate_inference_speed(mod, data_loader):
    start_time = time.time()
    mod.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations for inference
        for batch in data_loader:
            _ = mod(batch['ocr_feature'], batch['cnn_feature'])
    total_time = time.time() - start_time
    return total_time

# Load the saved model from model_training
model = ProductClassifier(vocab_len, embedding_size, lstm_hidden_size, cnn_feat_size, output_classes)
model.load_state_dict(torch.load('trained_model.pth'))

# Create a sample dataloader (Ensure sample data is filled accordingly)
sample_data = ProductDataset(...)  # You need to fill this with appropriate sample data
sample_loader = DataLoader(sample_data, batch_size=32)

# Measure the inference time of the original model
orig_time_taken = evaluate_inference_speed(model, sample_loader)
print(f"Original Model Inference Time: {orig_time_taken:.4f} seconds")

# Optimize the model using TorchScript
optimized_model = torch.jit.script(model)

# Measure the inference time of the optimized model
opt_time_taken = evaluate_inference_speed(optimized_model, sample_loader)
print(f"Optimized Model Inference Time: {opt_time_taken:.4f} seconds")

# Save the optimized model for future use
torch.jit.save(optimized_model, 'optimized_model.pth')

print("Optimization complete. Optimized model saved as 'optimized_model.pth'.")
