import torch
from utils import visualize_samples, save_samples_to_csv, load_and_preprocess_data
from models.unet import UNet1D
"""
Function to generate new samples using the model, and save them to a CSV file.
The model is loaded from the specified checkpoint, and new samples are generated using the model.
The generated samples are then saved to a CSV file for further analysis.
"""

# Function to generate new samples using the model
def generate_samples(model, num_samples, T, device, data_min, data_max):
    model.eval()
    with torch.no_grad():
        # Start from Gaussian noise
        x = torch.randn(num_samples, 2, 100, device=device)  # Ensure the shape matches the training data
        for t in reversed(range(T)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            #print("t_tensor's shape", t_tensor.shape)
            x = model(x, t_tensor)

            # Rescale the samples back to original range
            x = (x - x.min().min()) / (x.max().max() - x.min().min())
        #x = x * (data_max - data_min) + data_min
        return x


# Example usage, assuming the model is already trained, and the data is loaded, normalized, preprocessed.
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 1000
    sampled_epochs = 200 # Epochs to sample from

    # Load the trained model
    weight_path = './CheckPoints/ddpm_unet1d_weights_epoch_' + str(sampled_epochs) + '.pth'
    model = UNet1D(in_channels=2, out_channels=2).to(device)  # Ensure the model's input and output channels match
    model.load_state_dict(torch.load(weight_path))
    model.eval()  # Set the model to evaluation mode

    # Load min and max values used during training
    dataloader, data_min, data_max = load_and_preprocess_data('./dataset/pedestrians_positions_MI_converted.csv',
                                                              batch_size=32)

    # Generate new samples
    num_samples = 100  # Number of samples to generate
    generated_samples = generate_samples(model, num_samples, T, device, data_min, data_max)

    # Visualize the generated samples
    visualize_samples(generated_samples)

    # Save the generated samples to a CSV file
    output_file = './Generated_Position_Data/generated_pedestrians_positions_epoch' + str(sampled_epochs) + '.csv'
    save_samples_to_csv(generated_samples, output_file, data_min, data_max)
