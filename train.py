import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import load_and_preprocess_data, q_sample, visualize_samples
from models.unet import UNet1D
"""
Training function for the DDPM model with time embedding.
The model is trained using the specified dataloader for a specified number of epochs.
The model is saved at regular intervals, and the loss curve is plotted at the end of training.
"""
# Training function with time embedding
def train_ddpm(model, dataloader, num_epochs=200, save_interval=5, sample_interval=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()

        # using tqdm to show the progress bar
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for batch in dataloader:
                x_start = batch[0].to(device)  # Shape: (batch_size, 1, 200)
                t = torch.randint(0, T, (x_start.size(0),), device=x_start.device).long()
                noise = torch.randn_like(x_start, device=device)
                x_noisy = q_sample(x_start, t, noise, beta_t)
                optimizer.zero_grad()
                x_recon = model(x_noisy, t)
                loss = criterion(x_recon, x_start)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                # Update progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss}")

        # Visualize generated samples at intervals
        if (epoch + 1) % sample_interval == 0:
            model.eval()
            with torch.no_grad():
                x = torch.randn(1, 2, 100, device=device)
                for t_ in reversed(range(T)):
                    t_tensor = torch.full((1,), t_, device=device, dtype=torch.long)
                    x = model(x, t_tensor)
                    x = x * (data_max - data_min) + data_min
                visualize_samples(x)

        # Save model weights at intervals
        if (epoch + 1) % save_interval == 0:
            weight_path = f'./CheckPoints/ddpm_unet1d_weights_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), weight_path)
            print(f"Model weights saved to '{weight_path}'")

    # Plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

    # Save final model weights
    final_weight_path = './CheckPoints/ddpm_unet1d_weights_final.pth'
    torch.save(model.state_dict(), final_weight_path)
    print(f"Final model weights saved to '{final_weight_path}'")

# Training the model, and saving the final weights, and the loss curve, at the end of training, and visualizing the generated samples at regular intervals during training.
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    T = 1000
    beta_t = torch.linspace(0.0001, 0.02, T, device=device)

    # Load the dataset
    file_path = './dataset/pedestrians_positions_MI_converted.csv'
    dataloader, data_min, data_max = load_and_preprocess_data(file_path, batch_size=32)

    # Instantiate the model and move it to the device
    model = UNet1D().to(device)

    # Train the model
    train_ddpm(model, dataloader)
