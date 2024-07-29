import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

"""
functions used in the project: get_sinusoidal_embeddings, q_sample, load_and_preprocess_data, visualize_samples, save_samples_to_csv
"""

# Function to create sinusoidal time embeddings
def get_sinusoidal_embeddings(timesteps: torch.Tensor, embedding_dim: int, device: torch.device = 'cpu'):
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # output shape: (timesteps, embedding_dim)
    return emb

# Function to add noise to the data
def q_sample(x_start, t, noise, beta_t):
    beta_t_t = beta_t[t].view(-1, 1, 1).to(x_start.device)
    return torch.sqrt(1 - beta_t_t) * x_start + torch.sqrt(beta_t_t) * noise

# Function to load and preprocess data with normalization
def load_and_preprocess_data(file_path, batch_size=32):
    data = pd.read_csv(file_path, header=None, sep='\\s+')
    # Normalize the data to [0, 1] range
    data_min = data.min().min()
    data_max = data.max().max()
    data = (data - data_min) / (data_max - data_min)

    # Reshape data to match model input (N, 2, 100)
    data_tensor = torch.tensor(data.values, dtype=torch.float32).view(-1, 2, 100)
    assert data_tensor.shape[1] == 2 and data_tensor.shape[2] == 100, "Each data row must have 200 values (100 pedestrians, x and y coordinates)."
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, data_min, data_max

# Function to visualize generated samples
def visualize_samples(samples):
    for sample in samples:
        x_coords = sample[0, :].cpu()
        y_coords = sample[1, :].cpu()
        plt.scatter(x_coords, y_coords)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Generated Pedestrian Positions')
    plt.show()

# Function to save generated samples to a CSV file
def save_samples_to_csv(samples, output_file, data_min, data_max):
    samples = samples.cpu().numpy()  # Convert samples to NumPy array
    # Scale back to original range
    samples = samples * (data_max - data_min) + data_min
    samples = samples.reshape(samples.shape[0], -1)  # Flatten the samples
    df = pd.DataFrame(samples)
    df.to_csv(output_file, sep=' ', index=False, header=False)
    print(f"Generated samples saved to {output_file}")
# Test the functions, and visualize the samples, given a random input tensor, and a random time tensor.
if __name__ == "__main__":
    file_path = './dataset/pedestrians_positions_MI.csv'
    dataloader, data_min, data_max = load_and_preprocess_data(file_path, batch_size=32)
    t = torch.randint(0, 1000, (32,))
    print("t.shape:",t.shape)
    embedding_dims = 256
    t_emb = get_sinusoidal_embeddings(t, embedding_dims, device=torch.device('cpu'))
    print("t_emb.shape:",t_emb.shape) # torch.Size([32, 256])
    for batch in dataloader:
        # batch[0].shape = torch.Size([32, 2, 100]), because is not directly a tensor, but a tuple of tensors, there could be labels in the second element of the tuple.
        print("batch's shape:",batch[0].shape)
        visualize_samples(batch[0])
        break
