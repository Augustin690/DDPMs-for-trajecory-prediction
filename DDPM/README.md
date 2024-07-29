# DDPM Project

This project is based on the DDPM (Denoising Diffusion Probabilistic Models) for generating new samples. The generated samples are saved in the `sampledimgs` directory.

## Project Structure

- `Diffusion/train.py`: Contains the script for training the DDPM model.
- `Diffusion/Diffusion.py`: Contains the implementation of the diffusion process.
- `Diffusion/Model.py`: Contains the implementation of the DDPM model.
- `main.py`: Contains the script for generating new samples using the trained model.

## Usage

1. Train the DDPM model and save the weights to a `.pth` file, which is saved in the `CheckPoints` directory, and will be used for generating new samples.
2. Run `sampling.py` to generate new samples using the trained model. The generated samples are saved to the 'sampledimgs' directory.

## Results

The generated samples are saved in the `sampledimgs` directory. You can view the results by opening the images in this directory.

Please note that the quality of the generated samples depends on the number of training epochs and the quality of the training dataset.