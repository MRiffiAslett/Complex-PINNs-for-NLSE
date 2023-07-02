# Complex Physics-Informed Neural Networks for Solving the Nonlinear Schrödinger Equation

This repository contains an implementation of a complex-valued Physics-Informed Neural Network (PINN) used to solve the Nonlinear Schrödinger Equation (NLSE).

## Problem Statement

The NLSE is a partial differential equation that describes the evolution of complex wave fields in various contexts, including optics and quantum mechanics. It is expressed as:

i ∂ψ/∂t + ∂²ψ/∂x² + 2|ψ|²ψ = 0

In this project, we train a PINN to learn a known solution of this equation and the underlying physics described by it.

## Methodology

A complex-valued PINN is defined and implemented using TensorFlow. The network predicts the real and imaginary parts of the wave function ψ, which are then used to compute the residual of the NLSE. The model is trained to minimize this residual, effectively learning to solve the NLSE.

## File Structure

- `Complex_PINN_Solver.py`: The main Python script for defining and training the complex-valued PINN.

## Requirements

To install the necessary Python packages for this project, run:

pip install -r requirements.txt


## Usage

After installing the necessary packages, you can run the main Python script as follows:

python Complex_PINN_Solver.py

## Output

The script outputs the loss value during training and the predictions of the trained model on test data. These predictions can be compared with the known solution to assess the accuracy of the model.
