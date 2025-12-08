# Neural Network Inference on OISC

A simple 2-layer perceptron running on OISC architecture that solves the XOR problem.

## Architecture
- Input Layer: 2 neurons
- Hidden Layer: 2 neurons (sigmoid activation)
- Output Layer: 1 neuron (sigmoid activation)

## Pre-trained Weights
The network has been pre-trained to solve XOR:
- Input: [0,0] → Output: 0
- Input: [0,1] → Output: 1
- Input: [1,0] → Output: 1
- Input: [1,1] → Output: 0

## Files
- `nn_inference.py` - Main neural network implementation
- `weights.json` - Pre-trained weights
- `test_nn.py` - Test script with all XOR cases
- `oisc_nn.ai` - OISC/Forth-style implementation (future)

## Quick Start
```bash
python test_nn.py
```

## How It Works
1. Forward pass computes: `output = sigmoid(W2 @ sigmoid(W1 @ input + b1) + b2)`
2. All operations use basic OISC primitives (add, multiply, compare)
3. Sigmoid approximation uses lookup table for efficiency

## Future Extensions
- [ ] Implement sigmoid in pure OISC code
- [ ] Add more activation functions (ReLU, Tanh)
- [ ] Multi-class classification (softmax)
- [ ] Larger networks (deeper/wider)

