"""
Neural Network Inference Engine
Simple 2-layer perceptron for XOR problem
Demonstrates AI running on minimalist OISC-inspired architecture
"""

import numpy as np
import json


def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_oisc_style(x):
    """
    OISC-style sigmoid approximation using basic operations
    This version uses piece-wise linear approximation
    """
    # Clamp input to reasonable range
    if x < -5:
        return 0.0
    elif x > 5:
        return 1.0
    else:
        # Simple approximation: sigmoid(x) ≈ 0.5 + 0.2*x for x in [-2.5, 2.5]
        return 0.5 + 0.2 * x


class SimpleNeuralNetwork:
    """
    Minimal Neural Network for XOR
    Architecture: 2 -> 2 -> 1
    """
    
    def __init__(self, weights_file='weights.json'):
        """Load pre-trained weights"""
        with open(weights_file, 'r') as f:
            data = json.load(f)
        
        self.W1 = np.array(data['weights']['W1'])
        self.b1 = np.array(data['weights']['b1'])
        self.W2 = np.array(data['weights']['W2'])
        self.b2 = np.array(data['weights']['b2'])
        
        print(f"✓ Loaded {data['description']}")
        print(f"  Architecture: {data['architecture']}")
    
    def forward(self, x, use_oisc_style=False):
        """
        Forward pass through the network
        
        Args:
            x: Input vector [x1, x2]
            use_oisc_style: If True, use OISC-style sigmoid approximation
        
        Returns:
            Output value (0 to 1)
        """
        x = np.array(x)
        
        # Layer 1: Input -> Hidden
        z1 = np.dot(self.W1, x) + self.b1
        
        # Activation
        if use_oisc_style:
            a1 = np.array([sigmoid_oisc_style(z) for z in z1])
        else:
            a1 = sigmoid(z1)
        
        # Layer 2: Hidden -> Output
        z2 = np.dot(self.W2, a1) + self.b2
        
        # Output activation
        if use_oisc_style:
            a2 = sigmoid_oisc_style(z2[0])
        else:
            a2 = sigmoid(z2)
        
        return float(a2)
    
    def predict(self, x, threshold=0.5):
        """Binary prediction"""
        output = self.forward(x)
        return 1 if output >= threshold else 0
    
    def show_computation_steps(self, x):
        """Show detailed computation steps (for educational purposes)"""
        print(f"\n{'='*60}")
        print(f"Forward Pass for Input: {x}")
        print(f"{'='*60}")
        
        x = np.array(x)
        
        # Layer 1
        print(f"\n1️⃣  Hidden Layer Computation:")
        print(f"   z1 = W1 @ x + b1")
        z1 = np.dot(self.W1, x) + self.b1
        print(f"   z1 = {z1}")
        
        a1 = sigmoid(z1)
        print(f"   a1 = sigmoid(z1) = {a1}")
        
        # Layer 2
        print(f"\n2️⃣  Output Layer Computation:")
        print(f"   z2 = W2 @ a1 + b2")
        z2 = np.dot(self.W2, a1) + self.b2
        print(f"   z2 = {z2}")
        
        a2 = sigmoid(z2)
        print(f"   output = sigmoid(z2) = {a2[0]:.6f}")
        
        prediction = 1 if a2[0] >= 0.5 else 0
        print(f"\n   Prediction: {prediction}")
        print(f"{'='*60}\n")
        
        return a2[0]


def main():
    """Demo the neural network"""
    print("\n" + "="*60)
    print("🧠 Neural Network Inference Engine on OISC")
    print("="*60)
    
    # Load network
    nn = SimpleNeuralNetwork('weights.json')
    
    # XOR test cases
    test_cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0)
    ]
    
    print("\n📊 Testing XOR Problem:")
    print("-" * 60)
    
    for inputs, expected in test_cases:
        output = nn.forward(inputs)
        prediction = nn.predict(inputs)
        status = "✓" if prediction == expected else "✗"
        
        print(f"{status} Input: {inputs} → Output: {output:.6f} → Prediction: {prediction} (Expected: {expected})")
    
    # Detailed computation for one example
    nn.show_computation_steps([1, 0])
    
    print("\n" + "="*60)
    print("✅ Neural Network Inference Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

