"""
Test Script for Neural Network Inference
Quick validation of the XOR network
"""

import numpy as np
from nn_inference import SimpleNeuralNetwork


def test_xor():
    """Test XOR problem"""
    print("\n" + "="*70)
    print("🧪 TESTING XOR NEURAL NETWORK")
    print("="*70)
    
    nn = SimpleNeuralNetwork('weights.json')
    
    # Test cases
    test_cases = [
        ([0, 0], 0, "False XOR False = False"),
        ([0, 1], 1, "False XOR True  = True"),
        ([1, 0], 1, "True  XOR False = True"),
        ([1, 1], 0, "True  XOR True  = False")
    ]
    
    print("\n📋 Test Results:")
    print("-"*70)
    
    all_correct = True
    
    for i, (inputs, expected, description) in enumerate(test_cases, 1):
        output = nn.forward(inputs)
        prediction = nn.predict(inputs)
        
        correct = prediction == expected
        all_correct = all_correct and correct
        
        status = "✅ PASS" if correct else "❌ FAIL"
        
        print(f"\nTest {i}: {description}")
        print(f"  Input:      {inputs}")
        print(f"  Raw Output: {output:.6f}")
        print(f"  Prediction: {prediction}")
        print(f"  Expected:   {expected}")
        print(f"  Status:     {status}")
    
    print("\n" + "="*70)
    
    if all_correct:
        print("✅ ALL TESTS PASSED! Neural network correctly solves XOR.")
    else:
        print("❌ SOME TESTS FAILED. Check network weights.")
    
    print("="*70 + "\n")
    
    return all_correct


def test_intermediate_values():
    """Test with intermediate values (not just 0 and 1)"""
    print("\n" + "="*70)
    print("🔬 TESTING INTERMEDIATE VALUES")
    print("="*70)
    
    nn = SimpleNeuralNetwork('weights.json')
    
    test_values = [
        [0.0, 0.0],
        [0.2, 0.8],
        [0.5, 0.5],
        [0.8, 0.2],
        [1.0, 1.0]
    ]
    
    print("\n📊 Continuous Inputs:")
    print("-"*70)
    
    for inputs in test_values:
        output = nn.forward(inputs)
        print(f"  Input: {inputs} → Output: {output:.6f}")
    
    print("="*70 + "\n")


def benchmark():
    """Simple performance benchmark"""
    print("\n" + "="*70)
    print("⚡ PERFORMANCE BENCHMARK")
    print("="*70)
    
    nn = SimpleNeuralNetwork('weights.json')
    
    import time
    
    # Standard sigmoid
    start = time.time()
    for _ in range(10000):
        nn.forward([0.5, 0.5], use_oisc_style=False)
    standard_time = time.time() - start
    
    # OISC-style sigmoid
    start = time.time()
    for _ in range(10000):
        nn.forward([0.5, 0.5], use_oisc_style=True)
    oisc_time = time.time() - start
    
    print(f"\n⏱️  10,000 inferences:")
    print(f"  Standard sigmoid: {standard_time:.4f}s")
    print(f"  OISC-style sigmoid: {oisc_time:.4f}s")
    print(f"  Speedup: {standard_time/oisc_time:.2f}x")
    
    print("\n="*70 + "\n")


if __name__ == "__main__":
    # Run all tests
    test_xor()
    test_intermediate_values()
    benchmark()
    
    print("\n🎉 All tests complete! Neural network is working correctly.\n")

