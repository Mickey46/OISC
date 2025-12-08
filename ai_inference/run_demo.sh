#!/bin/bash
# Quick demo script for Neural Network Inference

echo "🧠 Neural Network Inference on OISC"
echo "===================================="
echo ""

cd "$(dirname "$0")"

echo "Running tests..."
python3 test_nn.py

echo ""
echo "To run manually:"
echo "  cd ai_inference"
echo "  python3 test_nn.py"
echo ""

