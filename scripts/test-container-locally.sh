#!/bin/bash
# Script to test the benchmark container locally

set -e

echo "=== MerLin Benchmark Container Test ==="
echo

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    echo "Please install Docker to run containerized benchmarks"
    exit 1
fi

# Build the container
echo "🔨 Building benchmark container..."
docker build -f Dockerfile.benchmark -t merlin-benchmark:test .

# Create results directory
mkdir -p benchmark-results

# Run quick benchmark test
echo "🚀 Running quick benchmark test..."
docker run --rm \
    -v "$(pwd)/benchmark-results:/app/results" \
    -w /app \
    merlin-benchmark:test \
    pytest tests/benchmark_slos_core.py --benchmark-json=/app/results/test-benchmark-results.json -v --benchmark-only -k "config0 and cpu"

# Check if results were generated
if [ -f "benchmark-results/test-benchmark-results.json" ]; then
    echo "✅ Benchmark container test successful!"
    echo "📊 Results saved to benchmark-results/test-benchmark-results.json"
    
    # Show summary
    echo
    echo "📈 Benchmark Summary:"
    python3 -c "
import json
try:
    with open('benchmark-results/test-benchmark-results.json', 'r') as f:
        data = json.load(f)
    benchmarks = data.get('benchmarks', [])
    print(f'Total benchmarks run: {len(benchmarks)}')
    for bench in benchmarks[:3]:  # Show first 3
        name = bench['name'].split('::')[-1] if '::' in bench['name'] else bench['name']
        stats = bench['stats']
        print(f'  {name}: {stats[\"mean\"]:.2f}μs ± {stats[\"stddev\"]:.2f}μs')
except Exception as e:
    print(f'Could not parse results: {e}')
"
else
    echo "❌ Benchmark container test failed - no results generated"
    exit 1
fi

echo
echo "🧪 Running correctness tests..."
docker run --rm \
    merlin-benchmark:test \
    pytest tests/test_slos_correctness.py -v -x

echo "✅ All tests passed!"
echo
echo "💡 To run the full benchmark suite:"
echo "   docker-compose -f docker-compose.benchmark.yml up benchmark"
echo
echo "💡 To run memory benchmarks:"
echo "   docker-compose -f docker-compose.benchmark.yml up memory"