#!/bin/bash

# Choose action based on the first argument
case "$1" in
  train)
    # Train the model with DNS data
    shift
    python main.py --dns_log /app/data/dns.log --train "$@"
    ;;
  
  evaluate)
    # Evaluate the model with DNS and DGA data
    shift
    python main.py --dns_log /app/data/dns.log --dga_file /app/data/dga_domains.csv --evaluate "$@"
    ;;

  all)
    # Train and evaluate
    shift
    python main.py --dns_log /app/data/dns.log --dga_file /app/data/dga_domains.csv --train --evaluate "$@"
    ;;
  
  benchmark)
    # Run performance benchmarks
    shift

    exp_id="$1"
    shift
    echo "Running benchmark for experiment: $exp_id"
    python -m pytest /app/tests/test_benchmark_detector_rps.py -v --benchmark-json="/app/results/exp_${exp_id}/benchmark_results.json" --exp-id="$exp_id"
    
    ;;
  
  shell)
    # Provide a shell for debugging or manual execution
    exec /bin/bash
    ;;
  
  *)
    # Pass all arguments to main.py script
    python main.py "$@"
    ;;
esac