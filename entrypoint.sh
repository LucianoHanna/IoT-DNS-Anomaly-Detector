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
  
  shell)
    # Provide a shell for debugging or manual execution
    exec /bin/bash
    ;;
  
  *)
    # Pass all arguments to main.py script
    python main.py "$@"
    ;;
esac