# IoT Botnet Detector - Docker Version

This project implements a botnet detection system for IoT devices using DNS query analysis and One-class SVM classification, as described in the paper "Botnet Detection in IoT Devices Using DNS Query Analysis with One-class SVM".

## Requirements

- Docker
- Docker Compose

No other configuration is needed! The entire environment is automatically configured by Docker.

## How to Use

### Build the Docker Image

```bash
docker-compose build
```

### Train and evaluate a Model

```bash
docker-compose run botnet-detector all
```

### Train the Model with Your Own Data

1. Place your DNS log file in `./data/dns.log`
2. Run:

```bash
docker-compose run botnet-detector train
```

### Evaluate the Model

1. Make sure you have:
   - DNS logs in `./data/dns.log`
   - DGA domain list in `./data/dga_domains.csv`
2. Run:

```bash
docker-compose run botnet-detector evaluate
```

### Advanced Commands

To access a shell inside the container (for debugging):

```bash
docker-compose run botnet-detector shell
```

## Input File Formats

### DNS Log File (dns.log)

The expected format is dnsmasq logs:

```
Month Day HH:MM:SS ... query[TYPE] domain-name ...
```

Example:
```
Apr 11 14:25:31 gateway dnsmasq[123]: query[A] google.com from 192.168.1.2
```

### DGA Domains File (dga_domains.csv)

CSV format with header:

```
domain,family,category
xjwpwvnhgi.com,malware1,dga
rlqxpuocqb.net,malware1,dga
zgvepuzyux.biz,malware2,dga
```

## Important Notes

- Trained models are saved in the `models/` directory and persist between container runs.
- Evaluation results (metrics and graphs) are saved in the `results/` directory.
- Experiment IDs are automatically assigned sequentially (exp_1, exp_2, etc.)
- The system uses configuration parameters in `config.py` for feature extraction and model training.
- To use different files or paths, mount additional volumes as needed.

## Project Structure

- `feature_extraction.py`: Contains the domain feature extraction logic
- `botnet_detector.py`: Implements the One-class SVM model and evaluation methods
- `main.py`: Main script handling command-line arguments and workflow
- `config.py`: Configuration parameters for the system

## Configuration Options

You can modify the following parameters in `config.py`:

- `TRAIN_TEST_SPLIT`: Percentage of data used for training
- `NGRAM_SIZE`: Size of n-grams for feature extraction
- `NGRAM_TOP_K`: Number of most common n-grams to consider as "normal"
- `FEATURE_CONFIG`: Enable/disable specific features
- `SVM_PARAMS`: One-class SVM model parameters