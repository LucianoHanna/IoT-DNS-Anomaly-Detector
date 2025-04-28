import pytest
import os
import shutil
import pandas as pd
from botnet_detector import BotnetDetector
from dns_log_parser import DNSLogParser

def setup_config_for_benchmark(exp_id):
    """
    Copia o arquivo de configuração do experimento especificado para o arquivo config.py atual
    """
    exp_config_path = os.path.join("/app/results", f"exp_{exp_id}", f"config_exp_{exp_id}.py")
    current_config_path = "/app/config.py"
    
    if not os.path.exists(exp_config_path):
        raise FileNotFoundError(f"Arquivo de configuração para o experimento {exp_id} não encontrado: {exp_config_path}")
    
    if os.path.exists(current_config_path):
        backup_path = current_config_path + ".backup"
        shutil.copy2(current_config_path, backup_path)

    shutil.copy2(exp_config_path, current_config_path)
    
    print(f"Configuração do experimento {exp_id} aplicada com sucesso.")

@pytest.fixture
def detector(request):
    """Fixture para carregar um detector pré-treinado"""
    # Obter o número do experimento da linha de comando
    exp_id = request.config.getoption("--exp-id")
    
    setup_config_for_benchmark(exp_id)
    model_dir = os.path.join("/app/models", f"exp_{exp_id}")
    detector = BotnetDetector(model_dir=model_dir)
    
    # Carregar o modelo pré-treinado
    if not detector.load_model():
        pytest.skip(f"Falha ao carregar o modelo do experimento {exp_id}.")
    
    print(f"\n[INFO] Usando modelo do experimento: {exp_id}")
    
    return detector

@pytest.fixture
def benign_domains():
    """Fixture para carregar domínios benignos do DNS log"""
    dns_log_path = "/app/data/dns.log"
    
    if not os.path.exists(dns_log_path):
        pytest.skip("Arquivo DNS log não encontrado em /app/data/dns.log")
    
    parser = DNSLogParser()
    dns_data = parser.parse_dnsmasq_log(dns_log_path)
    
    # Remover duplicatas
    unique_domains = dns_data['domain'].unique().tolist()
    
    return unique_domains

@pytest.fixture
def malicious_domains():
    """Fixture para carregar domínios maliciosos do arquivo DGA"""
    dga_file_path = "/app/data/dga_domains.csv"
    
    if not os.path.exists(dga_file_path):
        pytest.skip("Arquivo DGA não encontrado em /app/data/dga_domains.csv")
    
    dga_data = pd.read_csv(dga_file_path)
    
    if 'domain' not in dga_data.columns:
        pytest.skip("Formato de arquivo DGA inválido. Coluna 'domain' não encontrada.")
    
    # Remover duplicatas
    unique_domains = dga_data['domain'].unique().tolist()
    
    return unique_domains

# Função para calcular e formatar o RPS
def calculate_rps(benchmark_stats, batch_size):
    mean_time = benchmark_stats['mean']
    min_time = benchmark_stats['min']
    max_time = benchmark_stats['max']
    
    mean_rps = batch_size / mean_time
    min_rps = batch_size / max_time
    max_rps = batch_size / min_time
    
    return {
        'batch_size': batch_size,
        'mean_time_ms': mean_time * 1000,
        'mean_rps': mean_rps,
        'min_rps': min_rps,
        'max_rps': max_rps
    }


# Testes de benchmark para o método predict com cálculo de RPS

def test_predict_single_domain(benchmark, detector, benign_domains):
    """Benchmark para predição de um único domínio"""
    domain = benign_domains[0]
    
    # Executar o benchmark
    benchmark.pedantic(detector.predict, args=(domain,), iterations=1, rounds=1000)
    
    # Calcular RPS para um único domínio
    rps_stats = calculate_rps(benchmark.stats, 1)
    
    # Adicionar dados ao relatório
    benchmark.extra_info.update(rps_stats)


def test_predict_batch_small(benchmark, detector, benign_domains):
    """Benchmark para predição de um lote pequeno de domínios (10)"""
    if len(benign_domains) < 10:
        pytest.skip("Não há domínios suficientes para este teste")
    
    domains = benign_domains[:10]
    
    # Executar o benchmark
    benchmark.pedantic(detector.predict, args=(domains,), iterations=1, rounds=1000)
    
    # Calcular RPS para o lote de 10 domínios
    rps_stats = calculate_rps(benchmark.stats, 10)
    
    # Adicionar dados ao relatório
    benchmark.extra_info.update(rps_stats)


def test_predict_batch_medium(benchmark, detector, benign_domains):
    """Benchmark para predição de um lote médio de domínios (100)"""
    if len(benign_domains) < 100:
        pytest.skip("Não há domínios suficientes para este teste")
    
    domains = benign_domains[:100]
    
    # Executar o benchmark
    benchmark.pedantic(detector.predict, args=(domains,), iterations=1, rounds=1000)
    
    # Calcular RPS para o lote de 100 domínios
    rps_stats = calculate_rps(benchmark.stats, 100)
    
    # Adicionar dados ao relatório
    benchmark.extra_info.update(rps_stats)

def test_predict_malicious_domains(benchmark, detector, malicious_domains):
    """Benchmark usando domínios maliciosos"""
    if not malicious_domains:
        pytest.skip("Não há domínios maliciosos disponíveis")
    
    # Limitar a 100 domínios para manter o teste eficiente
    domains = malicious_domains[:100] if len(malicious_domains) > 100 else malicious_domains
    batch_size = len(domains)
    
    # Executar o benchmark
    benchmark.pedantic(detector.predict, args=(domains,), iterations=1, rounds=1000)
    
    # Calcular RPS para o lote de domínios maliciosos
    rps_stats = calculate_rps(benchmark.stats, batch_size)
    
    # Adicionar dados ao relatório
    benchmark.extra_info.update(rps_stats)
