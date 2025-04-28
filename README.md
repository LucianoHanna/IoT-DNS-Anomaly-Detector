# IoT Botnet Detector - Docker Version

## Sobre o Projeto

**Título do Artigo:** Detecção de Botnets em Dispositivos IoT Utilizando Análise de Consultas DNS com One-class SVM

**Resumo:** A crescente proliferação de dispositivos IoT expandiu a superfície de ataque para botnets. Propomos uma abordagem para detecção em dispositivos IoT usando análise de consultas DNS com One-class SVM, que permite monitoramento em pontos de concentração da rede sem acesso direto aos dispositivos. Nossa metodologia usa 80% dos dados para treinamento com tráfego benigno e 20% para validação junto com domínios DGA. Os resultados demonstraram alta eficácia, com acurácia de 99,42%, precisão de 99,99%, recall de 99,42% e falsos positivos de 5,45%. O modelo identificou com 100% de recall 17 das 25 famílias de DGA testadas, demonstrando robustez frente a diferentes algoritmos de geração de domínios.

**Objetivos:** Este projeto implementa um sistema de detecção de botnets para dispositivos IoT, analisando consultas DNS e identificando domínios gerados por algoritmos (DGA) frequentemente utilizados para comunicação de comando e controle (C&C). A solução pode ser implementada em gateways ou servidores DNS, sem exigir intervenção nos dispositivos IoT.

## Requisitos

- Docker
- Docker Compose

Não é necessária nenhuma outra configuração! Todo o ambiente é configurado automaticamente pelo Docker.

## Como Usar

### Compilar a Imagem Docker

```bash
docker-compose build
```

### Treinar e avaliar um Modelo

```bash
docker-compose run botnet-detector all
```

### Treinar o Modelo com Seus Próprios Dados

```bash
docker-compose run botnet-detector train
```

### Avaliar o Modelo

```bash
docker-compose run botnet-detector evaluate
```

A saída desse comando irá exibir métricas resumidas, para visualização detalhada e com maior precisão, verifique o arquivo results/exp_N/results_summary_exp_N.txt

## Formatos de Arquivos de Entrada

### Arquivo de Log DNS (dns.log)

O formato esperado é de logs dnsmasq:

```
Month Day HH:MM:SS ... query[TYPE] domain-name ...
```

Exemplo:
```
Apr 11 14:25:31 gateway dnsmasq[123]: query[A] google.com from 192.168.1.2
```

### Arquivo de Domínios DGA (dga_domains.csv)

Formato CSV com cabeçalho:

```
domain,family,category
xjwpwvnhgi.com,malware1,dga
rlqxpuocqb.net,malware1,dga
zgvepuzyux.biz,malware2,dga
```

## Estrutura do Projeto

- `src/feature_extraction.py`: Implementa a extração de características de domínios
- `src/botnet_detector.py`: Contém o modelo One-class SVM e métodos de avaliação
- `src/main.py`: Script principal para processamento de comandos
- `src/config.py`: Parâmetros de configuração do sistema
- `data/`: Diretório para arquivos de entrada (DNS logs e domínios DGA)
- `models/`: Armazena modelos treinados
- `results/`: Armazena resultados e métricas de avaliação

## Notas Importantes

- Os modelos treinados são salvos no diretório `models/`.
- Os resultados da avaliação são salvos no diretório `results/`.
- Os IDs de experimento são atribuídos automaticamente de forma sequencial (exp_1, exp_2, etc.)
- O sistema usa parâmetros de configuração em `config.py` para extração de características e treinamento do modelo.

## Reprodução dos Experimentos

A tabela abaixo mapeia os experimentos descritos no artigo com as configurações disponíveis no repositório:

| Experimento no Artigo | Arquivo de Configuração | Características Habilitadas |
|----------------------|------------------------|----------------------------|
| Experimento 1 | `results/exp_1/config_exp_1.py` | Comprimento, proporção alfanumérica, entropia, n-gramas |
| Experimento 2 | `results/exp_2/config_exp_2.py` | Experimento 1 + vogais, dígitos, subdomínios |
| Experimento 3 | `results/exp_15/config_exp_15.py` | Proporção alfanumérica, n-gramas |

### Experimentos com Características Individuais

Os seguintes experimentos avaliam cada característica isoladamente:

| Característica | Arquivo de Configuração |
|---------------|------------------------|
| Comprimento do domínio | `results/exp_8/config_exp_8.py` |
| Proporção alfanumérica | `results/exp_9/config_exp_9.py` |
| Entropia | `results/exp_10/config_exp_10.py` |
| Características n-gramas | `results/exp_11/config_exp_11.py` |
| Proporção de vogais | `results/exp_12/config_exp_12.py` |
| Proporção de dígitos | `results/exp_13/config_exp_13.py` |
| Quantidade de subdomínios | `results/exp_14/config_exp_14.py` |

### Conjuntos de Dados
Este repositório inclui dois conjuntos de dados:

1. **Tráfego DNS Benigno**: Logs DNS coletados de dispositivos IoT em operação normal
2. **Domínios DGA**: Conjunto de 25 famílias de domínios gerados por algoritmos, para avaliação da detecção

## Dependências
O projeto utiliza Python 3.9 com as seguintes bibliotecas:
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0
- joblib==1.3.1
- pytest==8.3.5
- pytest-benchmark==5.1.0

Todas as dependências são instaladas automaticamente pelo Docker.

## Exemplos

### Exemplo 1: Treinar com Configuração Padrão
```bash
docker-compose run botnet-detector train
```

Saída esperada:
```
Using sequential experiment ID: exp_18
Saved configuration to /app/results/exp_18/config_exp_18.py
Processing DNS logs from /app/data/dns.log
Training data: 295 unique domains
Testing data (benign): 275 unique domains
Extracting features from 295 domains...
Training n-gram model on benign domains...
N-gram model trained: identified 75 common n-grams
N-gram model saved to /app/models/exp_18
Training One-class SVM model...
Model and scaler saved to /app/models/exp_18
Training completed in 0.02 seconds
```

### Exemplo 2: Avaliar com Domínios DGA
```bash
docker-compose run botnet-detector evaluate
```

Saída esperada (parcial):
```
Using sequential experiment ID: exp_18
Saved configuration to /app/results/exp_18/config_exp_18.py
Processing DNS logs from /app/data/dns.log
Training data: 295 unique domains
Testing data (benign): 275 unique domains
Loading DGA domains from /app/data/dga_domains.csv
Loaded 337500 malicious domains from DGA file
DGA family conficker: 13500 domains
[...]
DGA family vawtrak: 13500 domains
N-gram model loaded from /app/models/exp_18
Model and scaler loaded from /app/models/exp_18
Evaluation completed in 27.45 seconds


Results saved to /app/results/exp_18/results_summary_exp_18.txt
```

### Exemplo 3: Executar Benchmark
```bash
docker-compose run botnet-detector benchmark 1
```

Saída esperada:
```
Running benchmark for experiment: 18
Applying configuration from experiment 18...
Configuration applied successfully.
```

Será criado arquivo /results/exp_18/benchmark_results.json com resultados detalhados do benchmark, incluindo dados sobre o ambiente e, também, métrica sobre quantidade de domínios avaliados por segundo no caminho benchmark[N].extra_info.
```json
"extra_info": {
   "batch_size": 100,
   "mean_time_ms": 21.022614180972596,
   "mean_rps": 4756.782345865873,
   "min_rps": 4056.2471682921364,
   "max_rps": 4804.142093083238
}
```

## Configuração Personalizada

Você pode modificar os seguintes parâmetros em `config.py`:

- `TRAIN_TEST_SPLIT`: Porcentagem de dados usada para treinamento
- `NGRAM_SIZE`: Tamanho dos n-gramas para extração de características
- `NGRAM_TOP_K`: Número de n-gramas mais comuns a considerar como "normais"
- `FEATURE_CONFIG`: Habilitar/desabilitar características específicas
- `SVM_PARAMS`: Parâmetros do modelo One-class SVM