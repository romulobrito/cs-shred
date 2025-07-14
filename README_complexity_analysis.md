# Analise de Complexidade Computacional - oldroyd.py

Este documento explica como implementar e usar a analise de complexidade computacional (tempo e memoria) para o codigo `oldroyd.py`.

## Visao Geral

A analise de complexidade computacional foi implementada para monitorar e analisar:
- **Tempo de execucao** de cada operacao principal
- **Uso de memoria** durante a execucao
- **Complexidade algoritmica** (Big O notation)
- **Escalabilidade** dos algoritmos
- **Gargalos de performance**

## Arquivos Implementados

### 1. `oldroyd.py` (Modificado)
- Adicionada classe `ComplexityAnalyzer` para monitoramento
- Decorators `@monitor_performance` nas funcoes principais
- Analise automatica de complexidade durante execucao
- Geracao de relatorios de performance

### 2. `complexity_analysis.py` (Novo)
- Analisador detalhado de complexidade computacional
- Funcoes para estimar complexidade empirica
- Geracao de graficos e relatorios
- Benchmark de algoritmos

### 3. `run_complexity_analysis.py` (Novo)
- Script de exemplo para executar analise completa
- Simulacao de benchmark com diferentes tamanhos de dados
- Geracao de relatorios e graficos

## Como Usar

### 1. Execucao Basica

```bash
# Executa o codigo original com monitoramento de complexidade
python oldroyd.py

# Executa analise detalhada de complexidade
python run_complexity_analysis.py
```

### 2. Monitoramento Manual

```python
from complexity_analysis import DetailedComplexityAnalyzer

# Inicializa analisador
analyzer = DetailedComplexityAnalyzer()

# Analisa complexidade teorica
theoretical_analysis = analyzer.analyze_oldroyd_complexity()

# Gera relatorio
report = analyzer.generate_complexity_report("./results")

# Gera graficos
analyzer.plot_complexity_analysis("./results")
```

### 3. Integracao com Codigo Existente

```python
from oldroyd import complexity_analyzer, monitor_performance

# Adiciona monitoramento a uma funcao
@monitor_performance
def minha_funcao():
    # seu codigo aqui
    pass

# Acessa resultados
print(complexity_analyzer.operation_times)
print(complexity_analyzer.memory_usage)
```

## Complexidade dos Algoritmos

### 1. Carregamento de Dados (`load_data`)
- **Tempo**: O(T×H×W)
- **Memoria**: O(T×H×W)
- **Descricao**: Carregamento de dados do arquivo .npy com transposicao

### 2. Subamostragem (`subsample`)
- **Tempo**: O(T×H×W)
- **Memoria**: O(T×H×W)
- **Descricao**: Subamostragem aleatoria de colunas e snapshots

### 3. Preparacao de Dados (`prepare_datasets`)
- **Tempo**: O(T×H×W + T×lags×num_sensors)
- **Memoria**: O(T×H×W + T×lags×num_sensors)
- **Descricao**: Criacao dos conjuntos de treino, validacao e teste

### 4. Treinamento do Modelo (`train_model`)
- **Tempo**: O(epochs × batches × (lags×num_sensors×hidden_size + hidden_size×l1 + l1×l2 + l2×output_size))
- **Memoria**: O(batch_size × (lags×num_sensors + hidden_size + l1 + l2 + output_size))
- **Descricao**: Treinamento do modelo neural (SHRED ou CS-SHRED)

### 5. Avaliacao do Modelo (`evaluate_model`)
- **Tempo**: O(test_samples × output_size)
- **Memoria**: O(test_samples × output_size)
- **Descricao**: Avaliacao do modelo no conjunto de teste

## Otimizacoes Recomendadas

### Memoria
- Usar `torch.utils.data.DataLoader` com `num_workers` para paralelizacao
- Implementar gradient checkpointing para modelos grandes
- Usar mixed precision training (`torch.cuda.amp`)
- Limpar cache de GPU periodicamente

### Tempo
- Paralelizar operacoes de dados com multiprocessing
- Usar operacoes vetorizadas do NumPy/PyTorch
- Implementar early stopping mais eficiente
- Usar compilacao JIT do PyTorch (`torch.jit`)

### Escalabilidade
- Implementar processamento distribuido
- Usar tecnicas de compressao de dados
- Implementar carregamento lazy de dados
- Otimizar arquitetura do modelo para dados especificos

## Arquivos de Saida

### 1. `complexity_analysis.json`
Relatorio detalhado com:
- Analise teorica dos algoritmos
- Medidas empiricas de tempo e memoria
- Resumo da complexidade geral

### 2. `complexity_analysis_plots.png`
Graficos de analise:
- Complexidade de tempo por algoritmo
- Complexidade de memoria por algoritmo
- Comparacao tempo vs memoria
- Heatmap de complexidade

### 3. `complexity_discussion.txt`
Discussao detalhada sobre:
- Analise de cada algoritmo
- Otimizacoes recomendadas
- Consideracoes de escalabilidade

## Exemplo de Saida

```
=== ANALISE DE COMPLEXIDADE: load_data ===
Forma dos dados: (600, 400, 400)
Total de elementos: 96,000,000
Complexidade de tempo: O(T*H*W)
Complexidade de memoria: O(96,000,000 * 8 bytes) = O(732.42 MB)
Memoria estimada: 732.42 MB
==================================================

Tempo de load_data: 2.3456 segundos
Memoria usada em load_data: 732.42 MB
Pico de memoria em load_data: 750.15 MB
```

## Dependencias

```bash
pip install psutil memory-profiler tracemalloc matplotlib seaborn
```

## Configuracao Avancada

### 1. Personalizar Monitoramento

```python
# Configurar nivel de detalhe
complexity_analyzer.verbose = True

# Adicionar metricas customizadas
complexity_analyzer.custom_metrics = ['gpu_memory', 'cpu_usage']
```

### 2. Benchmark com Diferentes Tamanhos

```python
# Testar com diferentes tamanhos de dados
test_sizes = [(50,50,50), (100,100,100), (200,200,200)]

for size in test_sizes:
    # Executar com dados de tamanho 'size'
    # Coletar metricas
    # Analisar escalabilidade
```

### 3. Comparacao de Modelos

```python
# Comparar SHRED vs CS-SHRED
models = ['SHRED', 'CS-SHRED']

for model in models:
    # Executar treinamento
    # Coletar metricas
    # Comparar performance
```

## Troubleshooting

### 1. Erro de Memoria
- Reduzir tamanho do batch
- Usar gradient checkpointing
- Implementar processamento em lotes

### 2. Tempo de Execucao Alto
- Verificar se GPU esta sendo usada
- Otimizar operacoes de dados
- Usar mixed precision training

### 3. Metricas Inconsistentes
- Verificar se tracemalloc esta funcionando
- Confirmar que psutil tem permissoes adequadas
- Verificar se nao ha vazamentos de memoria

## Contribuindo

Para adicionar novas metricas ou analises:

1. Adicione novas funcoes na classe `ComplexityAnalyzer`
2. Implemente decorators para monitoramento automatico
3. Atualize a documentacao
4. Adicione testes para validar as metricas

## Referencias

- [Big O Notation](https://en.wikipedia.org/wiki/Big_O_notation)
- [PyTorch Performance](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Memory Profiling in Python](https://pypi.org/project/memory-profiler/)
- [psutil Documentation](https://psutil.readthedocs.io/) 