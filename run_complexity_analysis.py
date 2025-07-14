#!/usr/bin/env python3
"""
Script para executar analise de complexidade computacional do oldroyd.py

Este script demonstra como usar as funcoes de analise de complexidade
implementadas no codigo oldroyd.py e no modulo complexity_analysis.py
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from complexity_analysis import DetailedComplexityAnalyzer, generate_complexity_discussion

def run_complexity_analysis_example():
    """
    Executa um exemplo de analise de complexidade computacional
    """
    
    print("=" * 80)
    print("ANALISE DE COMPLEXIDADE COMPUTACIONAL - oldroyd.py")
    print("=" * 80)
    
    # Cria diretorio para resultados
    results_dir = r"/home/romulo/Documentos/lpips-env/results/shred/complexity_analysis"
    os.makedirs(results_dir, exist_ok=True)
    
    # Inicializa o analisador
    analyzer = DetailedComplexityAnalyzer()
    
    print("\n1. ANALISE TEORICA DOS ALGORITMOS")
    print("-" * 50)
    
    # Obtem analise teorica
    theoretical_analysis = analyzer.analyze_oldroyd_complexity()
    
    for algorithm, details in theoretical_analysis.items():
        print(f"\n{algorithm.upper()}:")
        print(f"  Complexidade de tempo: {details['time_complexity']}")
        print(f"  Complexidade de memoria: {details['memory_complexity']}")
        print(f"  Descricao: {details['description']}")
    
    print("\n2. SIMULACAO DE BENCHMARK")
    print("-" * 50)
    
    # Simula dados de benchmark para diferentes tamanhos
    data_sizes = [
        (50, 50, 50),     # Pequeno: 125K elementos
        (100, 100, 100),  # Medio: 1M elementos  
        (200, 200, 200),  # Grande: 8M elementos
    ]
    
    # Simula medidas de tempo (baseadas na complexidade teorica)
    simulated_times = {
        'load_data': [0.1, 0.8, 6.4],      # O(n)
        'subsample': [0.2, 1.6, 12.8],     # O(n)
        'prepare_datasets': [0.3, 2.4, 19.2], # O(n)
        'train_model': [1.0, 8.0, 64.0],   # O(n) por epoch
        'evaluate_model': [0.1, 0.8, 6.4]  # O(n)
    }
    
    # Simula medidas de memoria
    simulated_memory = {
        'load_data': [10, 80, 640],        # O(n) em MB
        'subsample': [10, 80, 640],        # O(n) em MB
        'prepare_datasets': [15, 120, 960], # O(n) em MB
        'train_model': [50, 400, 3200],    # O(n) em MB
        'evaluate_model': [5, 40, 320]     # O(n) em MB
    }
    
    # Analisa complexidade empirica
    for algorithm in simulated_times.keys():
        time_complexity, memory_complexity = analyzer.analyze_algorithm_complexity(
            algorithm,
            data_sizes,
            simulated_times[algorithm],
            simulated_memory[algorithm]
        )
        print(f"\n{algorithm}:")
        print(f"  Complexidade de tempo empirica: {time_complexity}")
        print(f"  Complexidade de memoria empirica: {memory_complexity}")
    
    print("\n3. GERACAO DE RELATORIOS")
    print("-" * 50)
    
    # Gera relatorio detalhado
    report = analyzer.generate_complexity_report(results_dir)
    print(f"Relatorio salvo em: {results_dir}/detailed_complexity_analysis.json")
    
    # Gera graficos
    analyzer.plot_complexity_analysis(results_dir)
    print(f"Graficos salvos em: {results_dir}/complexity_analysis_plots.png")
    
    print("\n4. RESUMO DA ANALISE")
    print("-" * 50)
    
    print(f"Total de algoritmos analisados: {report['summary']['total_algorithms']}")
    print(f"Operacao mais complexa: {report['summary']['most_complex_operation'][0]}")
    print(f"Operacoes intensivas em memoria: {report['summary']['memory_intensive_operations']}")
    
    print("\n5. OTIMIZACOES RECOMENDADAS")
    print("-" * 50)
    
    optimizations = {
        'load_data': [
            'Usar carregamento lazy para arquivos grandes',
            'Implementar compressao de dados',
            'Usar memoria mapeada (mmap) para arquivos muito grandes'
        ],
        'subsample': [
            'Implementar subamostragem in-place',
            'Usar arrays esparsos para dados subamostrados',
            'Paralelizar operacoes de subamostragem'
        ],
        'prepare_datasets': [
            'Usar DataLoader com num_workers para paralelizacao',
            'Implementar processamento em lotes',
            'Usar generators para economizar memoria'
        ],
        'train_model': [
            'Implementar mixed precision training',
            'Usar gradient checkpointing para modelos grandes',
            'Implementar distributed training',
            'Otimizar tamanho do batch'
        ],
        'evaluate_model': [
            'Avaliar em lotes para economizar memoria',
            'Implementar metricas incrementais',
            'Usar GPU para acelerar computacao'
        ]
    }
    
    for algorithm, opt_list in optimizations.items():
        print(f"\n{algorithm.upper()}:")
        for opt in opt_list:
            print(f"  - {opt}")
    
    print("\n6. DISCUSSAO DETALHADA")
    print("-" * 50)
    
    # Imprime discussao detalhada
    discussion = generate_complexity_discussion()
    print(discussion)
    
    # Salva discussao em arquivo
    with open(os.path.join(results_dir, 'complexity_discussion.txt'), 'w') as f:
        f.write(discussion)
    
    print(f"\nDiscussao salva em: {results_dir}/complexity_discussion.txt")
    
    print("\n" + "=" * 80)
    print("ANALISE DE COMPLEXIDADE CONCLUIDA!")
    print("=" * 80)

def analyze_memory_usage_patterns():
    """
    Analisa padroes de uso de memoria para diferentes tamanhos de dados
    """
    
    print("\nANALISE DE PADROES DE MEMORIA")
    print("-" * 50)
    
    # Tamanhos de dados para analise
    sizes = [(50, 50, 50), (100, 100, 100), (200, 200, 200)]
    
    print("Estimativa de uso de memoria para diferentes tamanhos:")
    print("Tamanho\t\tElementos\tMemoria (MB)\tMemoria (GB)")
    print("-" * 60)
    
    for size in sizes:
        elements = np.prod(size)
        memory_mb = elements * 8 / 1024 / 1024  # 8 bytes por elemento float64
        memory_gb = memory_mb / 1024
        
        print(f"{size}\t{elements:,}\t\t{memory_mb:.1f}\t\t{memory_gb:.3f}")
    
    print("\nRECOMENDACOES DE MEMORIA:")
    print("- Para dados pequenos (< 1GB): Usar CPU")
    print("- Para dados medios (1-8GB): Usar GPU com batch pequeno")
    print("- Para dados grandes (> 8GB): Usar processamento distribuido")

def analyze_scalability():
    """
    Analisa escalabilidade dos algoritmos
    """
    
    print("\nANALISE DE ESCALABILIDADE")
    print("-" * 50)
    
    # Simula crescimento de tempo para diferentes tamanhos
    base_size = 100
    sizes = [base_size, base_size*2, base_size*4, base_size*8]
    
    print("Crescimento de tempo para operacoes principais:")
    print("Tamanho\t\tload_data\tsubsample\ttrain_model")
    print("-" * 50)
    
    for size in sizes:
        elements = size**3
        load_time = elements / 1e6  # Simulacao O(n)
        subsample_time = elements / 1e6  # Simulacao O(n)
        train_time = elements / 1e5  # Simulacao O(n) por epoch
        
        print(f"{size}^3\t\t{load_time:.2f}s\t\t{subsample_time:.2f}s\t\t{train_time:.2f}s")
    
    print("\nOBSERVACOES DE ESCALABILIDADE:")
    print("- Operacoes de dados (load, subsample) escalam linearmente")
    print("- Treinamento e' o gargalo principal para dados grandes")
    print("- Memoria pode se tornar limitante antes do tempo")

if __name__ == "__main__":
    try:
        # Executa analise completa
        run_complexity_analysis_example()
        
        # Analises adicionais
        analyze_memory_usage_patterns()
        analyze_scalability()
        
        print("\n" + "=" * 80)
        print("TODAS AS ANALISES CONCLUIDAS COM SUCESSO!")
        print(f"Verifique os arquivos gerados em ./results/complexity_analysis/")
        print("=" * 80)
        
    except Exception as e:
        print(f"Erro durante a analise: {e}")
        sys.exit(1) 