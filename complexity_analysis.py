"""
Analise de Complexidade Computacional para oldroyd.py

Este modulo fornece funcoes para analisar a complexidade computacional
(tempo e memoria) dos algoritmos implementados no codigo oldroyd.py.
"""

import numpy as np
import time
import psutil
import json
import os
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

class DetailedComplexityAnalyzer:
    """
    Analisador detalhado de complexidade computacional
    """
    
    def __init__(self):
        self.analysis_results = {}
        self.benchmark_data = {}
        
    def analyze_algorithm_complexity(self, algorithm_name: str, data_sizes: List[Tuple], 
                                   time_measurements: List[float], memory_measurements: List[float]):
        """
        Analisa a complexidade de um algoritmo baseado em medidas empiricas
        
        Args:
            algorithm_name: Nome do algoritmo
            data_sizes: Lista de tuplas com tamanhos dos dados
            time_measurements: Medidas de tempo correspondentes
            memory_measurements: Medidas de memoria correspondentes
        """
        
        # Calcula complexidade de tempo
        time_complexity = self._estimate_time_complexity(data_sizes, time_measurements)
        
        # Calcula complexidade de memoria
        memory_complexity = self._estimate_memory_complexity(data_sizes, memory_measurements)
        
        # Armazena resultados
        self.analysis_results[algorithm_name] = {
            'time_complexity': time_complexity,
            'memory_complexity': memory_complexity,
            'data_sizes': data_sizes,
            'time_measurements': time_measurements,
            'memory_measurements': memory_measurements
        }
        
        return time_complexity, memory_complexity
    
    def _estimate_time_complexity(self, data_sizes: List[Tuple], times: List[float]) -> str:
        """
        Estima a complexidade de tempo baseada em medidas empiricas
        """
        if len(data_sizes) < 2:
            return "O(1)"
        
        # Calcula o produto dos tamanhos para cada entrada
        size_products = [np.prod(size) for size in data_sizes]
        
        # Calcula razoes de crescimento
        time_ratios = []
        size_ratios = []
        
        for i in range(1, len(times)):
            time_ratio = times[i] / times[i-1]
            size_ratio = size_products[i] / size_products[i-1]
            time_ratios.append(time_ratio)
            size_ratios.append(size_ratio)
        
        # Estima a complexidade baseada nas razoes
        avg_time_ratio = np.mean(time_ratios)
        avg_size_ratio = np.mean(size_ratios)
        
        if avg_time_ratio < 1.5:
            return "O(1)"
        elif avg_time_ratio < avg_size_ratio * 1.2:
            return "O(n)"
        elif avg_time_ratio < avg_size_ratio**2 * 1.2:
            return "O(n^2)"
        elif avg_time_ratio < avg_size_ratio**3 * 1.2:
            return "O(n^3)"
        else:
            return f"O(n^{np.log(avg_time_ratio)/np.log(avg_size_ratio):.1f})"
    
    def _estimate_memory_complexity(self, data_sizes: List[Tuple], memory: List[float]) -> str:
        """
        Estima a complexidade de memoria baseada em medidas empiricas
        """
        if len(data_sizes) < 2:
            return "O(1)"
        
        size_products = [np.prod(size) for size in data_sizes]
        memory_ratios = []
        size_ratios = []
        
        for i in range(1, len(memory)):
            memory_ratio = memory[i] / memory[i-1]
            size_ratio = size_products[i] / size_products[i-1]
            memory_ratios.append(memory_ratio)
            size_ratios.append(size_ratio)
        
        avg_memory_ratio = np.mean(memory_ratios)
        avg_size_ratio = np.mean(size_ratios)
        
        if avg_memory_ratio < 1.5:
            return "O(1)"
        elif avg_memory_ratio < avg_size_ratio * 1.2:
            return "O(n)"
        elif avg_memory_ratio < avg_size_ratio**2 * 1.2:
            return "O(n^2)"
        else:
            return f"O(n^{np.log(avg_memory_ratio)/np.log(avg_size_ratio):.1f})"
    
    def analyze_oldroyd_complexity(self):
        """
        Analise especifica da complexidade dos algoritmos em oldroyd.py
        """
        
        analysis = {
            'load_data': {
                'time_complexity': 'O(T*H*W)',
                'memory_complexity': 'O(T*H*W)',
                'description': 'Carregamento de dados do arquivo .npy'
            },
            'subsample': {
                'time_complexity': 'O(T*H*W)',
                'memory_complexity': 'O(T*H*W)',
                'description': 'Subamostragem aleatoria de colunas e snapshots'
            },
            'prepare_datasets': {
                'time_complexity': 'O(T*H*W + T*lags*num_sensors)',
                'memory_complexity': 'O(T*H*W + T*lags*num_sensors)',
                'description': 'Preparacao dos conjuntos de treino, validacao e teste'
            },
            'train_model': {
                'time_complexity': 'O(epochs * batches * (lags*num_sensors*hidden_size + hidden_size*l1 + l1*l2 + l2*output_size))',
                'memory_complexity': 'O(batch_size * (lags*num_sensors + hidden_size + l1 + l2 + output_size))',
                'description': 'Treinamento do modelo neural (SHRED ou CS-SHRED)'
            },
            'evaluate_model': {
                'time_complexity': 'O(test_samples * output_size)',
                'memory_complexity': 'O(test_samples * output_size)',
                'description': 'Avaliacao do modelo no conjunto de teste'
            }
        }
        
        return analysis
    
    def generate_complexity_report(self, save_path: str):
        """
        Gera um relatorio detalhado de complexidade computacional
        """
        
        # Analise teorica dos algoritmos
        theoretical_analysis = self.analyze_oldroyd_complexity()
        
        # Analise pratica (se houver dados de benchmark)
        practical_analysis = self.analysis_results
        
        report = {
            'theoretical_analysis': theoretical_analysis,
            'practical_analysis': practical_analysis,
            'summary': {
                'total_algorithms': len(theoretical_analysis),
                'most_complex_operation': max(theoretical_analysis.items(), 
                                            key=lambda x: self._complexity_score(x[1]['time_complexity'])),
                'memory_intensive_operations': [k for k, v in theoretical_analysis.items() 
                                              if 'O(n^2)' in v['memory_complexity'] or 'O(n^3)' in v['memory_complexity']]
            }
        }
        
        # Salva o relatorio
        report_path = os.path.join(save_path, 'detailed_complexity_analysis.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        return report
    
    def _complexity_score(self, complexity_str: str) -> float:
        """
        Converte string de complexidade em score numerico para comparacao
        """
        if 'O(1)' in complexity_str:
            return 1
        elif 'O(n)' in complexity_str:
            return 2
        elif 'O(n^2)' in complexity_str:
            return 3
        elif 'O(n^3)' in complexity_str:
            return 4
        else:
            # Extrai o expoente se existir
            import re
            match = re.search(r'O\(n\^(\d+\.?\d*)\)', complexity_str)
            if match:
                return float(match.group(1))
            return 5  # Complexidade desconhecida ou muito alta
    
    def plot_complexity_analysis(self, save_path: str):
        """
        Gera graficos de analise de complexidade
        """
        
        theoretical_analysis = self.analyze_oldroyd_complexity()
        
        # Grafico de complexidade de tempo
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Complexidade de tempo
        plt.subplot(2, 2, 1)
        algorithms = list(theoretical_analysis.keys())
        time_complexities = [theoretical_analysis[alg]['time_complexity'] for alg in algorithms]
        time_scores = [self._complexity_score(comp) for comp in time_complexities]
        
        plt.bar(algorithms, time_scores, color='skyblue')
        plt.title('Complexidade de Tempo dos Algoritmos')
        plt.ylabel('Score de Complexidade')
        plt.xticks(rotation=45)
        
        # Subplot 2: Complexidade de memoria
        plt.subplot(2, 2, 2)
        memory_complexities = [theoretical_analysis[alg]['memory_complexity'] for alg in algorithms]
        memory_scores = [self._complexity_score(comp) for comp in memory_complexities]
        
        plt.bar(algorithms, memory_scores, color='lightcoral')
        plt.title('Complexidade de Memoria dos Algoritmos')
        plt.ylabel('Score de Complexidade')
        plt.xticks(rotation=45)
        
        # Subplot 3: Comparacao tempo vs memoria
        plt.subplot(2, 2, 3)
        plt.scatter(time_scores, memory_scores, s=100, alpha=0.7)
        for i, alg in enumerate(algorithms):
            plt.annotate(alg, (time_scores[i], memory_scores[i]), 
                        xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Complexidade de Tempo')
        plt.ylabel('Complexidade de Memoria')
        plt.title('Tempo vs Memoria')
        
        # Subplot 4: Heatmap de complexidade
        plt.subplot(2, 2, 4)
        complexity_matrix = np.array([time_scores, memory_scores])
        sns.heatmap(complexity_matrix, 
                   xticklabels=algorithms,
                   yticklabels=['Tempo', 'Memoria'],
                   annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('Heatmap de Complexidade')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'complexity_analysis_plots.png'), dpi=300, bbox_inches='tight')
        plt.show()

def benchmark_oldroyd_operations():
    """
    Funcao para fazer benchmark das operacoes principais do oldroyd.py
    """
    
    # Tamanhos de dados para teste
    test_sizes = [
        (50, 50, 50),    # Pequeno
        (100, 100, 100), # Medio
        (200, 200, 200), # Grande
    ]
    
    analyzer = DetailedComplexityAnalyzer()
    
    for size in test_sizes:
        print(f"\nTestando com tamanho: {size}")
        
        # Simula operacoes (aqui voce pode integrar com o codigo real)
        # load_data_time = benchmark_load_data(size)
        # subsample_time = benchmark_subsample(size)
        # etc.
        
        print(f"Tamanho {size}: Tempo estimado baseado na complexidade teorica")

def generate_complexity_discussion():
    """
    Gera uma discussao detalhada sobre a complexidade computacional
    """
    
    discussion = """
    ================================================================================
    DISCUSSAO DE COMPLEXIDADE COMPUTACIONAL - oldroyd.py
    ================================================================================
    
    1. CARREGAMENTO DE DADOS (load_data)
       - Complexidade de tempo: O(T*H*W)
       - Complexidade de memoria: O(T*H*W)
       - Descricao: Carregamento de dados do arquivo .npy com transposicao
       - Otimizacoes possiveis: Carregamento lazy, compressao de dados
    
    2. SUBAMOSTRAGEM (subsample)
       - Complexidade de tempo: O(T*H*W)
       - Complexidade de memoria: O(T*H*W)
       - Descricao: Subamostragem aleatoria de colunas e snapshots
       - Otimizacoes possiveis: Subamostragem in-place, uso de sparse arrays
    
    3. PREPARACAO DE DADOS (prepare_datasets)
       - Complexidade de tempo: O(T*H*W + T*lags*num_sensors)
       - Complexidade de memoria: O(T*H*W + T*lags*num_sensors)
       - Descricao: Criacao dos conjuntos de treino, validacao e teste
       - Otimizacoes possiveis: Processamento em lotes, uso de generators
    
    4. TREINAMENTO DO MODELO (train_model)
       - Complexidade de tempo: O(epochs * batches * (lags*num_sensors*hidden_size + hidden_size*l1 + l1*l2 + l2*output_size))
       - Complexidade de memoria: O(batch_size * (lags*num_sensors + hidden_size + l1 + l2 + output_size))
       - Descricao: Treinamento do modelo neural (SHRED ou CS-SHRED)
       - Otimizacoes possiveis: Mixed precision, gradient checkpointing, distributed training
    
    5. AVALIACAO DO MODELO (evaluate_model)
       - Complexidade de tempo: O(test_samples * output_size)
       - Complexidade de memoria: O(test_samples * output_size)
       - Descricao: Avaliacao do modelo no conjunto de teste
       - Otimizacoes possiveis: Avaliacao em lotes, metricas incrementais
    
    ================================================================================
    OTIMIZACOES RECOMENDADAS:
    ================================================================================
    
    1. MEMORIA:
       - Usar torch.utils.data.DataLoader com num_workers para paralelizacao
       - Implementar gradient checkpointing para modelos grandes
       - Usar mixed precision training (torch.cuda.amp)
       - Limpar cache de GPU periodicamente
    
    2. TEMPO:
       - Paralelizar operacoes de dados com multiprocessing
       - Usar operacoes vetorizadas do NumPy/PyTorch
       - Implementar early stopping mais eficiente
       - Usar compilacao JIT do PyTorch (torch.jit)
    
    3. ESCALABILIDADE:
       - Implementar processamento distribuido
       - Usar tecnicas de compressao de dados
       - Implementar carregamento lazy de dados
       - Otimizar arquitetura do modelo para dados especificos
    
    ================================================================================
    """
    
    return discussion

if __name__ == "__main__":
    # Exemplo de uso
    analyzer = DetailedComplexityAnalyzer()
    
    # Gera analise teorica
    theoretical_analysis = analyzer.analyze_oldroyd_complexity()
    
    # Gera relatorio
    report = analyzer.generate_complexity_report("./results")
    
    # Gera graficos
    analyzer.plot_complexity_analysis("./results")
    
    # Imprime discussao
    print(generate_complexity_discussion()) 