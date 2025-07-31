#!/usr/bin/env python3
"""
Análise de diferentes estratégias de otimização para CS-SHRED
Considerando Error Norm e SSIM como métricas principais
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_trial_results(results_dir="./results/csshred/turb_v3_ssim_fast/"):
    """
    Analisa resultados de trials do Optuna para diferentes funções objetivo.
    """
    results_path = Path(results_dir)
    trial_files = list(results_path.glob("*_results.json"))
    
    trials_data = []
    for file_path in sorted(trial_files):
        with open(file_path, 'r') as f:
            data = json.load(f)
            trials_data.append(data)
    
    return trials_data

def compare_objective_functions(trials_data):
    """
    Compara diferentes funções objetivo baseadas nos resultados existentes.
    """
    results = []
    
    for trial in trials_data:
        error_norm = trial.get('error_norm', 0)
        
        # Compatibilidade com diferentes formatos de JSON
        ssim_mean = trial.get('ssim_score_mean', trial.get('ssim_score', 0))
        ssim_last = trial.get('ssim_score_last', trial.get('ssim_score', 0))
        
        # Se ainda é 0, pode ser que não tenha SSIM calculado
        if ssim_mean == 0 and 'ssim_score' not in trial:
            continue  # Pular trials sem dados de SSIM
        
        # Diferentes estratégias de função objetivo
        strategies = {
            'error_only': error_norm,
            'balanced_mean_ssim': 0.5 * error_norm + 0.5 * (1 - ssim_mean),
            'ssim_priority': 0.3 * error_norm + 0.7 * (1 - ssim_mean),
            'error_priority': 0.7 * error_norm + 0.3 * (1 - ssim_mean),
            'last_ssim_focus': 0.5 * error_norm + 0.5 * (1 - ssim_last),
            'multiplicative': error_norm * (2 - ssim_mean),
            'harmonic_mean': 2 / (1/max(error_norm, 1e-8) + 1/max(1-ssim_mean+1e-8, 1e-8)),
        }
        
        trial_result = {
            'trial_number': trial.get('trial_number', 0),
            'error_norm': error_norm,
            'ssim_mean': ssim_mean,
            'ssim_last': ssim_last,
            **strategies
        }
        results.append(trial_result)
    
    return results

def find_best_trials_by_strategy(results):
    """
    Encontra os melhores trials para cada estratégia de função objetivo.
    """
    strategies = ['error_only', 'balanced_mean_ssim', 'ssim_priority', 
                 'error_priority', 'last_ssim_focus', 'multiplicative', 'harmonic_mean']
    
    best_trials = {}
    
    for strategy in strategies:
        values = [r[strategy] for r in results]
        best_idx = np.argmin(values)
        best_trials[strategy] = {
            'trial_number': results[best_idx]['trial_number'],
            'objective_value': values[best_idx],
            'error_norm': results[best_idx]['error_norm'],
            'ssim_mean': results[best_idx]['ssim_mean'],
            'ssim_last': results[best_idx]['ssim_last']
        }
    
    return best_trials

def plot_pareto_frontier(results):
    """
    Plota a fronteira de Pareto entre Error Norm e SSIM.
    """
    errors = [r['error_norm'] for r in results]
    ssims = [r['ssim_mean'] for r in results]
    trial_nums = [r['trial_number'] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot principal
    scatter = plt.scatter(errors, ssims, c=trial_nums, cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Trial Number')
    
    # Destacar alguns pontos importantes
    best_error_idx = np.argmin(errors)
    best_ssim_idx = np.argmax(ssims)
    
    plt.scatter(errors[best_error_idx], ssims[best_error_idx], 
               color='red', s=100, marker='x', label=f'Best Error (Trial {trial_nums[best_error_idx]})')
    plt.scatter(errors[best_ssim_idx], ssims[best_ssim_idx], 
               color='blue', s=100, marker='x', label=f'Best SSIM (Trial {trial_nums[best_ssim_idx]})')
    
    plt.xlabel('Normalized Error')
    plt.ylabel('Mean SSIM')
    plt.title('Pareto Frontier: Error vs SSIM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adicionar anotações para alguns trials
    for i, (e, s, t) in enumerate(zip(errors, ssims, trial_nums)):
        if i % 3 == 0:  # Anotar a cada 3 trials para não sobrecarregar
            plt.annotate(f'T{t}', (e, s), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('./results/csshred/turb_v3_ssim_fast/pareto_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_recommendations():
    """
    Gera recomendações baseadas na análise.
    """
    recommendations = {
        "current_issue": [
            "SSIM valores muito baixos (0.5-0.7) indicam reconstrução estrutural pobre",
            "Optuna atual só minimiza error_norm, ignorando qualidade visual (SSIM)",
            "Necessário balancear ambas as métricas para melhor qualidade geral"
        ],
        
        "proposed_solutions": [
            "Função objetivo multicriterial: 0.5*error + 0.5*(1-ssim)",
            "Priorizar SSIM: 0.3*error + 0.7*(1-ssim) para melhor qualidade visual",
            "Usar SSIM do último snapshot como foco principal",
            "Implementar early stopping baseado em platô de ambas métricas"
        ],
        
        "hyperparameter_suggestions": [
            "Aumentar lambdaSNR para melhorar SNR e consequentemente SSIM",
            "Ajustar l1_tol e opt_tol para melhor convergência CS",
            "Experimentar diferentes arquiteturas (hidden_size, layers)",
            "Considerar dropout mais baixo para preservar detalhes"
        ],
        
        "expected_improvements": [
            "SSIM objetivo: > 0.85 (vs atual ~0.7)",
            "Manter error_norm < 0.3",
            "Melhor trade-off entre fidelidade numérica e qualidade visual"
        ]
    }
    
    return recommendations

def main():
    """
    Função principal de análise.
    """
    print("🔍 Analisando resultados de trials para otimização SSIM...")
    
    # Carregar dados dos trials
    trials_data = analyze_trial_results()
    print(f"Encontrados {len(trials_data)} trials para análise")
    
    # Comparar diferentes funções objetivo
    comparison_results = compare_objective_functions(trials_data)
    
    # Encontrar melhores trials por estratégia
    best_trials = find_best_trials_by_strategy(comparison_results)
    
    print("\n📊 MELHORES TRIALS POR ESTRATÉGIA:")
    print("="*60)
    for strategy, trial_info in best_trials.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Trial: {trial_info['trial_number']}")
        print(f"  Objective: {trial_info['objective_value']:.4f}")
        print(f"  Error: {trial_info['error_norm']:.4f}")
        print(f"  SSIM Mean: {trial_info['ssim_mean']:.4f}")
        print(f"  SSIM Last: {trial_info['ssim_last']:.4f}")
    
    # Plotar análise Pareto
    plot_pareto_frontier(comparison_results)
    
    # Gerar recomendações
    recommendations = generate_recommendations()
    
    print("\n💡 RECOMENDAÇÕES:")
    print("="*60)
    for category, items in recommendations.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for item in items:
            print(f"  • {item}")
    
    # Salvar análise detalhada
    analysis_results = {
        'best_trials_by_strategy': best_trials,
        'recommendations': recommendations,
        'total_trials_analyzed': len(trials_data)
    }
    
    with open('./results/csshred/turb_v3_ssim_fast/ssim_optimization_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    print(f"\n💾 Análise salva em: ./results/csshred/turb_v3_ssim_fast/ssim_optimization_analysis.json")

if __name__ == "__main__":
    main() 