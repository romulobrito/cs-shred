#!/usr/bin/env python3
"""
Script de teste simplificado para verificar a implementacao da analise de complexidade
"""

import sys
import os
import time
import numpy as np
import psutil
import json
import matplotlib.pyplot as plt

# Adiciona o diretorio atual ao path para importar os modulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_complexity_analyzer_standalone():
    """Testa a classe ComplexityAnalyzer sem depender do oldroyd.py"""
    
    print("Testando ComplexityAnalyzer (standalone)...")
    
    try:
        # Importa apenas as classes necessarias
        from complexity_analysis import DetailedComplexityAnalyzer
        
        # Cria instancia
        analyzer = DetailedComplexityAnalyzer()
        
        # Testa analise teorica
        theoretical_analysis = analyzer.analyze_oldroyd_complexity()
        
        expected_algorithms = ['load_data', 'subsample', 'prepare_datasets', 'train_model', 'evaluate_model']
        for alg in expected_algorithms:
            assert alg in theoretical_analysis, f"Algoritmo {alg} nao encontrado"
        
        print("✓ Analise teorica funcionando")
        
        # Testa estimacao de complexidade
        data_sizes = [(50, 50, 50), (100, 100, 100)]
        times = [0.1, 0.8]  # Simula crescimento O(n)
        memory = [10, 80]   # Simula crescimento O(n)
        
        time_comp, mem_comp = analyzer.analyze_algorithm_complexity(
            "test_alg", data_sizes, times, memory
        )
        
        assert "O(" in time_comp, "Complexidade de tempo invalida"
        assert "O(" in mem_comp, "Complexidade de memoria invalida"
        print("✓ Estimacao de complexidade funcionando")
        
        # Testa geracao de relatorio
        os.makedirs("./test_results", exist_ok=True)
        report = analyzer.generate_complexity_report("./test_results")
        
        assert "theoretical_analysis" in report
        assert "summary" in report
        print("✓ Geracao de relatorio funcionando")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no ComplexityAnalyzer standalone: {e}")
        return False

def test_memory_functions_standalone():
    """Testa funcoes de memoria sem depender do oldroyd.py"""
    
    print("\nTestando funcoes de memoria (standalone)...")
    
    try:
        # Testa psutil diretamente
        process = psutil.Process()
        memory = process.memory_info().rss / 1024 / 1024
        assert memory > 0, "Uso de memoria deve ser positivo"
        print("✓ psutil funcionando")
        
        # Testa tracemalloc
        import tracemalloc
        tracemalloc.start()
        test_array = np.random.rand(1000, 1000)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        assert current > 0, "tracemalloc deve funcionar"
        print("✓ tracemalloc funcionando")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro nas funcoes de memoria standalone: {e}")
        return False

def test_dependencies():
    """Testa se todas as dependencias estao instaladas"""
    
    print("\nTestando dependencias...")
    
    dependencies = [
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('psutil', 'psutil'),
        ('torch', 'torch'),
        ('sklearn', 'sklearn'),
        ('memory_profiler', 'memory_profiler'),
        ('json', 'json'),
        ('time', 'time'),
        ('os', 'os')
    ]
    
    missing_deps = []
    
    for dep_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✓ {dep_name} disponivel")
        except ImportError:
            print(f"✗ {dep_name} nao encontrado")
            missing_deps.append(dep_name)
    
    if missing_deps:
        print(f"\nDependencias faltando: {missing_deps}")
        print("Instale com: pip install -r requirements_complexity.txt")
        return False
    
    return True

def test_complexity_analysis_module():
    """Testa o modulo complexity_analysis.py diretamente"""
    
    print("\nTestando modulo complexity_analysis...")
    
    try:
        from complexity_analysis import DetailedComplexityAnalyzer, generate_complexity_discussion
        
        # Testa geracao de discussao
        discussion = generate_complexity_discussion()
        assert len(discussion) > 0, "Discussao nao pode estar vazia"
        print("✓ Geracao de discussao funcionando")
        
        # Testa analisador
        analyzer = DetailedComplexityAnalyzer()
        analysis = analyzer.analyze_oldroyd_complexity()
        assert len(analysis) > 0, "Analise nao pode estar vazia"
        print("✓ Analise de complexidade funcionando")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no modulo complexity_analysis: {e}")
        return False

def test_run_complexity_analysis():
    """Testa o script run_complexity_analysis.py"""
    
    print("\nTestando script run_complexity_analysis...")
    
    try:
        # Importa funcoes do script
        from run_complexity_analysis import analyze_memory_usage_patterns, analyze_scalability
        
        # Testa analise de memoria
        analyze_memory_usage_patterns()
        print("✓ Analise de padroes de memoria funcionando")
        
        # Testa analise de escalabilidade
        analyze_scalability()
        print("✓ Analise de escalabilidade funcionando")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no script run_complexity_analysis: {e}")
        return False

def run_all_tests():
    """Executa todos os testes"""
    
    print("=" * 60)
    print("TESTE SIMPLIFICADO DA ANALISE DE COMPLEXIDADE")
    print("=" * 60)
    
    tests = [
        ("Dependencias", test_dependencies),
        ("ComplexityAnalyzer Standalone", test_complexity_analyzer_standalone),
        ("Funcoes de Memoria Standalone", test_memory_functions_standalone),
        ("Modulo complexity_analysis", test_complexity_analysis_module),
        ("Script run_complexity_analysis", test_run_complexity_analysis)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Erro inesperado: {e}")
            results.append((test_name, False))
    
    # Resumo dos resultados
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSOU" if result else "✗ FALHOU"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("A implementacao de analise de complexidade esta funcionando corretamente.")
        return True
    else:
        print(f"\n⚠️  {total - passed} teste(s) falharam.")
        print("Verifique os erros acima e corrija antes de usar.")
        return False

def cleanup_test_files():
    """Remove arquivos de teste criados"""
    
    test_files = [
        "./test_results/detailed_complexity_analysis.json",
        "./test_results"
    ]
    
    for file_path in test_files:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
        except:
            pass

if __name__ == "__main__":
    try:
        success = run_all_tests()
        
        # Limpa arquivos de teste
        cleanup_test_files()
        
        if success:
            print("\n✅ Implementacao pronta para uso!")
            print("\nPara usar:")
            print("1. python oldroyd.py  # Executa com monitoramento")
            print("2. python run_complexity_analysis.py  # Analise detalhada")
        else:
            print("\n❌ Corrija os erros antes de usar a implementacao.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nTeste interrompido pelo usuario.")
        cleanup_test_files()
        sys.exit(1)
    except Exception as e:
        print(f"\nErro inesperado: {e}")
        cleanup_test_files()
        sys.exit(1) 