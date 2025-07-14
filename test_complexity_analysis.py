#!/usr/bin/env python3
"""
Script de teste para verificar a implementacao da analise de complexidade
"""

import sys
import os
import time
import numpy as np
import psutil
import json

# Adiciona o diretorio atual ao path para importar os modulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_complexity_analyzer():
    """Testa a classe ComplexityAnalyzer"""
    
    print("Testando ComplexityAnalyzer...")
    
    try:
        from oldroyd import ComplexityAnalyzer
        
        # Cria instancia
        analyzer = ComplexityAnalyzer()
        
        # Testa timer
        analyzer.start_timer()
        time.sleep(0.1)  # Simula trabalho
        elapsed = analyzer.end_timer("test_operation")
        
        assert elapsed > 0, "Timer nao funcionou"
        print("✓ Timer funcionando")
        
        # Testa monitoramento de memoria
        analyzer.start_memory_tracking()
        # Aloca memoria
        test_array = np.random.rand(1000, 1000)
        memory_used = analyzer.end_memory_tracking("test_memory")
        
        assert memory_used >= 0, "Monitoramento de memoria nao funcionou"
        print("✓ Monitoramento de memoria funcionando")
        
        # Testa analise de complexidade
        data_shape = (100, 100, 100)
        analyzer.analyze_data_complexity(data_shape, "test_analysis")
        
        assert "test_analysis" in analyzer.complexity_analysis
        print("✓ Analise de complexidade funcionando")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no ComplexityAnalyzer: {e}")
        return False

def test_detailed_analyzer():
    """Testa a classe DetailedComplexityAnalyzer"""
    
    print("\nTestando DetailedComplexityAnalyzer...")
    
    try:
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
        print(f"✗ Erro no DetailedComplexityAnalyzer: {e}")
        return False

def test_monitor_performance():
    """Testa o decorator monitor_performance"""
    
    print("\nTestando decorator monitor_performance...")
    
    try:
        from oldroyd import monitor_performance, complexity_analyzer
        
        # Funcao de teste
        @monitor_performance
        def test_function():
            time.sleep(0.1)
            return "test_result"
        
        # Executa funcao
        result = test_function()
        
        assert result == "test_result", "Funcao nao retornou resultado esperado"
        assert "test_function" in complexity_analyzer.operation_times
        print("✓ Decorator monitor_performance funcionando")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no decorator: {e}")
        return False

def test_memory_functions():
    """Testa funcoes de memoria"""
    
    print("\nTestando funcoes de memoria...")
    
    try:
        from oldroyd import get_memory_usage, print_memory_status
        
        # Testa get_memory_usage
        memory = get_memory_usage()
        assert memory > 0, "Uso de memoria deve ser positivo"
        print("✓ get_memory_usage funcionando")
        
        # Testa print_memory_status (nao deve dar erro)
        print_memory_status("test")
        print("✓ print_memory_status funcionando")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro nas funcoes de memoria: {e}")
        return False

def test_dependencies():
    """Testa se todas as dependencias estao instaladas"""
    
    print("\nTestando dependencias...")
    
    dependencies = [
        ('numpy', 'np'),
        ('matplotlib', 'matplotlib'),
        ('psutil', 'psutil'),
        ('torch', 'torch'),
        ('sklearn', 'sklearn'),
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

def run_all_tests():
    """Executa todos os testes"""
    
    print("=" * 60)
    print("TESTE DA IMPLEMENTACAO DE ANALISE DE COMPLEXIDADE")
    print("=" * 60)
    
    tests = [
        ("Dependencias", test_dependencies),
        ("ComplexityAnalyzer", test_complexity_analyzer),
        ("DetailedComplexityAnalyzer", test_detailed_analyzer),
        ("Decorator monitor_performance", test_monitor_performance),
        ("Funcoes de memoria", test_memory_functions)
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