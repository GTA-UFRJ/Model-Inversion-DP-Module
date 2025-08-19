import subprocess
import sys
import os

def run_script(script_name, extra_args=None):
    """
    Executa um script Python usando o mesmo interpretador que está executando este script.
    Para a execução se o script chamado falhar.
    """
    if not os.path.exists(script_name):
        print(f"\n[ERRO FATAL] O script '{script_name}' não foi encontrado. Abortando.")
        sys.exit(1)

    print(f"\n{'='*50}\n--- Iniciando a execução de: {script_name} ---\n{'='*50}")
    
    try:
        cmd = [sys.executable, script_name]
        if extra_args:
            cmd.extend(extra_args)

        subprocess.run(cmd, check=True)

        print(f"\n--- Script '{script_name}' concluído com sucesso! ---")

    except subprocess.CalledProcessError:
        print(f"\n[ERRO FATAL] A execução de '{script_name}' falhou. Verifique o log de erros acima. Abortando.")
        sys.exit(1)

def main():
    """
    Define a ordem de execução de todos os scripts do projeto.
    """
    # 1. Geração de Dados Brutos
    run_script("MNIST_FL_inversion_comparison.py", [
        "--number_of_seeds", "30",
        "--classes", "0,1,2,3,4,5,6,7,8,9",
        "--noise_scales", "0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1"
    ])

    # 2. Análise dos Resultados e Geração de Gráficos
    analysis_scripts = [
        "organize_attack_results.py",
        "organize_accuracy_results.py",
    ]
    for script in analysis_scripts:
        run_script(script)

    # 3. Teste de Significância Estatística
    run_script("Mann_Whitney.py", [
        "--selected_noise", "0.07"
    ])

    print(f"\n{'='*50}\n>>> PROCESSO COMPLETO FINALIZADO COM SUCESSO! <<<\n{'='*50}")

if __name__ == "__main__":
    main()
