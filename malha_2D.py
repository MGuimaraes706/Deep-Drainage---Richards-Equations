import subprocess
import sys

# Executa a geração da malha 2D
print("[Pipeline] Gerando malha 2D...")
ret1 = subprocess.run([sys.executable, "malha_python.py"])
if ret1.returncode != 0:
    print("[Pipeline] Erro ao gerar a malha 2D!")
    sys.exit(1)

# Executa o solver
print("[Pipeline] Executando solver_richards2.py...")
ret2 = subprocess.run([sys.executable, "solver_richards2.py"])
if ret2.returncode != 0:
    print("[Pipeline] Erro ao executar o solver!")
    sys.exit(1)

print("[Pipeline] Processo completo!")
