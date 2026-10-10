# Deep Drainage - Richards Equations (Pipeline Clássico)

## 📌 Sobre o Projeto
Este repositório contém um simulador computacional desenvolvido em **Python** (utilizando a biblioteca **FEniCS** para o Método de Elementos Finitos) focado na resolução da **Equação de Richards**. Ele modela o fluxo de fluidos em meios porosos (saturados e não-saturados), aplicado a problemas de drenagem profunda e infiltração.

O pipeline de simulação é robusto e engloba a geração de malhas não estruturadas via **Gmsh**, conversão eficiente de domínios, e saída de dados em alta fidelidade para o **ParaView**.

> 📖 **Documentação Matemática:** Para compreender em detalhes a formulação fraca, discretização e as técnicas numéricas empregadas no solver, consulte o documento [**`guia.pdf`**](./guia.pdf) anexo a este repositório.

## 🚀 Fluxo de Trabalho

1. **Gere a malha no Gmsh**
   - Abra o arquivo `.geo` no Gmsh.
   - Exporte a malha como `malha_tubular.msh` (2D, formato Gmsh 2.2 recomendado).

2. **Converta a malha para XDMF**
   - Execute:
     ```powershell
     python converter_malha.py
     ```
   - Isso irá gerar:
     - `malha_dominio.xdmf` (domínio 2D)
     - `malha_bordas.xdmf` (bordas/condições de contorno)

3. **Execute o solver**
   - Execute:
     ```powershell
     python solver_richards2.py
     ```
   - O terminal exibirá um relatório colorido e detalhado, incluindo:
     - Resumo da malha
     - Parâmetros físicos do cenário
     - Configuração do solver
     - Condições de contorno
     - Progresso temporal
     - Arquivos gerados
     - Estatísticas finais

4. **Saídas**
   - `resultados/drenagem.pvd` (visualização no ParaView)
   - `malha_dominio.h5` (malha em HDF5)

## Observações
- O pipeline NÃO depende de scripts de geração de malha em Python.
- Apenas os arquivos `converter_malha.py` e `solver_richards2.py` são necessários após a geração da malha no Gmsh.
- Parâmetros físicos e de cenário estão em `config.py` e `base.py`.
- O relatório do solver mostra todos os dados relevantes do cenário e execução.

## Exemplo de Relatório

```
================================================================================
RELATÓRIO DE EXECUÇÃO DA SIMULAÇÃO

################################################################################
[1] DADOS DE ENTRADA (INPUT)
################################################################################
[ARQUIVO DE MALHA]

Caminho:         malha_dominio.xdmf
Caminho Bordas:  malha_bordas.xdmf
Pontos (Nós):    299
Elementos (Tri): 66

[PARÂMETROS FÍSICOS]

Vazão do Rio (Q):          2.00 m³/s
Coeficiente de Manning (n): 0.013
Declividade (S0):          0.005 m/m
Cota de Fundo do Duto:     1.000 m
Lâmina d'água (h):         0.780 m
Nível Absoluto da Água:    1.780 m

[CONFIGURAÇÃO DO SOLVER]

Tempo Total:               24 horas
Passo de Tempo (dt):       3600.0 s (1 hora)
Método de Integração:      Backward Euler
Tolerância Picard:         1e-4
Max Iterações:             15

[CONDIÇÕES DE CONTORNO]

Esquerda: Dirichlet Variável (Lençol Alto)
Direita:  Dirichlet Variável (Lençol Baixo)
Topo:     Dirichlet Constante (Sucção)
Rio:      Mista (Seepage Face / Hidrostática)

################################################################################
[2] DADOS DE SAÍDA (OUTPUT)
################################################################################
[PROCESSAMENTO TEMPORAL]

Tempo t =  1.0 h | Status: Convergiu (Iter: 4) | Erro: 1.25e-04
...
Tempo t = 24.0 h | Status: Convergiu (Iter: 3) | Erro: 3.21e-05

[ARQUIVOS GERADOS]

Visualização: resultados/drenagem.pvd (ParaView)
Malha HDF5:   malha_dominio.h5

[RESUMO ESTATÍSTICO FINAL (t=24h)]

Pressão Mínima:      -2.450 m (Superfície Seca)
Pressão Máxima:       0.780 m (Fundo do Rio)
Pressão Média:       -0.542 m
Status da Execução:  SUCESSO
================================================================================
FIM DO RELATÓRIO
```

---

Dúvidas? Consulte os scripts ou peça ajuda!
