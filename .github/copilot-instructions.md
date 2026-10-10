# Copilot Instructions

## Big Picture
- Projeto modela drenagem profunda resolvendo a equação de Richards 2D em malhas exportadas do Gmsh; tudo roda em Python puro (NumPy + SciPy + meshio + h5py).
- `solver_richards_transiente.py` é o solver oficial: FEM P1, Van Genuchten–Mualem, Backward Euler com iterações de Picard, exporta coleções VTU e relata diagnósticos coloridos em `saida_terminal_pipeline.txt`.
- `solver_richards2.py` mantém um pipeline “clássico” simplificado usado para relatórios interativos; consulte-o se precisar reproduzir a experiência textual antiga, mas priorize o solver transiente.

## Código e Estruturas-Chave
- Malhas: `malha_tubular.geo/.msh` → `converter_malha.py` → `malha_dominio.xdmf` (triângulos + tags) e `malha_bordas.xdmf`. Sempre carregue tags de material via `malha_dominio.h5:data2`.
- Materiais: `base.py` define TAG_* constantes, propriedades Van Genuchten e utilitários (`condutividade_hidraulica`, `conteudo_umidade`). Nunca codifique números mágicos fora desse arquivo; use os helpers para garantir coerência.
- Cenário: `config.py` centraliza parâmetros hidráulicos globais (vazão, Manning, níveis). Use-o em novos scripts em vez de duplicar valores.
- Condições de contorno: atualmente identificadas por coordenadas extremas (min/max x,y). Se adicionar fronteiras especiais, reutilize as listas `nodes_*` e imponha Dirichlet modificando matrizes em formato LIL antes de converter para CSR.
- Monitoramento: tags de pontos físicos `Piezo_*` vindos do .geo são usados apenas para amostragem; leia-os diretamente da malha se precisar registrar séries específicas.

## Fluxos de Trabalho Essenciais
- Converter malha: `python converter_malha.py` (gera XDMF/HDF5 e imprime resumo).
- Rodar solver transiente: `python solver_richards_transiente.py`. Resultados parciais são salvos a cada 36 passos em `resultados/drenagem_*.vtu` e `drenagem.pvd`.
- Rodar pipeline clássico: `python solver_richards2.py` (gera `saida_solver2.vtu` e imprime relatório completo em terminal + `saida_terminal_pipeline.txt`).

## Padrões e Convenções
- Matrizes globais sempre montadas em `lil_matrix`; só converta para CSR imediatamente antes de chamar `spsolve` para evitar `SparseEfficiencyWarning`.
- Tempo padrão: 24 h, `dt=600 s`; Picard com `TOL_PICARD=1e-4`, `MAX_ITER=15`. Preserve esses valores a menos que o usuário peça outra coisa, e logue mudanças no relatório.
- Gravação de log: use `print_log` helper para toda saída importante — ele duplica no terminal e em `saida_terminal_pipeline.txt`.
- Exportação VTU: para malhas 2D, sempre crie coordenada z=0 (`np.column_stack`) antes de escrever com meshio.

## Dicas para Novas Contribuições
- Precisando de novos materiais/camadas, adicione tags e propriedades em `base.py` e atualize o `.geo`/converter conforme necessário.
- Ajustes em condições de contorno acopladas (ex.: nível do rio dinâmico) devem ocorrer no solver transiente dentro do loop Picard, mantendo a imposição direta após montar `K_global`.
- Scripts de utilidade/visualização devem ler `config.py` e reaproveitar `print_log` para manter o histórico centralizado.

> Feedback: avise se alguma parte do fluxo (malha → solver → saída) ainda estiver obscura para que possamos detalhar melhor nesta instrução.
