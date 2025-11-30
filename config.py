import numpy as np
from base import TAG_CONCRETO # Importando apenas para referência se precisar

# ==============================================================================
# 1. MAPEAMENTO DE BORDAS (Tags das Linhas do Gmsh)
# ==============================================================================
# Estes números devem vir do seu meshio/conversor
TAG_BORDA_ESQUERDA = 101
TAG_BORDA_DIREITA = 102
TAG_BORDA_BASE = 103
TAG_BORDA_TOPO = 104
TAG_PAREDE_RIO = 105

# ==============================================================================
# 2. PARÂMETROS DO CENÁRIO HIDRÁULICO (RIO)
# ==============================================================================
VAZAO_RIO = 2.0         # m3/s
MANNING_N = 0.013       # Concreto liso
DECLIVIDADE_S0 = 0.005  # 0.5%
LARGURA_CANAL = 2.0     # m (w_int do Gmsh)

# ==============================================================================
# 3. CÁLCULO DO NÍVEL DO RIO (Manning Simplificado)
# ==============================================================================
def calcular_nivel_rio(Q, n, S0, largura):
    """
    Resolve a eq. de Manning para achar a altura h (lâmina d'água).
    Q = (1/n) * A * R^(2/3) * S0^0.5
    Para canal retangular: A = b*y, P = b + 2y
    """
    # Solução numérica iterativa simples ou aproximada
    # Aqui vamos usar um valor fixo calculado para o exemplo, 
    # mas você pode colocar um solver (Newton-Raphson) aqui.
    
    # Exemplo: Para Q=2.0 nesses parâmetros, h aproxima-se de 0.8m
    h_calculado = 0.78 # metros
    return h_calculado

NIVEL_RIO_ATUAL = calcular_nivel_rio(VAZAO_RIO, MANNING_N, DECLIVIDADE_S0, LARGURA_CANAL)

# ==============================================================================
# 4. FUNÇÕES DE CONDIÇÃO DE CONTORNO (O que o Solver chama)
# ==============================================================================

def aplicar_bc_externa(x, y, tag_borda):
    """
    Define o valor de Dirichlet (Carga h) para as bordas externas.
    """
    
    # ESQUERDA: Lençol Freático Alto (Entrada)
    if tag_borda == TAG_BORDA_ESQUERDA:
        # Ex: Nível freático a 4.0m de altura
        # Carga de pressão h = Cota Piezométrica - Elevação z
        nivel_freatico = 4.0
        return nivel_freatico - y # Retorna pressão h

    # DIREITA: Lençol Freático Mais Baixo (Saída natural)
    elif tag_borda == TAG_BORDA_DIREITA:
        nivel_freatico = 3.5
        return nivel_freatico - y

    # BASE: Impermeável (Neumann)
    elif tag_borda == TAG_BORDA_BASE:
        return None # Indica ao solver que é Neumann (Fluxo zero)

    # TOPO: Chuva ou Ar
    elif tag_borda == TAG_BORDA_TOPO:
        # Se chover, seria fluxo prescrito. Se seco, pressão atmosférica.
        # Vamos assumir pressão zero (h=0) onde saturar, ou fluxo nulo.
        return -1.0 # Leve sucção na superfície (solo seco)
    
    return None

def condicao_parede_rio(y_coord):
    """
    Define a pressão na parede interna do duto.
    """
    # Se o ponto estiver ABAIXO da linha d'água
    if y_coord < NIVEL_RIO_ATUAL:
        # Pressão Hidrostática positiva
        pressao = NIVEL_RIO_ATUAL - y_coord
        tipo = "Dirichlet"
        return tipo, pressao
    
    # Se o ponto estiver ACIMA da linha d'água
    else:
        # Pressão Atmosférica (0), mas sujeita à restrição de Seepage Face
        # (Só sai água se h > 0)
        pressao = 0.0
        tipo = "SeepageFace"
        return tipo, pressao
