import numpy as np

# ==============================================================================
# 1. MAPEAMENTO DE TAGS (Deve bater com os IDs gerados no .xdmf/.msh)
# ==============================================================================
# Dica: Ao usar meshio, verifique quais números ele atribuiu.
# Geralmente segue a ordem de criação ou definição no .geo.
TAG_SOLO_INFERIOR = 1  # Areia
TAG_SOLO_SUPERIOR = 2  # Argila
TAG_ENVELOPE_BRITA = 3 # Brita
TAG_CONCRETO = 4       # Aduela

# ==============================================================================
# 2. PROPRIEDADES DOS MATERIAIS (Dicionário Central)
# ==============================================================================
# ks: Condutividade Saturada (m/s)
# theta_s: Porosidade / Conteúdo Saturado (m3/m3)
# theta_r: Conteúdo Residual (m3/m3)
# alpha: Parâmetro de Van Genuchten (1/m) - inverso da pressão de entrada de ar
# n: Parâmetro de forma de poros (adimensional)

MATERIAIS = {
    TAG_SOLO_INFERIOR: {
        "nome": "Areia Fina",
        "ks": 5.0e-5,    
        "theta_s": 0.35, 
        "theta_r": 0.05,
        "alpha": 3.5,    
        "n": 2.5
    },
    TAG_SOLO_SUPERIOR: {
        "nome": "Argila Limosa",
        "ks": 1.0e-7,
        "theta_s": 0.45,
        "theta_r": 0.15,
        "alpha": 0.8,
        "n": 1.4
    },
    TAG_ENVELOPE_BRITA: {
        "nome": "Brita de Drenagem",
        "ks": 1.0e-2,   # Muito alto!
        "theta_s": 0.40,
        "theta_r": 0.01,
        "alpha": 10.0,
        "n": 3.0
    },
    TAG_CONCRETO: {
        "nome": "Concreto da Aduela",
        "ks": 1.0e-13,  # Praticamente impermeável
        "theta_s": 0.10,
        "theta_r": 0.0,
        "alpha": 0.001,
        "n": 1.1
    }
}

# ==============================================================================
# 3. EQUAÇÕES CONSTITUTIVAS (Van Genuchten-Mualem)
# ==============================================================================

def calcular_se(h, alpha, n):
    """Calcula a Saturação Efetiva (Se). h deve ser pressão (negativa no não saturado)."""
    if h >= 0:
        return 1.0
    m = 1.0 - (1.0 / n)
    return (1.0 + abs(alpha * h)**n)**(-m)

def condutividade_hidraulica(h, material_tag):
    """
    Retorna o K(h) para uma dada pressão h e um tipo de material.
    """
    props = MATERIAIS.get(material_tag)
    if not props:
        raise ValueError(f"Material Tag {material_tag} não encontrado.")
    
    ks = props['ks']
    
    # Se estiver saturado (h >= 0), retorna Ks
    if h >= 0:
        return ks
        
    # Se não saturado, aplica Van Genuchten
    alpha = props['alpha']
    n = props['n']
    m = 1.0 - (1.0 / n)
    
    Se = calcular_se(h, alpha, n)
    
    # Fórmula de Mualem
    k_rel = (Se**0.5) * (1.0 - (1.0 - Se**(1.0/m))**m)**2
    return ks * k_rel

def capacidade_hidrica(h, material_tag):
    """
    Retorna C(h) = d(theta)/dh. Necessário para o termo transiente.
    """
    # Implementação da derivada da curva de retenção (Opcional por enquanto)
    pass