import numpy as np

# ==============================================================================
# 1. MAPEAMENTO DE TAGS (Deve bater com o .xdmf gerado)
# ==============================================================================
TAG_SOLO_INFERIOR = 1  # Areia
TAG_SOLO_SUPERIOR = 2  # Argila
TAG_ENVELOPE_BRITA = 3 # Brita
TAG_CONCRETO = 4       # Aduela

# ==============================================================================
# 2. PROPRIEDADES DOS MATERIAIS
# ==============================================================================
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
        "ks": 1.0e-2,   
        "theta_s": 0.40,
        "theta_r": 0.01,
        "alpha": 10.0,
        "n": 3.0
    },
    TAG_CONCRETO: {
        "nome": "Concreto",
        "ks": 1.0e-13,  
        "theta_s": 0.10,
        "theta_r": 0.00,
        "alpha": 0.001,
        "n": 1.1
    }
}

def obter_propriedades(material_tag):
    """Retorna propriedades do material de forma segura."""
    try:
        tag = int(material_tag)
    except:
        return MATERIAIS[TAG_SOLO_INFERIOR]
    return MATERIAIS.get(tag, MATERIAIS[TAG_SOLO_INFERIOR])

def condutividade_hidraulica(h, material_tag):
    """
    Retorna o K(h) calculado via Van Genuchten.
    """
    props = obter_propriedades(material_tag)
    ks = props['ks']
    
    if h >= 0:
        return ks
        
    alpha = props['alpha']
    n = props['n']
    m = 1.0 - (1.0 / n)
    
    # Saturação Efetiva (com proteção matemática abs)
    denom = (1.0 + abs(alpha * h)**n)**m
    Se = 1.0 / denom
    
    k_rel = (Se**0.5) * (1.0 - (1.0 - Se**(1.0/m))**m)**2
    return ks * k_rel

def conteudo_umidade(h, material_tag):
    """
    Retorna Theta(h) para cálculo da capacidade hídrica.
    """
    props = obter_propriedades(material_tag)
    theta_s = props['theta_s']
    theta_r = props.get('theta_r', 0.01)
    alpha = props['alpha']
    n = props['n']
    m = 1.0 - (1.0/n)
    
    if h >= 0:
        return theta_s
        
    denom = (1.0 + abs(alpha * h)**n)**m
    Se = 1.0 / denom
    
    return theta_r + (theta_s - theta_r) * Se

Q_RIO = 2.0  # Vazão do Rio (m³/s)