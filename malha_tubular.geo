// ==========================================
// 1. CONFIGURAÇÃO DO MOTOR GRÁFICO
// ==========================================
SetFactory("OpenCASCADE"); // Habilita operações booleanas (cortar/unir)

// ==========================================
// 2. PARÂMETROS
// ==========================================
lc = 0.3;          // Refinamento geral
lc_fino = 0.05;    // Refinamento perto da aduela

// Dimensões do Solo
L = 10.0;
H_total = 6.0;
H_camada_inf = 2.5; // A camada de areia vai do fundo (0) até 2.5m de altura

// Dimensões da Aduela
w_int = 2.0;
h_int = 1.5;
espessura = 0.20;

// Posição da Aduela (Note: se você mudar o Y, o corte é automático!)
x_aduela = 4.0; 
y_aduela = 1.5; // Está parcialmente na areia e parcialmente na argila? O Gmsh resolve.

// ==========================================
// 3. CRIAÇÃO DAS FORMAS BÁSICAS
// ==========================================

// Retângulo 1: Solo Inferior (Areia/Aquoso)
// Rectangle(ID) = {x, y, z, largura, altura};
Solo_Inferior = news; 
Rectangle(Solo_Inferior) = {0, 0, 0, L, H_camada_inf};

// Retângulo 2: Solo Superior (Argila/Menos permeável)
Solo_Superior = news;
Rectangle(Solo_Superior) = {0, H_camada_inf, 0, L, H_total - H_camada_inf};

// Retângulo 3: Aduela EXTERNA (O Concreto + O Buraco)
Aduela_Ext = news;
Rectangle(Aduela_Ext) = {x_aduela, y_aduela, 0, w_int + 2*espessura, h_int + 2*espessura};

// Retângulo 4: Aduela INTERNA (O Buraco de ar/água)
Aduela_Int = news;
Rectangle(Aduela_Int) = {x_aduela + espessura, y_aduela + espessura, 0, w_int, h_int};

// ==========================================
// 4. OPERAÇÕES BOOLEANAS (A Mágica)
// ==========================================

// Passo A: Definir a parede de Concreto
// Concreto = Aduela Externa - Aduela Interna
Concreto_Shape[] = BooleanDifference{ Surface{Aduela_Ext}; Delete; }{ Surface{Aduela_Int}; Delete; };

// Passo B: Inserir a Aduela nos Solos (Cortar o espaço dela)
// Isso recorta o formato da aduela tanto do solo de cima quanto do de baixo
Solos_Finais[] = BooleanDifference{ Surface{Solo_Inferior, Solo_Superior}; Delete; }{ Surface{Concreto_Shape[0]}; };

// Agora temos 3 superfícies principais no array Solos_Finais e Concreto_Shape:
// Solos_Finais[0] -> Solo Inferior Recortado
// Solos_Finais[1] -> Solo Superior Recortado
// Concreto_Shape[0] -> A Parede de Concreto

// ==========================================
// 5. REFINAMENTO DE MALHA (Opcional)
// ==========================================
// Força malha fina apenas nos pontos da aduela
PointsOfAduela[] = PointsOf{ Surface{Concreto_Shape[0]}; };
Characteristic Length{PointsOfAduela[]} = lc_fino;

// Malha grossa no resto
PointsOfSoil[] = PointsOf{ Surface{Solos_Finais[0], Solos_Finais[1]}; };
Characteristic Length{PointsOfSoil[]} = lc;

// ==========================================
// 6. TAGS FÍSICAS (O que você vai usar na equação)
// ==========================================

Physical Surface("Solo_Inferior_Areia") = {Solos_Finais[0]};
Physical Surface("Solo_Superior_Argila") = {Solos_Finais[1]};
Physical Surface("Concreto_Aduela") = {Concreto_Shape[0]};

// Fronteiras (Bordas)
// Aqui usamos um truque para pegar todas as linhas de contorno automaticamente
Physical Curve("Contorno_Externo") = CombinedBoundary{ Surface{Solos_Finais[0], Solos_Finais[1]}; };

// Interface interna do duto (para Seepage Face)
Physical Curve("Parede_Interna_Dreno") = CombinedBoundary{ Surface{Aduela_Int}; };

Mesh.Algorithm = 6;