// ==========================================
// MALHA FINAL: ADUELA EM SOLO ESTRATIFICADO
// ==========================================
SetFactory("OpenCASCADE");

// 1. PARÂMETROS GEOMÉTRICOS
lc = 0.4;           // Malha geral
lc_fino = 0.08;     // Malha fina na parede do duto (CRÍTICO para ver a infiltração)

// Solo
L = 10.0;           // Largura
H_total = 6.0;      // Altura
H_camada_inf = 2.5; // Interface Areia/Argila

// Aduela
w_int = 2.0; h_int = 1.5; espessura = 0.20;
x_aduela = 4.0; y_aduela = 1.0; 

// 2. FORMAS
Solo_Inf = news; Rectangle(Solo_Inf) = {0, 0, 0, L, H_camada_inf};
Solo_Sup = news; Rectangle(Solo_Sup) = {0, H_camada_inf, 0, L, H_total - H_camada_inf};

// Aduela (Externa e Interna)
Aduela_Ext = news; Rectangle(Aduela_Ext) = {x_aduela, y_aduela, 0, w_int+2*espessura, h_int+2*espessura};
Aduela_Int = news; Rectangle(Aduela_Int) = {x_aduela+espessura, y_aduela+espessura, 0, w_int, h_int};

// 3. CORTES (Operações Booleanas)
// Define a parede de concreto
Concreto_Shape[] = BooleanDifference{ Surface{Aduela_Ext}; Delete; }{ Surface{Aduela_Int}; Delete; };
// Corta os solos com a parede de concreto
Solos_Finais[] = BooleanDifference{ Surface{Solo_Inf, Solo_Sup}; Delete; }{ Surface{Concreto_Shape[0]}; };

// 4. REFINAMENTO
// Refinar apenas na interface concreto-solo e concreto-água
PointsOfAduela[] = PointsOf{ Surface{Concreto_Shape[0]}; };
Characteristic Length{PointsOfAduela[]} = lc_fino;
PointsOfSoil[] = PointsOf{ Surface{Solos_Finais[0], Solos_Finais[1]}; };
Characteristic Length{PointsOfSoil[]} = lc;

// 5. GRUPOS FÍSICOS (ATUALIZADO PARA SEU ESTUDO)
Physical Surface("Solo_Inferior") = {Solos_Finais[0]};
Physical Surface("Solo_Superior") = {Solos_Finais[1]};
Physical Surface("Concreto") = {Concreto_Shape[0]};

// Bordas Externas
Physical Curve("Contorno_Externo") = CombinedBoundary{ Surface{Solos_Finais[0], Solos_Finais[1]}; };

// Borda Interna (Onde o Rio passa)
// Aqui o solver aplicará a pressão do rio
Physical Curve("Parede_Interna_Rio") = CombinedBoundary{ Surface{Aduela_Int}; };

Mesh.Algorithm = 6;