SetFactory("OpenCASCADE");

// ==========================================
// 1. PARÂMETROS
// ==========================================
lc = 0.5;           // Malha Solo
lc_brita = 0.15;    // Malha Brita (Transição)
lc_concreto = 0.1;  // Malha Concreto

// Solo
L = 10.0; H_total = 6.0; H_camada_inf = 2.5;

// Aduela (Concreto)
w_int = 2.0; h_int = 1.5; espessura = 0.20;
x_aduela = 4.0; y_aduela = 1.0; 

// Envelope de Brita (NOVO PARÂMETRO)
esp_brita = 0.30; // 30cm de brita ao redor da aduela

// ==========================================
// 2. FORMAS GEOMÉTRICAS
// ==========================================

// Solos
Solo_Inf = news; Rectangle(Solo_Inf) = {0, 0, 0, L, H_camada_inf};
Solo_Sup = news; Rectangle(Solo_Sup) = {0, H_camada_inf, 0, L, H_total - H_camada_inf};

// Envelope de Brita (Retângulo maior que a aduela)
// Ele começa antes e termina depois da aduela
Brita_Shape = news; 
Rectangle(Brita_Shape) = {x_aduela - esp_brita, y_aduela - esp_brita, 0, 
                          w_int + 2*espessura + 2*esp_brita, 
                          h_int + 2*espessura + 2*esp_brita};

// Aduela (Externa e Interna)
Aduela_Ext = news; Rectangle(Aduela_Ext) = {x_aduela, y_aduela, 0, w_int+2*espessura, h_int+2*espessura};
Aduela_Int = news; Rectangle(Aduela_Int) = {x_aduela+espessura, y_aduela+espessura, 0, w_int, h_int};

// ==========================================
// 3. CORTES (A ordem importa muito aqui!)
// ==========================================

// A. Definir o Concreto (Aduela Ext - Aduela Int)
Concreto_Final[] = BooleanDifference{ Surface{Aduela_Ext}; Delete; }{ Surface{Aduela_Int}; Delete; };

// B. Definir a Brita (Brita Bruta - Concreto Ext)
// O "buraco" da brita é o retângulo externo da aduela
// Nota: Usamos BooleanDifference mantendo a geometria da aduela para não sumir com ela
Brita_Final[] = BooleanDifference{ Surface{Brita_Shape}; Delete; }{ Surface{Concreto_Final[0]}; };

// C. Definir os Solos (Solos Brutos - Brita Externa)
Solos_Finais[] = BooleanDifference{ Surface{Solo_Inf, Solo_Sup}; Delete; }{ Surface{Brita_Final[0]}; };

// ==========================================
// 4. PIEZÔMETROS (PONTOS DE OBSERVAÇÃO)
// ==========================================
// Criamos pontos físicos forçados na malha para extrair dados exatos depois

// Piezômetro 1: Longe da aduela (ver nível original)
P1 = newp; Point(P1) = {1.0, 4.0, 0, lc};

// Piezômetro 2: Perto da aduela (ver cone de rebaixamento)
P2 = newp; Point(P2) = {3.0, 2.0, 0, lc};

// Piezômetro 3: Embaixo da aduela (ver subpressão)
P3 = newp; Point(P3) = {x_aduela + w_int/2, y_aduela - 0.5, 0, lc};

// Forçar esses pontos na malha
Point{P1, P2, P3} In Surface{Solos_Finais[1]}; // Ou no solo correspondente

// ==========================================
// 5. REFINAMENTO E TAGS
// ==========================================

// Refinar Brita
PointsOfBrita[] = PointsOf{ Surface{Brita_Final[0]}; };
Characteristic Length{PointsOfBrita[]} = lc_brita;

// Tags Físicas
Physical Surface("Solo_Inferior") = {Solos_Finais[0]};
Physical Surface("Solo_Superior") = {Solos_Finais[1]};
Physical Surface("Envelope_Brita") = {Brita_Final[0]}; // Nova Região!
Physical Surface("Concreto") = {Concreto_Final[0]};

// Fronteiras
Physical Curve("Contorno_Externo") = CombinedBoundary{ Surface{Solos_Finais[0], Solos_Finais[1]}; };
Physical Curve("Parede_Interna_Rio") = CombinedBoundary{ Surface{Aduela_Int}; }; // Onde o rio passa

// Pontos de Monitoramento (Para o solver salvar histórico aqui)
Physical Point("Piezo_Longe") = {P1};
Physical Point("Piezo_Perto") = {P2};
Physical Point("Piezo_Fundo") = {P3};

Mesh.Algorithm = 6;