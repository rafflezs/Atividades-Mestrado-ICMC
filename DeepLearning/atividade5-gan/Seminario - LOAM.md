# Seminario - LOAM

# Introdução

- Gestão Florestal -> inventáros de madeira -> Realizado com varredura a Laser (TLS)
- VANTs primariamente utilizado a seguir trilhas

- Problemas no mapeamento devido limitações
   . LOAM precisam de texturam
   .

- O estudo propõe assimilar LOAM + VANTs para etimar o DAP da floresta.

# SLOAM
- Algoritmo odometria LIDAR
- Algoritmo LIDAR

## Modelagem
- Solo> plano definido pela normal e deslocamento.
- Arvore> cilindo parametrizado

## Estimativa dos Modelos
### Solo
- Devidoa a dificuldade da vegetação rasteira, utiliza-se SVD para estimar o Solo

### Arvores
- Uso de minimos quadrados

Estimar modelo -> Associar Dados

# Algoritmos

## Odometria
- Estima o movimento do LIDAR entre varreduras consecutivas
- Processa informações semânticas

## Mapeamento
- Recebe a entrada da __Odometria__
- Constroi o mapa global do ambiente a partir de multiplas varreduras
    . Realiza comparações da varredura atual com o mapeamento

# Aprendizado de Máquina
Devido a falta de datasets para treinamento, foi gerado um ambiente virtual para rotular nuvens de pontos.

## ERFNET
- Utiliza-se DL para extrair caracteristicas das arvores.
- Rede projetada para segmentação semântica
- Divide-se em __Encoder__ e __Decoder__

### Arquitetura

$$ Downsample -> Non Bottleneck ID -> Non Bottleneck ID -> Downsample -> Deconvolution $$

Entrada: matriz $h * w$
Saida: segmentação prevista para cada imagem amostrada (soma das previsões amostradas)

### O Solo
- Considera plano, aproveita-se de heuristica simples para extraão de caracteristicas.

### Detecção de Instâncias

# Experimentos
- Testes realizados por VANTs em Nova Jersey (EUA).
- 2 experimentos
    . Médio: caminhada em linha reta de um minuto no ambiente flroestal denso
    . DIficil: trajetoria de voo de 2 minutos em ambiente florestal denso
- Métricas de Avaliação:
    . Qualitativa: avaliação da trajetória e da nuvem de pontos
    . Quantitativa: Erro entre o inicio e fim da trajetória no experimento do VANT e DAP.

## Resultados
- SLOAM apresentou menor erro de desvio e apresentou boa contagem das árvores.
- GICP tem magnetude de desvio similar ao SLOAM
- A-SLOAM apresenta desvios significativos
- T265 falhou miserávelmente

- Deve-se considerar problemas de _ghosting_, possivelmente causados por rotação indevida.

# Conclusão
- SLOAM demonstrou ser boa abordagem, embora apresente lacunas que são preenchidas pelo DL.
- DL desepenha papel fundamental no método, durante a FCN para segmentação precisa das árvores.
    . Fornece recursos semanticos rovustos.
    . Permite estimativa precisa de atributos
    . Incrementa a robustez do sistema.