# Simulación de la Transición de Fase Electrodébil con Materia Oscura

## Descripción
Simulación Monte Carlo que estudia la transición de fase electrodébil incluyendo un campo de materia oscura acoplado al Higgs.

## Requisitos
- Python 3.x
- Bibliotecas: NumPy, Matplotlib

## Parámetros clave
- Masa del Higgs: 125 GeV
- VEV del Higgs: 246.22 GeV
- Masa materia oscura: 80 GeV
- Constantes de acoplamiento:
- λφ (Higgs): calculada
- λφχ (acoplamiento): 0.5
- λχ (materia oscura): 0.1

## Método
Algoritmo Metropolis-Hastings para evolucionar los campos φ (Higgs) y χ (materia oscura) en el rango 50-350 GeV.

## Resultados
El script genera un gráfico mostrando:
- Evolución de <φ> vs temperatura
- Evolución de <χ> vs temperatura

## Ejecución
1. Ejecutar todas las celdas del notebook en orden
2. Los resultados se grafican automáticamente

## Interpretación
- Identifica la temperatura crítica de transición
- Analiza el acoplamiento entre Higgs y materia oscura
- Estudia efectos de la materia oscura en la dinámica de la transición

## Archivos
- montecarlo_1D.ipynb: Notebook principal con la simulación

## Contacto
- Daniel Ruiz Mejía: danielruizm1610@gmail.com
