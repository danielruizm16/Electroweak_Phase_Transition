# Simulación de la Transición de Fase Electrodébil con Materia Oscura

## Descripción
Simulación Monte Carlo que estudia la transición de fase electrodébil incluyendo un campo de materia oscura acoplado al Higgs.

## Utilidad del proyecto

El proyecto cuenta con varias aplicaciones. En primer lugar, permite investigar la fenomenología de posibles candidatos a materia oscura mediante el estudio de cómo un campo escalar adicional podría acoplarse al bosón de Higgs, analizando los efectos observables durante la transición de fase electrodébil. Desde la perspectiva de física más allá del Modelo Estándar, proporciona un marco teórico-práctico al estudiar la posibilidad de una transición de fase de primer orden, la cual se caracteriza por un fondo estocástico de ondas gravitacionales, contrario al caso del modelo estándar en el que la transición de fase electrodébil es un crossover. Técnicamente, implementa métodos numéricos avanzados como el algoritmo Metropolis-Hastings aplicado a teoría cuántica de campos, sentando las bases para simulaciones más complejas


## Requisitos
- Python 3.x
- Bibliotecas: NumPy, Matplotlib

## Parámetros clave
- Masa del Higgs: 125 GeV
- VEV del Higgs: 246.22 GeV
- Masa materia oscura: 80 GeV (Parámetro a variar)
- Constantes de acople:
- $\lambda_\phi$ (Higgs): calculada
- $\lambda_{\phi \chi}$ (acople): 0.5 (Parámetro a variar)
- $\lambda_\chi$ (materia oscura): 0.1

## Método
Algoritmo Metropolis-Hastings para evolucionar los campos $\phi$ (Higgs) y $\chi$ (materia oscura) en el rango de temperaturas de 50-350 GeV.

## Resultados
El script genera un gráfico mostrando:
- Evolución de $\langle \phi \rangle$ vs temperatura
- Evolución de $\langle \chi \rangle$ vs temperatura

## Ejecución
1. Ejecutar todas las celdas del notebook en orden
2. Los resultados se grafican automáticamente

## Interpretación
- Identifica la temperatura crítica de transición
- Analiza la dependencia con el acople entre el Higgs y la materia oscura
- Estudia efectos de la masa de la materia oscura en la dinámica de la transición
- Considerar la posibilidad de obtener una transición de fase de primer orden

## Archivos
- montecarlo_1D.ipynb: Notebook principal con la simulación
- lambda_0.2_mDM_150: Gráfica para $\lambda_{\phi \chi}=0.2$ y $m_{DM}=150$ GeV
- lambda_0.5_mDM_100: Gráfica para $\lambda_{\phi \chi}=0.5$ y $m_{DM}=100$ GeV
- lambda_0.5_mDM_200: Gráfica para $\lambda_{\phi \chi}=0.5$ y $m_{DM}=200$ GeV
- lambda_0.5_mDM_80: Gráfica para $\lambda_{\phi \chi}=0.5$ y $m_{DM}=80$ GeV


## Contacto
- Daniel Ruiz Mejía: danielruizm1610@gmail.com
