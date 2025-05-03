# Simulación de la Transición de Fase Electrodébil con Monte Carlo Paralelizado


## Descripción  
Extensión de estudios previos sobre la transición de fase electrodébil mediante **Monte Carlo paralelizado**, logrando una aceleración de ~3× en tiempos de simulación. El modelo utiliza dos campos escalares ($\phi$ y $\chi$) en redes 1D/2D, con paralelización por temperaturas independientes para determinar la temperatura crítica ($T_c$) con alta precisión.

## Métodos  
- Algoritmo paralelizado (multiprocessing).  
- Simulaciones en redes 1D y 2D.  
- Validación cruzada secuencial vs. paralelo.  

## Resultados  
- Reducción de tiempo de simulación (~3×).  
- Determinación de $T_c$ gráficamente.  
