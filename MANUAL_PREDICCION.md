# Manual técnico: análisis, modelado y predicción de sucesos

> **Premisa (hipotética, NO demostrada).** Todo suceso es *potencialmente* modelable
> y predecible cuando existen datos suficientes, variables observables, leyes
> conocidas, capacidad de cálculo y un modelo adecuado. Esto **no es una verdad
> científica probada**: es un marco de trabajo. En cada predicción se señalan
> supuestos, límites, errores e incertidumbre. Cuando los datos no alcanzan, la
> respuesta correcta es: **"No puede realizarse una predicción confiable con los
> datos disponibles."**

**Ecuación rectora del manual:**

```
Predicción = datos + modelo + supuestos + capacidad de cálculo + incertidumbre
```

y toda predicción se expresa como una **distribución condicional**, no un número:

$$P(Y \mid X, D, M)$$

donde **Y** = suceso, **X** = variables observadas, **D** = datos disponibles, **M** = modelo.

---

## Índice
1. Los siete profesionales y sus herramientas
2. Instrumentos, software y lenguajes
3. Catálogo de fórmulas (física, química, biología, estadística, caos)
4. Ficha de cada fórmula: qué predice, datos, supuestos, fallos, ejemplo, verificación
5. Procedimiento universal de modelado (12 pasos)
6. Código funcional (Python, ejecutado y validado)
7. Distinciones clave (predicción, explicación, correlación, causalidad…)
8. Por qué la predicción es probabilística: P(Y|X,D,M)
9. Arquitectura computacional integrada
10. Pseudocódigo universal
11. Límites de "todo es predecible"
12. Conclusión

---

## 1. Los siete profesionales y sus herramientas

| Profesional | Qué aporta al problema de predicción | Herramientas núcleo |
|---|---|---|
| **Ingeniero** | Diseño de sistemas, control, medición, tolerancias | CAD, MATLAB/Simulink, control PID, FEM (elementos finitos), sensores/PLC |
| **Químico** | Cinética, equilibrio, termodinámica de reacciones | Espectrofotómetro, cromatografía (GC/HPLC), calorimetría, software cinético |
| **Matemático** | Formalización, EDO/EDP, optimización, demostración | Álgebra lineal, cálculo, teoría de EDP, SymPy, teoría de la medida |
| **Biólogo** | Dinámica de poblaciones, genética, epidemiología | PCR, secuenciación, microscopía, modelos SIR/Lotka-Volterra |
| **Físico** | Leyes fundamentales, mecánica, fluidos, termodinámica | Simulación numérica, Navier-Stokes, mecánica estadística, LabVIEW |
| **Estadístico** | Inferencia, incertidumbre, diseño experimental | Contrastes de hipótesis, regresión, bootstrap, R, diseño factorial |
| **Científico de datos** | Datos masivos, ML, validación fuera de muestra | Python (pandas, scikit-learn, PyTorch), pipelines, MLOps, feature engineering |

**Principio de integración:** ningún profesional predice solo. El físico aporta la
**ley**, el estadístico la **incertidumbre**, el científico de datos el **ajuste
empírico** y el ingeniero la **medición**. La predicción robusta combina modelo
mecanicista (por qué) con modelo estadístico (cuánto y con qué error).

---

## 2. Instrumentos, software y lenguajes

- **Sensores/instrumentos:** termopares, acelerómetros (IMU), pH-metros,
  espectrofotómetros, GPS/RTK, LIDAR, sensores de flujo, células de carga.
- **Métodos de laboratorio:** titulación, cromatografía, espectroscopía UV-Vis
  (Beer-Lambert), electroquímica (Nernst), calorimetría, qPCR.
- **Técnicas estadísticas:** regresión, ANOVA, bootstrap, MCMC, series temporales
  (ARIMA), pruebas de bondad de ajuste (χ², KS), corrección por comparaciones
  múltiples (Benjamini-Hochberg, Holm).
- **Simulación:** elementos finitos (FEM), dinámica de fluidos computacional (CFD),
  Monte Carlo, dinámica molecular, agentes.
- **Software:** MATLAB, R, Python, COMSOL, ANSYS, OpenFOAM, Stan/PyMC, TensorFlow.
- **Lenguajes:** Python (prototipado y ML), R (estadística), C++/Fortran (cálculo
  intensivo), Julia (científico moderno), SQL (datos).

---

## 3–4. Catálogo de fórmulas

> Formato de cada ficha: **fórmula → símbolos/unidades → qué predice → datos →
> supuestos → cuándo falla → ejemplo → verificación experimental.**

### FÍSICA

#### Segunda ley de Newton
$$\vec{F} = m\,\vec{a} = m\frac{d^2\vec{x}}{dt^2}$$
- **Símbolos:** F = fuerza [N]; m = masa [kg]; a = aceleración [m/s²]; x = posición [m].
- **Predice:** trayectoria/aceleración de un cuerpo dadas las fuerzas.
- **Datos:** masa, fuerzas aplicadas, condiciones iniciales (x₀, v₀).
- **Supuestos:** marco inercial, masa constante, velocidades ≪ c.
- **Falla:** relativista (v→c), cuántico (escala atómica), masa variable (cohetes).
- **Ejemplo:** F=10 N, m=2 kg → a = 5 m/s². Tras 3 s desde reposo: v = a·t = 15 m/s.
- **Verificación:** medir con acelerómetro/cronómetro y comparar; residuo < error del sensor.

#### Conservación de masa, energía y momento
$$\frac{\partial \rho}{\partial t} + \nabla\!\cdot(\rho \vec{u}) = 0 \quad;\quad E_{\text{tot}} = \text{cte} \quad;\quad \sum \vec{p} = \text{cte}$$
- **Símbolos:** ρ = densidad [kg/m³]; u = velocidad [m/s]; p = m·v = momento [kg·m/s]; E = energía [J].
- **Predice:** estados finales en colisiones, flujos, reacciones (lo que entra = lo que sale ± acumulación).
- **Supuestos:** sistema cerrado/aislado según la magnitud conservada.
- **Falla:** sistemas abiertos sin contabilizar flujos; conversión masa-energía (nuclear).
- **Ejemplo (momento):** dos masas 1 kg a +3 y −1 m/s chocan y quedan juntas → v = (3−1)/2 = 1 m/s.
- **Verificación:** medir velocidades antes/después; el momento total debe coincidir dentro del error.

#### Ecuación del calor (difusión)
$$\frac{\partial T}{\partial t} = \alpha\,\nabla^2 T$$
- **Símbolos:** T = temperatura [K]; α = difusividad térmica [m²/s]; ∇² = laplaciano.
- **Predice:** cómo se distribuye la temperatura en el tiempo y el espacio.
- **Datos:** α del material, condiciones de contorno e iniciales.
- **Supuestos:** medio homogéneo, sin convección ni radiación dominante.
- **Falla:** cambios de fase, materiales anisótropos, tiempos muy cortos (ley de Fourier no local).
- **Ejemplo:** barra con extremos a 0 °C, centro a 100 °C → decae exponencialmente hacia 0.
- **Verificación:** termopares en varios puntos; comparar el perfil medido con la solución numérica.

#### Ecuaciones de Navier-Stokes (fluidos)
$$\rho\left(\frac{\partial \vec{u}}{\partial t} + \vec{u}\cdot\nabla\vec{u}\right) = -\nabla p + \mu\nabla^2\vec{u} + \vec{f}$$
- **Símbolos:** u = velocidad; p = presión [Pa]; μ = viscosidad [Pa·s]; f = fuerzas de cuerpo.
- **Predice:** campo de velocidad/presión de un fluido (clima, aerodinámica, sangre).
- **Supuestos:** fluido newtoniano, continuo; incompresible si ∇·u = 0.
- **Falla:** turbulencia plena (sin resolución suficiente → se usa RANS/LES), gases enrarecidos.
- **Ejemplo:** flujo laminar en tubería (Poiseuille): perfil parabólico de velocidad.
- **Verificación:** anemometría/PIV; comparar campo medido con CFD (OpenFOAM). *Advertencia: existencia/suavidad general es un problema abierto del milenio.*

#### EDO y EDP
$$\text{EDO: } \frac{dy}{dt}=f(y,t) \qquad \text{EDP: } \frac{\partial u}{\partial t}=F\!\left(u,\frac{\partial u}{\partial x},\dots\right)$$
- **Predice:** evolución temporal (EDO) o espacio-temporal (EDP) de un sistema.
- **Datos:** función f, condiciones iniciales/de contorno.
- **Supuestos:** existencia y unicidad (Lipschitz para EDO).
- **Falla:** rigidez numérica, singularidades, condiciones de contorno mal planteadas.
- **Ejemplo:** dy/dt = −0.3y, y₀=100 → y(t)=100·e^(−0.3t) (ver código, coincide con numérico).
- **Verificación:** medir y(t) en varios instantes; comparar con la solución integrada.

### PROBABILIDAD Y ESTADÍSTICA

#### Teorema de Bayes
$$P(H\mid E) = \frac{P(E\mid H)\,P(H)}{P(E)}$$
- **Símbolos:** H = hipótesis; E = evidencia; P(H) = prior; P(H|E) = posterior.
- **Predice:** creencia actualizada tras observar datos.
- **Supuestos:** verosimilitud P(E|H) y prior bien definidos.
- **Falla:** prior mal elegido con pocos datos; verosimilitud incorrecta.
- **Ejemplo:** test con sensibilidad 99%, especificidad 95%, prevalencia 1% → P(enfermo|+) ≈ 16.7%. *(La baja prevalencia domina.)*
- **Verificación:** calibración: de todos los casos a los que asignas 70%, ~70% deben cumplirse.

#### Esperanza, varianza, distribuciones
$$E[X]=\sum x_i p_i \quad;\quad \mathrm{Var}(X)=E[(X-\mu)^2] \quad;\quad X\sim\mathcal{N}(\mu,\sigma^2)$$
- **Predice:** valor medio esperado y su dispersión.
- **Supuestos:** la distribución elegida (Normal, Poisson…) describe el fenómeno.
- **Falla:** colas pesadas (usar t-Student), datos multimodales, no estacionarios.
- **Ejemplo:** dado justo: E[X] = 3.5, Var = 2.92.
- **Verificación:** histograma empírico vs. distribución teórica (test KS/χ²).

#### Regresión lineal y logística
$$y = \beta_0 + \beta_1 x + \varepsilon \qquad P(y=1)=\frac{1}{1+e^{-(\beta_0+\beta_1 x)}}$$
- **Predice:** lineal → valor continuo; logística → probabilidad de clase.
- **Datos:** pares (x, y); logística requiere etiquetas 0/1.
- **Supuestos:** linealidad (en los parámetros), errores independientes; logística: log-odds lineal.
- **Falla:** relaciones no lineales, colinealidad, valores atípicos, extrapolación.
- **Ejemplo:** ver código — pendiente≈2.5, R² alto en test.
- **Verificación:** MAE/RMSE/R² en datos independientes; curva ROC para logística.

#### Series temporales
$$y_t = c + \sum_{i=1}^{p}\phi_i y_{t-i} + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j} + \varepsilon_t \quad(\text{ARIMA})$$
- **Predice:** valores futuros de una serie a partir de su historia.
- **Supuestos:** estacionariedad (tras diferenciar), estructura autorregresiva estable.
- **Falla:** cambios de régimen, tendencias no capturadas; **el error crece con el horizonte**.
- **Ejemplo:** ver código — AR(1) φ≈0.77; *el pronóstico multi-paso da R² negativo (peor que la media).*
- **Verificación:** validación temporal walk-forward; comparar contra un baseline ingenuo (último valor).

#### Proceso de Poisson
$$P(N=k)=\frac{(\lambda t)^k e^{-\lambda t}}{k!}$$
- **Símbolos:** λ = tasa media de eventos [1/t]; k = nº de eventos.
- **Predice:** nº de eventos raros e independientes en un intervalo (llamadas, fallos, desintegraciones).
- **Supuestos:** eventos independientes, tasa constante, no simultáneos.
- **Falla:** eventos agrupados (sobredispersión → usar binomial negativa), tasa variable.
- **Ejemplo:** λ=3 llamadas/hora → P(exactamente 5) = 3⁵e⁻³/5! ≈ 0.101.
- **Verificación:** contar eventos por intervalo; la media ≈ varianza si es Poisson.

#### Cadenas de Markov
$$P(X_{t+1}=j\mid X_t=i)=P_{ij}, \qquad \pi = \pi P$$
- **Predice:** probabilidad de estados futuros; distribución estacionaria π a largo plazo.
- **Supuestos:** propiedad de Markov (el futuro depende solo del presente).
- **Falla:** si hay memoria de más de un paso; matriz no ergódica.
- **Ejemplo:** P=[[0.7,0.3],[0.4,0.6]] → estacionaria π=[0.571, 0.429] (ver código).
- **Verificación:** estimar P de datos y comprobar que las frecuencias largas convergen a π.

#### Filtro de Kalman
$$\hat{x}_{k}= \hat{x}_{k}^- + K_k(z_k - \hat{x}_k^-), \qquad K_k=\frac{P_k^-}{P_k^-+R}$$
- **Predice:** el estado real de un sistema a partir de mediciones ruidosas (fusión sensor+modelo).
- **Supuestos:** sistema lineal, ruido gaussiano de covarianzas Q (proceso) y R (medición) conocidas.
- **Falla:** no linealidades fuertes (usar EKF/UKF), ruido no gaussiano.
- **Ejemplo:** ver código — reduce el RMSE de 0.33 (medición cruda) a 0.18.
- **Verificación:** comparar con verdad conocida (simulación) o con un sensor de referencia.

#### Métodos de Monte Carlo
$$\hat{\theta}=\frac{1}{N}\sum_{i=1}^{N} g(X_i), \qquad \text{error} \sim \frac{\sigma}{\sqrt{N}}$$
- **Predice:** integrales, probabilidades, propagación de incertidumbre por muestreo aleatorio.
- **Supuestos:** poder muestrear de la distribución correcta; muestras independientes.
- **Falla:** convergencia lenta (∝1/√N), maldición de la dimensión, muestreo sesgado.
- **Ejemplo:** ver código — π ≈ 3.141 ± 0.004 (IC 95%).
- **Verificación:** el error debe caer como 1/√N al aumentar las muestras.

#### Optimización matemática
$$\min_{\theta}\; L(\theta) \quad\Rightarrow\quad \nabla L(\theta^*)=0$$
- **Predice:** los parámetros que mejor ajustan un modelo o el óptimo de un sistema.
- **Métodos:** descenso de gradiente, Newton, programación lineal/entera, algoritmos genéticos.
- **Falla:** óptimos locales, funciones no convexas/no diferenciables, mal condicionamiento.
- **Ejemplo:** ajustar β minimizando el error cuadrático (mínimos cuadrados tiene solución cerrada).
- **Verificación:** comprobar que el gradiente ≈ 0 y que perturbar θ* empeora L.

### QUÍMICA

#### Ecuación de Arrhenius
$$k = A\,e^{-E_a/(RT)}$$
- **Símbolos:** k = constante de velocidad; A = factor pre-exponencial; Eₐ = energía de activación [J/mol]; R=8.314 J/(mol·K); T [K].
- **Predice:** cómo cambia la velocidad de reacción con la temperatura.
- **Supuestos:** Eₐ constante en el rango; una sola etapa dominante.
- **Falla:** mecanismos multi-etapa, catálisis, T fuera de rango.
- **Ejemplo:** ver código — k(298K)=0.72, k(310K)=2.30 → sube ~3× con +12 K.
- **Verificación:** medir k a varias T; ln k vs 1/T debe ser recta de pendiente −Eₐ/R.

#### Cinética y equilibrio químico
$$v = k[A]^m[B]^n \qquad K_{eq}=\frac{[C]^c[D]^d}{[A]^a[B]^b}$$
- **Predice:** velocidad de reacción y composición en equilibrio.
- **Supuestos:** sistema bien mezclado, T constante, órdenes de reacción conocidos.
- **Falla:** reacciones controladas por difusión, lejos del equilibrio.
- **Ejemplo:** A→B de primer orden: [A](t)=[A]₀e^(−kt).
- **Verificación:** seguir concentraciones por espectrofotometría vs. tiempo.

#### Energía libre de Gibbs
$$\Delta G = \Delta H - T\Delta S, \qquad \Delta G = -RT\ln K_{eq}$$
- **Predice:** si una reacción es espontánea (ΔG<0) y su equilibrio.
- **Supuestos:** T y P constantes.
- **Falla:** sistemas fuera de equilibrio, aproximaciones de actividad ideal.
- **Ejemplo:** ΔH=−50 kJ/mol, ΔS=−0.1 kJ/(mol·K), T=298 K → ΔG=−50−298(−0.1)=−20.2 kJ/mol → espontánea.
- **Verificación:** medir K_eq y despejar ΔG; calorimetría para ΔH.

#### Ley de Beer-Lambert
$$A = \varepsilon\, c\, l$$
- **Símbolos:** A = absorbancia; ε = absortividad molar [L/(mol·cm)]; c = concentración [mol/L]; l = camino [cm].
- **Predice:** concentración a partir de la luz absorbada.
- **Supuestos:** solución diluida, luz monocromática, sin dispersión.
- **Falla:** concentraciones altas (desviaciones), turbidez, fluorescencia.
- **Ejemplo:** ε=1000, l=1 cm, A=0.5 → c = 5×10⁻⁴ mol/L.
- **Verificación:** curva de calibración con estándares conocidos (recta A vs c).

#### Ecuación de Nernst
$$E = E^\circ - \frac{RT}{nF}\ln Q$$
- **Símbolos:** E = potencial [V]; E° = estándar; n = electrones; F=96485 C/mol; Q = cociente de reacción.
- **Predice:** potencial de una celda electroquímica según concentraciones.
- **Supuestos:** equilibrio en el electrodo, actividades ≈ concentraciones.
- **Falla:** corrientes altas, soluciones concentradas.
- **Ejemplo:** E°=0.34 V, n=2, Q=0.01, 298 K → E = 0.34 − (0.0257/2)ln(0.01) ≈ 0.399 V.
- **Verificación:** medir con voltímetro variando concentración.

### BIOLOGÍA

#### Crecimiento exponencial y logístico
$$\frac{dN}{dt}=rN \quad\Rightarrow\quad N(t)=N_0 e^{rt} \qquad ; \qquad \frac{dN}{dt}=rN\!\left(1-\frac{N}{K}\right)$$
- **Predice:** tamaño poblacional (exponencial sin límite; logístico con capacidad K).
- **Supuestos:** r constante; logístico: recursos limitados a K.
- **Falla:** ambientes variables, depredación, retardos temporales.
- **Ejemplo:** ver código — logístico r=0.5, K=1000 satura en ~1000; mitad en t≈9.
- **Verificación:** contar población en el tiempo (conteo celular, censo) y ajustar r, K.

#### Modelo SIR (epidemias)
$$\frac{dS}{dt}=-\beta\frac{SI}{N},\quad \frac{dI}{dt}=\beta\frac{SI}{N}-\gamma I,\quad \frac{dR}{dt}=\gamma I, \quad R_0=\frac{\beta}{\gamma}$$
- **Predice:** curva de infectados; si R₀>1 hay epidemia.
- **Supuestos:** mezcla homogénea, población cerrada, inmunidad tras recuperar.
- **Falla:** heterogeneidad, comportamiento cambiante, variantes.
- **Ejemplo:** ver código — β=0.4, γ=0.1 → R₀=4, pico ~404 infectados en día 27.
- **Verificación:** ajustar a datos reales de casos; validar el pico y su fecha fuera de muestra.

#### Cinética de Michaelis-Menten (enzimas)
$$v = \frac{V_{max}[S]}{K_m + [S]}$$
- **Símbolos:** v = velocidad; V_max = velocidad máxima; K_m = constante de Michaelis; [S] = sustrato.
- **Predice:** velocidad enzimática según sustrato.
- **Supuestos:** estado cuasi-estacionario, una enzima-un sustrato.
- **Falla:** inhibición, cooperatividad (usar Hill), múltiples sustratos.
- **Ejemplo:** V_max=10, K_m=2, [S]=2 → v = 10·2/(2+2) = 5 (mitad de V_max).
- **Verificación:** medir v a varias [S]; linealizar (Lineweaver-Burk) o ajuste no lineal.

#### Lotka-Volterra (depredador-presa)
$$\frac{dx}{dt}=\alpha x - \beta xy, \qquad \frac{dy}{dt}=\delta xy - \gamma y$$
- **Predice:** oscilaciones acopladas de presa (x) y depredador (y).
- **Supuestos:** sin límite de recursos para la presa, respuesta lineal.
- **Falla:** capacidad de carga, refugios, múltiples especies; las órbitas son estructuralmente inestables.
- **Ejemplo:** ciclos periódicos desfasados: el depredador sigue a la presa con retardo.
- **Verificación:** series de campo (linces/liebres); comparar periodo y amplitud.

#### Hardy-Weinberg (genética)
$$p^2 + 2pq + q^2 = 1, \qquad p+q=1$$
- **Predice:** frecuencias genotípicas esperadas si NO hay evolución.
- **Supuestos:** población grande, apareamiento aleatorio, sin selección/mutación/migración.
- **Falla:** justamente cuando hay evolución → la desviación **detecta** selección/deriva.
- **Ejemplo:** p=0.7, q=0.3 → AA=0.49, Aa=0.42, aa=0.09.
- **Verificación:** genotipar la población; χ² entre observado y esperado.

### TRANSVERSALES

#### Entropía e información (Shannon)
$$H = -\sum_i p_i \log_2 p_i$$
- **Predice:** incertidumbre/contenido de información; cota de compresión.
- **Supuestos:** distribución p conocida; eventos bien definidos.
- **Falla:** dependencias no capturadas, estimación con pocos datos (sesgo).
- **Ejemplo:** moneda justa H=1 bit; dado justo H=log₂6≈2.585 bits.
- **Verificación:** medir frecuencias y comparar la compresibilidad real con H.

#### Análisis de sensibilidad
$$S_i = \frac{\partial Y/Y}{\partial X_i/X_i} \quad(\text{sensibilidad relativa})$$
- **Predice:** qué variables de entrada dominan la salida (dónde invertir en medición).
- **Métodos:** derivadas locales, Sobol (global), one-at-a-time.
- **Falla:** local ignora interacciones; usar métodos globales si el modelo es no lineal.
- **Ejemplo:** ver código — en Y=ab+c², la variable **c** domina (S≈0.17 vs 0.019).
- **Verificación:** perturbar entradas reales y medir el cambio en la salida.

#### Propagación de errores
$$\sigma_f^2 \approx \sum_i \left(\frac{\partial f}{\partial x_i}\right)^2 \sigma_{x_i}^2$$
- **Predice:** la incertidumbre de un resultado a partir de la de sus entradas.
- **Supuestos:** errores pequeños, independientes; linealización válida.
- **Falla:** errores grandes, variables correlacionadas (falta el término cruzado).
- **Ejemplo:** f=xy, x=10±0.1, y=5±0.2 → σ_f ≈ √((5·0.1)²+(10·0.2)²) ≈ 2.06.
- **Verificación:** Monte Carlo: propagar por muestreo y comparar con la fórmula.

#### Teoría del caos y exponentes de Lyapunov
$$|\delta(t)| \approx |\delta_0|\,e^{\lambda t}, \qquad \lambda>0 \Rightarrow \text{caos}$$
- **Símbolos:** λ = exponente de Lyapunov; δ = separación de trayectorias vecinas.
- **Predice:** el **horizonte de predictibilidad**: t_max ≈ (1/λ)·ln(precisión_deseada/error_inicial).
- **Supuestos:** sistema determinista; λ estimado de datos/modelo.
- **Falla — clave del manual:** si λ>0, un error inicial minúsculo se amplifica
  exponencialmente. **Determinista ≠ predecible.** Ejemplo: el clima (λ>0) es
  impredecible más allá de ~2 semanas pese a leyes conocidas.
- **Ejemplo:** λ=0.9/día, error inicial 10⁻⁶ → predecible ~15 días antes de perder toda precisión.
- **Verificación:** medir la divergencia de trayectorias cercanas; la pendiente de ln|δ| vs t es λ.

---

## 5. Procedimiento universal de modelado (12 pasos)

1. **Definir el suceso** — qué exactamente se predice, en qué unidades, horizonte.
2. **Identificar variables** — cuáles son causa, cuáles proxy, cuáles ruido.
3. **Formular hipótesis** — relación esperada (mecanicista y/o empírica).
4. **Recopilar y limpiar datos** — outliers, faltantes, unidades, fugas de información.
5. **Seleccionar un modelo** — el más simple que capture la estructura (navaja de Occam).
6. **Estimar parámetros** — mínimos cuadrados, máxima verosimilitud, bayesiano.
7. **Entrenar** — solo con datos de entrenamiento.
8. **Validar con datos independientes** — test hold-out o validación temporal.
9. **Cuantificar la incertidumbre** — IC, intervalos de predicción, posteriores.
10. **Comparar con alternativas** — incluido un baseline ingenuo; usar AIC/BIC/validación cruzada.
11. **Actualizar con nueva evidencia** — reentrenar, Bayes secuencial.
12. **Detectar correlaciones falsas y variables ocultas** — confusores, causalidad, pruebas de robustez.

---

## 6. Código funcional

Todo implementado, ejecutado y validado en **`manual_prediccion.py`** (Python + numpy,
scipy, scikit-learn). Cubre: regresión lineal y logística, inferencia bayesiana,
Monte Carlo, EDO, SIR, crecimiento logístico, Arrhenius, cadena de Markov, filtro de
Kalman, series temporales, MAE/RMSE/R², intervalos de confianza y análisis de
sensibilidad. Cada función valida entradas, maneja errores, separa train/test cuando
aplica y **nunca afirma certeza absoluta**.

```bash
pip install numpy scipy scikit-learn
python manual_prediccion.py
```

Resultados verificados (extracto real de la ejecución): Monte Carlo π=3.141±0.004;
Kalman reduce RMSE 0.33→0.18; SIR R₀=4, pico día 27; **serie temporal AR(1): el
pronóstico multi-paso da R²=−0.08 — ilustra que predecir a futuro pierde poder**.

---

## 7. Distinciones clave (no confundir)

| Concepto | Qué es | Pregunta que responde |
|---|---|---|
| **Predicción** | Estimar un valor futuro/desconocido | ¿Qué pasará? |
| **Explicación** | Entender el mecanismo | ¿Por qué pasa? |
| **Correlación** | Asociación estadística | ¿Se mueven juntas? (no implica causa) |
| **Causalidad** | X *provoca* Y | ¿Si intervengo X, cambia Y? |
| **Simulación** | Generar escenarios desde un modelo | ¿Qué pasaría si…? |
| **Pronóstico** | Predicción con horizonte temporal e incertidumbre | ¿Qué valor y con qué margen mañana? |
| **Clasificación** | Asignar a categorías | ¿A qué clase pertenece? |
| **Estimación** | Inferir un parámetro | ¿Cuánto vale θ? |
| **Incertidumbre** | Rango de credibilidad del resultado | ¿Cuánto podría equivocarme? |

**Regla de oro:** correlación ≠ causalidad. Solo un experimento controlado o
inferencia causal (variables instrumentales, DAGs, contrafactuales) justifica actuar.

---

## 8. Por qué la predicción es probabilística

Una predicción honesta **no** es "Y = 42". Es una distribución:

$$P(Y \mid X, D, M)$$

- Depende de **X** (lo observado), **D** (los datos que entrenaron), **M** (el modelo elegido).
- Se comunica como **probabilidad** (70% de lluvia), **rango** (42 ± 5) o **escenarios**
  (optimista/base/pesimista).
- Un número sin incertidumbre es una afirmación no verificable — y peligrosa.

---

## 9. Arquitectura computacional integrada

```
[Sensores/IoT] ─┐
[APIs externas] ─┼─▶ [Ingesta] ─▶ [Limpieza/validación (pandera)] ─▶ [Almacén: SQL/DuckDB/Parquet]
[Bases de datos]─┘                                                        │
                                                                          ▼
     ┌──────────────────────────────────────────────────────────────────────────┐
     │  MOTOR DE MODELOS                                                          │
     │  • Modelo físico (EDO/EDP, CFD)   • Modelo estadístico (regresión, series) │
     │  • Machine learning (sklearn)     • Simulación (Monte Carlo)               │
     └──────────────────────────────────────────────────────────────────────────┘
                                                                          │
        ┌────────────────┬───────────────────┬────────────────┬──────────┘
        ▼                ▼                   ▼                ▼
   [Visualización]  [Alertas/umbral]   [Cuantificación     [Reentrenamiento
    (Plotly)         (si p>umbral)      de incertidumbre]    automático (cron/MLOps)]
```

- **Ingesta/limpieza:** validación de esquema (pandera/pydantic), control de calidad.
- **Modelos híbridos:** el físico da estructura; el ML corrige el residuo.
- **Incertidumbre siempre:** bandas de predicción, no solo el punto.
- **Actualización:** detección de *drift* → reentrenar; versionado de modelos.

---

## 10. Pseudocódigo universal (completado)

```python
def predecir_suceso(datos, ajustar_fn, predecir_fn):
    X, y = validar(datos)              # tipos, tamaños, finitos
    X, y = limpiar(X, y)               # NaN/outliers/unidades
    if len(y) < 10:
        return "No puede realizarse una predicción confiable con los datos disponibles."
    Xtr, Xte, ytr, yte = separar_train_test(X, y)   # nunca entrenar con test
    modelo   = ajustar_fn(Xtr, ytr)    # estimar parámetros
    pred     = predecir_fn(modelo, Xte)
    error    = metricas(yte, pred)     # MAE/RMSE/R² fuera de muestra
    incert   = 1.96 * residuos.std()   # banda ~95%
    return {"prediccion": pred, "incertidumbre": incert, "error_test": error,
            "advertencia": "Estimación probabilística, no certeza."}
```

Implementación real y ejecutable en `manual_prediccion.py`.

---

## 11. Límites de "todo es predecible"

La premisa es un **marco**, no una ley. En la práctica falla por:

| Límite | Por qué rompe la predicción |
|---|---|
| **Falta de datos** | Sin muestra suficiente no hay estimación estable. |
| **Variables no observadas** | Confusores ocultos sesgan el modelo. |
| **Ruido** | Parte de la señal es irreducible (error de medición + azar). |
| **Aleatoriedad** | Procesos i.i.d. sin memoria (lotería justa, decaimiento cuántico): P(Y) es fija; el pasado NO informa el futuro. |
| **Caos (λ>0)** | Error inicial se amplifica exponencialmente; horizonte finito aun con leyes exactas. |
| **Cambios estructurales** | El sistema cambia sus reglas (drift, cisnes negros). |
| **Errores humanos** | Sesgos, fugas de datos, mala especificación. |
| **Límites de cómputo** | Algunos problemas son intratables (NP-difíciles, alta dimensión). |
| **Modelos incompletos** | "Todos los modelos están equivocados; algunos son útiles" (Box). |
| **Problemas de medición** | No se puede medir con precisión arbitraria (incl. límite cuántico). |
| **Sucesos únicos** | Sin antecedentes suficientes, no hay base estadística. |

> **Caso frontera honesto — sorteos justos:** un proceso genuinamente aleatorio e
> independiente (una lotería auditada, un decaimiento radiactivo) tiene
> P(Y|X,D,M) = P(Y) constante: los datos históricos **no reducen la incertidumbre**
> del próximo resultado. No es un fallo del modelo — es que la información
> predictiva es cero. Aquí la respuesta correcta es reconocer la no-identificabilidad.

---

## 12. Conclusión

Bajo la premisa hipotética adoptada, **cualquier suceso puede tratarse como
potencialmente modelable y predecible**: se define, se instrumenta, se formula una
ley o un modelo empírico, se estiman parámetros y se cuantifica la incertidumbre.
Este manual da las herramientas —de Newton a Navier-Stokes, de Bayes a Kalman, de
Arrhenius a SIR— con su código ejecutable.

**Pero la premisa tiene un techo real e ineludible.** En la práctica, la precisión
está siempre limitada por los **datos**, los **supuestos**, la **medición**, el
**modelo** y la **incertidumbre** intrínseca. Hay tres murallas que ninguna
sofisticación derriba: la **aleatoriedad pura** (información predictiva nula), el
**caos** (horizonte finito) y los **límites de medición**. Por eso toda predicción
honesta se escribe como P(Y|X,D,M) —una distribución con su margen— y no como una
certeza. Predecir bien no es eliminar la incertidumbre: es **medirla con
honestidad** y actuar en consecuencia.
