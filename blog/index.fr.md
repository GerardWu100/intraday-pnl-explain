---
title: "Prévoir la variance réalisée de demain sans emprunter ses données"
description: "Une expérience walk-forward après clôture, avec unités explicites, purge des cibles indisponibles, comparaison réalisable à la persistance et limites assumées d'un échantillon de dix séances."
date: 2026-07-13
image: images/cover-realized-variance-forecast.png
categories: ["Quantitative Research", "Risk Management"]
---

# Prévoir la variance réalisée de demain sans emprunter ses données

Un modèle de volatilité peut sembler excellent à cause d'un seul mauvais horodatage. Une fenêtre glissante atteint la séance cible, la standardisation apprend sur le test, ou le score compare le modèle à une référence impossible à connaître au moment de la prévision. Aucun de ces échecs n'exige un code compliqué.

Ce projet pose une question volontairement étroite : après la clôture de la séance régulière, la variance réalisée du jour et son historique récent prévoient-ils mieux la séance suivante que la persistance ? Le jeu suivi contient 23 400 lignes d'une minute pour AAPL, MSFT, NVDA, JPM, XOM et CVX, sur dix séances entre le 24 mars et le 6 avril 2026.

Cet échantillon suffit pour tester une chaîne de recherche. Il ne peut pas établir un résultat de trading. Le manifeste ne précise ni le fournisseur, ni la méthode d'extraction, ni si les prix sont synthétiques. Je traite donc ces barres comme un jeu reproductible destiné au développement, et non comme une preuve empirique sur les marchés.

## L'horloge de la prévision

L'origine de la prévision se situe après la clôture au comptant de la date $d$. Tous les prix de la séance régulière de $d$ sont alors connus. La prévision vise la prochaine séance observée, notée $d+1$ par simplicité.

| Étape | Information la plus récente autorisée | Taille dans ce jeu |
|---|---|---:|
| Barres brutes | 9 h 30 à 15 h 59, heure de New York, à la date $d$ | 390 prix par titre-séance |
| Mesure quotidienne | Rendements formés uniquement dans le titre et la date $d$ | 389 rendements par titre-séance |
| Variables | Variance réalisée courante et historique jusqu'à $d$ | 30 lignes titre-date après amorçage |
| Cibles d'apprentissage | Dates cibles au plus tard à l'origine $d$ | 24 lignes sur quatre dates |
| Prévisions hors échantillon | Variables du 3 avril, cible du 6 avril | 6 lignes sur une date |

Les six lignes hors échantillon forment une coupe transversale, pas six périodes de test indépendantes. Un choc de volatilité de marché peut toucher les six actions en même temps. La taille effective du test temporel est égale à un.

## Des prix minute par minute à la variance réalisée

Pour le titre $i$, la date de négociation $d$ et l'observation intrajournalière $t$, $P_{i,d,t}$ désigne le prix en dollars. Le rendement logarithmique à une minute $r_{i,d,t}$ vaut

$$
r_{i,d,t}
= \log\left(\frac{P_{i,d,t}}{P_{i,d,t-1}}\right).
$$

Un rendement logarithmique est sans dimension. Son carré s'exprime en rendement décimal au carré. La variance réalisée quotidienne $RV_{i,d}$ additionne ces carrés pendant la séance régulière :

$$
RV_{i,d}
= \sum_{t=1}^{T_d} r_{i,d,t}^{2},
$$

où $T_d$ est le nombre de rendements utilisables dans la séance. Le chargeur convertit d'abord les horodatages du temps universel coordonné (UTC) vers `America/New_York`. Le décalage qui produit les rendements s'effectue ensuite dans chaque groupe titre-date, ce qui exclut le mouvement nocturne.

```python
group_columns = ["symbol", "date"]
prepared_bars["log_return"] = prepared_bars.groupby(group_columns)["price"].transform(
    lambda values: np.log(values / values.shift(1))
)

realized_daily = prepared_bars.groupby(group_columns, as_index=False).agg(
    realized_variance=("log_return", _sum_squared_log_returns),
    bar_count=("log_return", _non_missing_log_return_count),
)
```

Chacun des 60 groupes titre-séance contient 390 prix et 389 rendements. Ce comptage établit la complétude interne, pas la provenance des données ni le sens exact du prix. Le contrat brut nomme le champ `price`, sans indiquer s'il s'agit d'une clôture de barre, d'un milieu de fourchette ou d'une transaction.

![Variance réalisée quotidienne pour six titres](images/01_realized_variance_by_symbol.png)

Les panneaux montrent une large dispersion. NVDA atteint environ $5.89 \times 10^{-6}$, tandis que le minimum de l'échantillon vaut environ $9.45 \times 10^{-10}$. Le modèle utilise le logarithme de la variance afin que l'observation la plus élevée ne domine pas une régression quadratique en niveau.

La cible reste exprimée sur une séance. Si $A=252$ est le nombre supposé de séances par an, la variance et la volatilité annualisées seraient

$$
RV^{\mathrm{ann}}_{i,d}=A\,RV_{i,d},
\qquad
\sigma^{\mathrm{ann}}_{i,d}=\sqrt{A\,RV_{i,d}}.
$$

Cette annualisation suppose une variance quotidienne comparable sur l'année. Dans l'espace logarithmique, elle ajoute seulement une constante :

$$
\log(RV^{\mathrm{ann}}_{i,d})=\log(A)+\log(RV_{i,d}).
$$

Ajouter la même constante aux observations et aux prévisions ne change aucune erreur. L'annualisation ne modifierait donc pas le classement des modèles.

## Alignement des variables et de la cible

On note $y_{i,d+1}$ le logarithme de la variance réalisée à la séance suivante et $\mathbf{x}_{i,d}$ le vecteur de variables disponible après la clôture de $d$ :

$$
y_{i,d+1}=\log(RV_{i,d+1}),
\qquad
\widehat y_{i,d+1}=f(\mathbf{x}_{i,d}).
$$

Les cinq variables sont le logarithme de la variance courante, sa moyenne mobile sur cinq séances, son écart-type mobile sur cinq séances, sa variation absolue sur une séance et la proportion des rendements attendus présents. Leurs nouveaux noms indiquent directement leur temporalité. Les anciens qualifiaient la variance courante de `lag_1` et une variation absolue de variance de `range_proxy`, deux formulations qui invitaient une mauvaise lecture financière.

```python
frame["current_log_rv"] = frame["log_rv"]
frame["trailing_5_mean_log_rv"] = frame.groupby("symbol")["log_rv"].transform(
    lambda values: values.rolling(5).mean()
)
frame["target_date"] = frame.groupby("symbol")["date"].shift(-1)
frame["target_log_rv_next_day"] = frame.groupby("symbol")["log_rv"].shift(-1)
```

La fenêtre mobile consomme les quatre premières séances. Il reste cinq dates de variables. Pour la prévision hors échantillon formée le 3 avril, les dates des variables d'apprentissage se terminent le 2 avril et leurs cibles le 3 avril.

Un découpage fondé sur la date ne suffit pas lorsqu'un titre présente des séances manquantes. Une ligne ancienne peut pointer vers une cible postérieure à l'origine courante. L'ajustement walk-forward impose désormais les deux conditions suivantes :

```python
train_frame = frame[
    (frame["feature_date"] < test_date) & (frame["target_date"] <= test_date)
].copy()
```

La seconde condition purge les cibles encore indisponibles. Après la clôture du 3 avril, la variance réalisée du 3 avril est observable et peut servir de cible d'apprentissage. Celle du 6 avril ne l'est pas et reste la cible du test.

## Trois règles de prévision

La persistance prévoit que le logarithme de la variance de la prochaine séance sera égal à celui de la séance courante. La moyenne mobile reprend la moyenne des cinq séances. Ces deux règles sont réalisables à l'origine de la prévision.

La régression ridge combine les cinq variables. Pour $n$ lignes d'apprentissage, $y_j$ désigne la cible, $\mathbf{z}_j$ le vecteur standardisé, $\beta_0$ l'ordonnée à l'origine, $\boldsymbol{\beta}$ les cinq pentes et $\lambda \ge 0$ la pénalisation. Les paramètres ajustés résolvent

$$
(\widehat{\beta}_0,\widehat{\boldsymbol{\beta}})
= \arg\min_{\beta_0,\boldsymbol{\beta}}
\left[
\sum_{j=1}^{n}
\left(y_j-\beta_0-\mathbf{z}_j^{\mathsf T}\boldsymbol{\beta}\right)^2
+\lambda\sum_{k=1}^{5}\beta_k^2
\right].
$$

Pour la variable $k$, le code calcule la moyenne d'apprentissage $\mu_k$ et l'écart-type d'apprentissage $s_k$, puis pose

$$
z_{j,k}=\frac{x_{j,k}-\mu_k}{s_k}.
$$

La ligne de test utilise les mêmes $\mu_k$ et $s_k$. Elle ne contribue jamais à leur estimation. Le projet fixe $\lambda=1$ sans optimisation, car quatre dates d'apprentissage ne permettent pas une recherche d'hyperparamètres crédible.

Les 24 lignes d'apprentissage mélangent six titres. Elles ne constituent pas 24 historiques de volatilité indépendants. Les cinq variables sont aussi fortement liées par construction. L'unique vecteur de coefficients reste donc instable. Ses pentes standardisées vont de -0,713 pour la variance logarithmique courante à 0,942 pour sa variabilité mobile, mais leurs signes ne doivent pas être interprétés comme des estimations économiques.

## Un score réalisable face à la persistance

Pour $m$ prévisions hors échantillon, $e_{M,j}=y_j-\widehat y_{M,j}$ désigne l'erreur de variance logarithmique du modèle $M$. La racine de l'erreur quadratique moyenne, ou RMSE (*root mean squared error*), et l'erreur absolue moyenne, ou MAE (*mean absolute error*), valent

$$
\mathrm{RMSE}_M
=\sqrt{\frac{1}{m}\sum_{j=1}^{m}e_{M,j}^2},
\qquad
\mathrm{MAE}_M
=\frac{1}{m}\sum_{j=1}^{m}|e_{M,j}|.
$$

On note $SSE_M=\sum_j e_{M,j}^2$ la somme des erreurs au carré du modèle et $SSE_P$ celle de la persistance sur les mêmes lignes. Le score relatif à la persistance est

$$
\mathrm{Skill}_{M\mid P}
=1-\frac{SSE_M}{SSE_P}.
$$

La persistance vaut zéro par construction. Un score positif l'améliore. Un score négatif fait moins bien. Cette mesure remplace l'ancienne statistique, dont le dénominateur utilisait la moyenne transversale des résultats hors échantillon. Cette moyenne n'est connue qu'après leur réalisation et ne constituait donc pas une référence de prévision réalisable.

| Modèle | RMSE | MAE | Score face à la persistance |
|---|---:|---:|---:|
| Persistance | 1.300 | 1.209 | 0.000 |
| Moyenne sur cinq séances | 1.604 | 1.427 | -0.522 |
| Ridge | 1.190 | 1.089 | 0.162 |

![Comparaison des erreurs de prévision](images/02_model_error_comparison.png)

Ridge réduit l'erreur quadratique de 16,19 % face à la persistance sur cette seule coupe transversale. La moyenne sur cinq séances l'augmente de 52,20 %. La RMSE de ridge est inférieure de 8,45 % à celle de la persistance, car la RMSE prend la racine carrée de l'erreur quadratique moyenne.

Les erreurs absolues restent importantes sur le plan économique. Puisque la cible est logarithmique, une erreur absolue $a$ correspond à un rapport multiplicatif de variance égal à $\exp(a)$. L'exponentielle de la MAE de ridge donne $\exp(1.089)\approx2.97$. C'est un résumé d'échelle, pas un intervalle de confiance, mais il rappelle que « meilleur » ne signifie pas « précis ».

![Variance réalisée logarithmique observée et prévue](images/03_forecast_cross_section.png)

Le 6 avril, le logarithme observé de la variance va de -16,86 pour JPM à -14,41 pour CVX. Ridge surestime quatre titres, pas les six : il sous-estime CVX et XOM. La courbe reproduit une partie de l'ordre transversal, mais une date ne permet pas de distinguer une relation réutilisable d'une coïncidence.

## Pourquoi ce modèle n'est pas un modèle de P&L

La variance réalisée est une donnée de risque, pas un profit et perte (P&L). Une prévision de variance peut éclairer la couverture d'options, le ciblage de volatilité, les marges ou les scénarios. Chaque application exige toutefois une couche supplémentaire qui transforme la variance en positions et en flux financiers.

À titre d'intuition seulement, considérons une option couverte en delta sur un horizon court. $S$ désigne le prix spot, $\Gamma$ le gamma de l'option en devise par prix au carré, $RV$ la variance réalisée sur l'horizon et $v_{\mathrm{imp}}$ la variance implicite de même horizon payée à l'origine. En maintenant le spot et le gamma constants, on obtient l'approximation grossière

$$
\mathrm{P\&L}_{\Delta\text{-hedged}}
\approx \frac{1}{2}\Gamma S^2\left(RV-v_{\mathrm{imp}}\right).
$$

Le projet ne teste pas cette équation. Son $RV$ exclut les rendements nocturnes alors qu'une option vit en temps continu. Le gamma varie avec le spot et le temps, les coûts de transaction comptent, et une prévision de $\log(RV)$ ne fournit pas automatiquement la moyenne de la variance. Comme l'exponentielle est convexe, $\exp(\widehat y)$ estime une médiane conditionnelle sous les hypothèses usuelles sur l'erreur logarithmique, et non la moyenne conditionnelle, sauf correction du biais.

## Ce que démontre l'expérience

L'exécution reproductible vérifie l'affectation des séances, le calcul des rendements intrajournaliers, la temporalité des variables, la disponibilité des cibles, la standardisation limitée à l'apprentissage, les trois règles de prévision et la production des fichiers. Les tests comprennent un calcul manuel de variance réalisée, un cas de purge avec date manquante et un calcul manuel du score face à la persistance.

Elle ne démontre aucun pouvoir prédictif. Une étude sérieuse demanderait plusieurs années de données documentées, de nombreuses dates walk-forward couvrant différents régimes de volatilité, des erreurs par date et par titre, une mesure d'incertitude sur les différences de pertes appariées et un traitement de la variance nocturne adapté à la décision. Il faudrait aussi comparer plusieurs fréquences, car les prix à une minute peuvent contenir du bruit de microstructure.

Le résultat le plus défendable concerne donc la méthode : sur six prévisions d'une seule date, ridge affiche un score positif face à la persistance, mais l'échantillon n'a pratiquement aucun pouvoir pour séparer un avantage durable du bruit.

## Références

- Andersen, Bollerslev, Diebold et Labys, [« The Distribution of Realized Exchange Rate Volatility »](https://doi.org/10.1198/016214501750332965), *Journal of the American Statistical Association*, 2001.
- Corsi, [« A Simple Approximate Long-Memory Model of Realized Volatility »](https://doi.org/10.1093/jjfinec/nbp001), *Journal of Financial Econometrics*, 2009.
- Hansen et Lunde, [« Realized Variance and Market Microstructure Noise »](https://doi.org/10.1198/073500106000000071), *Journal of Business & Economic Statistics*, 2006.
- Hoerl et Kennard, [« Ridge Regression: Biased Estimation for Nonorthogonal Problems »](https://doi.org/10.1080/00401706.1970.10488634), *Technometrics*, 1970.
- Campbell et Thompson, [« Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average? »](https://doi.org/10.1093/rfs/hhm055), *Review of Financial Studies*, 2008. La logique de comparaison à une référence réalisable motive ici le score relatif à la persistance.
