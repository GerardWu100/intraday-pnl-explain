---
title: "Prévoir la variance réalisée du lendemain à partir de barres d'une minute"
description: "Une expérience walk-forward hors ligne qui transforme les rendements logarithmiques intrajournaliers en prévisions de variance réalisée, sans cacher les limites d'un échantillon de dix séances."
date: 2026-07-13
image: images/cover-realized-variance-forecast.png
categories: ["Quantitative Research", "Risk Management"]
---

# Prévoir la variance réalisée du lendemain à partir de barres d'une minute

Une prévision de volatilité peut être fausse avant même que le modèle ne voie les données. Il suffit d'attribuer un horodatage à la mauvaise séance, de laisser une fenêtre glissante atteindre le lendemain ou d'utiliser une séparation aléatoire entre apprentissage et test. Une régression banale peut alors paraître clairvoyante.

J'ai construit ce projet autour d'une question plus étroite : l'information disponible après la clôture permet-elle de mieux prévoir la variance réalisée du lendemain que deux règles naïves ? L'expérience fonctionne entièrement hors ligne à partir de fichiers parquet suivis dans le dépôt. Le chemin des données reste donc vérifiable, ce qui compte davantage ici que l'ajout d'un modèle supplémentaire.

Le jeu de démonstration contient 23 400 barres d'une minute pour AAPL, MSFT, NVDA, JPM, XOM et CVX. Il couvre dix séances régulières, du 24 mars au 6 avril 2026. C'est assez pour éprouver la chaîne de recherche. C'est beaucoup trop peu pour déterminer quelle prévision mérite d'être tradée.

## Des prix minute par minute à une cible quotidienne

Pour le titre $i$, la date de négociation $d$ et la barre intrajournalière $t$, $P_{i,d,t}$ désigne le prix observé. Le rendement logarithmique à une minute $r_{i,d,t}$ est le logarithme naturel du rapport entre deux prix consécutifs :

$$
r_{i,d,t}
= \log\left(\frac{P_{i,d,t}}{P_{i,d,t-1}}\right).
$$

La variance réalisée quotidienne $RV_{i,d}$ est la somme de ces rendements au carré pendant la séance régulière :

$$
RV_{i,d}
= \sum_{t=1}^{T_d} r_{i,d,t}^{2},
$$

où $T_d$ est le nombre de rendements intrajournaliers utilisables à la date $d$. Chaque séance suivie contient 390 prix, de 9 h 30 à 15 h 59, heure de New York. Elle produit donc 389 rendements. Les 60 observations titre-séance atteignent toutes ce nombre.

L'implémentation convertit d'abord les horodatages du temps universel coordonné vers `America/New_York`, attribue la date locale de la bourse, puis calcule les rendements dans chaque groupe titre-date. Le regroupement avant le décalage évite qu'un mouvement nocturne n'entre dans la mesure intrajournalière.

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

![Variance réalisée quotidienne pour six titres](images/01_realized_variance_by_symbol.png)

Les six panneaux montrent une variance très inégale, même sur dix séances. NVDA atteint l'observation la plus élevée, environ $5.89 \times 10^{-6}$, tandis que le minimum de l'échantillon vaut environ $9.45 \times 10^{-10}$. Le logarithme naturel de la variance comprime cet écart d'échelle et empêche la séance la plus volatile de dominer un ajustement quadratique en niveau.

## L'alignement qui tient le lendemain à l'écart des variables

Soit $\mathbf{x}_{i,d}$ le vecteur de variables connu après la clôture de la date $d$. La cible $y_{i,d+1}$ est le logarithme de la variance réalisée à la séance suivante :

$$
y_{i,d+1}=\log(RV_{i,d+1}),
\qquad
\widehat{y}_{i,d+1}=f(\mathbf{x}_{i,d}),
$$

où $f$ est la règle de prévision et $\widehat{y}_{i,d+1}$ sa prévision. Le vecteur contient cinq quantités :

- le logarithme de la variance réalisée du jour ;
- la moyenne et l'écart-type sur cinq jours du logarithme de la variance réalisée ;
- la variation absolue par rapport au logarithme de la variance réalisée de la veille ;
- la proportion des 389 rendements attendus qui sont présents dans la séance.

Le code source nomme la première variable `lag_1_log_rv`. Son sens économique est plus clair que son nom : la variance réalisée du jour $d$ est connue après la clôture et sert à prévoir le jour $d+1$. Le décalage négatif par groupe crée la cible. Aucune variable explicative n'est déplacée vers l'avenir.

```python
frame["lag_1_log_rv"] = frame["log_rv"]
frame["lag_5_mean_log_rv"] = frame.groupby("symbol")["log_rv"].transform(
    lambda values: values.rolling(5).mean()
)
frame["target_date"] = frame.groupby("symbol")["date"].shift(-1)
frame["target_log_rv_next_day"] = frame.groupby("symbol")["log_rv"].shift(-1)
```

Les variables glissantes sur cinq jours consomment les quatre premières séances. Il reste cinq dates de variables et 30 lignes titre-date correctement alignées. La procédure walk-forward utilise les quatre premières dates pour l'apprentissage et la cinquième, le 3 avril, pour un test sur la variance réalisée du 6 avril. L'évaluation publiée porte donc sur une date et six actions.

## Trois prévisions et une séparation temporelle honnête

La prévision par persistance pose que la variance logarithmique de demain sera égale à celle d'aujourd'hui. La moyenne glissante reprend la moyenne des cinq derniers jours. La régression ridge combine les cinq variables standardisées.

Pour $n$ observations d'apprentissage, $y_j$ désigne le logarithme observé de la variance du lendemain, $\mathbf{x}_j$ le vecteur de variables standardisé, $\beta_0$ l'ordonnée à l'origine, $\boldsymbol{\beta}$ le vecteur de coefficients et $\lambda$ la force non négative de la pénalisation. La régression ridge estime les coefficients en résolvant

$$
(\widehat{\beta}_0,\widehat{\boldsymbol{\beta}})
= \arg\min_{\beta_0,\boldsymbol{\beta}}
\left[
\sum_{j=1}^{n}
\left(y_j-\beta_0-\mathbf{x}_j^{\mathsf T}\boldsymbol{\beta}\right)^2
+\lambda\sum_{k=1}^{5}\beta_k^2
\right].
$$

La première somme mesure l'erreur quadratique d'apprentissage. La seconde ramène les cinq coefficients de pente vers zéro. Le projet fixe $\lambda=1$. À chaque étape walk-forward, les moyennes et écarts-types des variables proviennent uniquement des dates antérieures. Une standardisation sur l'échantillon complet laisserait la distribution du test contaminer l'apprentissage.

L'évaluation retient la racine de l'erreur quadratique moyenne, ou RMSE (*root mean squared error*), l'erreur absolue moyenne, ou MAE (*mean absolute error*), et le coefficient de détermination hors échantillon $R^2_{\mathrm{oos}}$. Pour $m$ prévisions, des valeurs observées $y_j$, des prévisions $\widehat y_j$ et la moyenne $\bar y$ de l'échantillon d'évaluation, les calculs sont

$$
\mathrm{RMSE}
=\sqrt{\frac{1}{m}\sum_{j=1}^{m}(y_j-\widehat y_j)^2},
\qquad
\mathrm{MAE}
=\frac{1}{m}\sum_{j=1}^{m}|y_j-\widehat y_j|,
$$

$$
R^2_{\mathrm{oos}}
=1-
\frac{\sum_{j=1}^{m}(y_j-\widehat y_j)^2}
{\sum_{j=1}^{m}(y_j-\bar y)^2}.
$$

Une RMSE et une MAE plus faibles sont préférables. Un $R^2_{\mathrm{oos}}$ négatif indique que la prévision commet davantage d'erreur quadratique que l'attribution de la même moyenne d'évaluation à chaque observation.

## Ce qui s'est passé à la date de test

| Modèle | RMSE | MAE | $R^2_{\mathrm{oos}}$ |
|---|---:|---:|---:|
| Persistance | 1.300 | 1.209 | -1.253 |
| Moyenne sur cinq jours | 1.604 | 1.427 | -2.429 |
| Ridge | 1.190 | 1.089 | -0.888 |

![Comparaison des erreurs de prévision](images/02_model_error_comparison.png)

La régression ridge affiche la plus petite erreur à cette date. Sa RMSE est inférieure de 8,45 % à celle de la persistance, tandis que la moyenne sur cinq jours termine dernière. Tous les $R^2_{\mathrm{oos}}$ sont négatifs. La conclusion reste modeste : la régression pénalisée perd moins nettement face à la moyenne transversale que les deux règles naïves.

![Variance réalisée logarithmique observée et prévue](images/03_forecast_cross_section.png)

La coupe transversale révèle les erreurs masquées par un seul chiffre agrégé. Le 6 avril, le logarithme de la variance observée va de -16.86 pour JPM à -14.41 pour CVX. Ridge reproduit certains écarts relatifs, notamment entre NVDA et JPM, mais surestime la variance de chaque titre. La persistance se rapproche davantage des deux valeurs énergétiques et reste trop élevée pour plusieurs autres actions. Sur une seule date, ce profil peut appartenir à un régime ponctuel plutôt qu'à un comportement reproductible.

Le fichier des coefficients fournit un autre diagnostic utile. `bar_completeness` vaut exactement 1 pour chaque observation. Sa valeur standardisée et son coefficient ajusté sont donc nuls. Un champ de qualité des données peut être pertinent en production sans apporter la moindre information dans un jeu de démonstration propre.

## Ce qu'il faudrait avant de croire au classement

L'exécution actuelle valide la plomberie : attribution à la séance locale, calcul des rendements, alignement des fenêtres glissantes, standardisation limitée à l'apprentissage, ajustement des modèles et production des fichiers. Tout fonctionne depuis un clone, sans base de données active. Elle ne valide pas un avantage prédictif.

Une comparaison crédible exige beaucoup plus de dates de test, pendant des marchés calmes comme agités. Je conserverais les six actions d'une même date dans le même bloc lors des séparations, je publierais les erreurs par date et par titre, puis j'estimerais l'incertitude autour de l'écart entre ridge et persistance. L'univers devrait aussi couvrir davantage de secteurs. Six grandes actions américaines partagent des chocs de volatilité de marché ; considérer les 30 lignes d'apprentissage comme indépendantes exagère donc la taille effective de l'échantillon.

La cible a elle aussi ses frontières. Elle exclut les rendements nocturnes et utilise des prix à une minute sans correction du bruit de microstructure. Une étude plus longue devrait comparer plusieurs fréquences d'échantillonnage, examiner les sauts et préciser si la décision de risque concerne la séance au comptant, la journée complète de clôture à clôture, ou les deux.

Voilà ce que cette petite expérience permet de retenir. Le classement des modèles reste provisoire. La chaîne de recherche rend la prochaine expérience, plus longue, bien plus difficile à tromper.
