# Arkitektur for solprognosprojekt

## 1. Mal

Vi ska bygga ett prognossystem for solproduktion i hemmet baserat pa historisk effektdata (`power_w`) med 15-minuters upplosning och externa vaderdata. Systemet ska kunna trana och jamfora flera modeller for olika tidshorisonter och koras i RunPod.

Primara mal:

- Prognoser for flera horisonter, till exempel:
  - `15 min`
  - `1 timme`
  - `3 timmar`
  - `6 timmar`
  - `24 timmar`
- Jambarforsok mellan enkla och mer avancerade modeller
- Tydlig separation mellan dataforberedelse, feature engineering, traning, evaluering och inferens
- Stabil korning i RunPod med reproducerbara treningjobb och sparade artefakter

## 2. Databild

Nuvarande produktionsdataset:

- Fil: `solar_dataset_2021_2024.csv`
- Kolumner: `datetime`, `power_w`
- Period: `2021-01-01 00:00` till `2024-04-02 12:30`
- Upplosning: 15 minuter
- Observation: tidsserien innehaller nagra luckor och maste resamplas till ett komplett 15-minutersindex

Kompletterande datakalla:

- Vaderdata pa samma eller finare tidsupplosning
- Minimikrav: molnighet, temperatur, globalstralning eller solinstralning om tillganglig
- Ovrigt onskvart: luftfuktighet, vindhastighet, nederbord, solhojd, solazimut

Implikation:

- Problemet ar ett multivariat forecast-problem
- Kalenderfeatures blir viktiga eftersom solproduktion styrs starkt av dygn och sasong
- Vaderfeatures blir centrala, sarskilt for horisonter fran `1h` och uppat
- Nattvarden med `0 W` ar inte brus utan en viktig del av signalen

## 3. Datakallor och kontrakt

Vi bor behandla produktion och vader som separata kallsystem som sammanfogas i en kontrollerad pipeline.

### 3.1 Produktionsdata

- Primarnyckel: `datetime`
- Malsignal: `power_w`
- Upplosning: 15 minuter

### 3.2 Vaderdata

Vaderdatan bor delas upp i tva typer:

1. Historisk vaderdata for traning
2. Prognostiserad vaderdata for inferens

Minsta rekommenderade schema:

- `datetime`
- `temperature_c`
- `cloud_cover_pct`
- `ghi_wm2` eller annan stralningsvariabel
- `humidity_pct`
- `wind_speed_ms`
- `precipitation_mm`

Viktigt arkitekturbeslut:

- Vid traning ska modellen bara fa tillgang till vaderinformation som realistiskt skulle ha varit kand vid prognostillfallet
- For korta horisonter kan vi anvanda observerad vaderhistorik
- For langre horisonter maste vi designa for att ta emot vaderprognoser, inte framtida observerad vaderdata

## 4. Rekommenderad systemdesign

Vi bygger losningen i fem lager:

1. `Ingestion`
   - Las produktionsdata
   - Las vaderdata
   - Parsea datum
   - Sortera, deduplicera, resampla till 15-minutersintervall
   - Synka alla serier pa gemensamt tidsindex
   - Markera saknade tidssteg och datakvalitet
   - Spara standardiserade mellanlager i `parquet`

2. `Feature layer`
   - Skapa tidsfeatures
   - Skapa solgeometrifeatures
   - Skapa laggade observationer for produktion
   - Skapa laggade vaderfeatures
   - Skapa rullande statistik
   - Skapa vaderbaserade framtidsfeatures for de horisonter dar prognosvader finns
   - Skapa target per prognoshorisont

3. `Training layer`
   - Trana separata modeller per horisont
   - Halla isar kort-horisont och lang-horisont feature-set
   - Behall en gemensam pipeline for preprocessing och evaluering
   - Jamfor baseline, tradmodeller och neural modell

4. `Evaluation layer`
   - Walk-forward eller expanding-window-validering
   - Metriker per horisont
   - Jamforelse med och utan vaderfeatures
   - Felanalys per sasong, timme och produktionstopp

5. `Inference layer`
   - En funktion som tar senaste historik och kommande vaderprognos
   - Returnerar prognoser for alla horisonter
   - Laddar sparade artefakter fran RunPod-volym eller objektlagring

## 5. Modellstrategi for flera horisonter

Det finns tre vanliga satt att hantera flera horisonter:

1. En modell per horisont
2. En multi-output-modell som predikterar flera steg samtidigt
3. Direkta sekvensmodeller, till exempel LSTM eller TCN

Rekommenderad start:

- Fas 1: en separat modell per horisont
- Fas 2: lagg till en multi-output-modell for jamforelse
- Fas 3: testa en sekvensmodell om de enklare modellerna inte racker

Varfor en modell per horisont forst:

- Enklare att debugga
- Enklare att tolka
- Ofta stark baseline for tidsserier med begransad datamangd
- Enkelt att parallellisera i RunPod

Exempel pa target-definition:

- `y_t+1` for 15 min
- `y_t+4` for 1 timme
- `y_t+12` for 3 timmar
- `y_t+24` for 6 timmar
- `y_t+96` for 24 timmar

Praktisk uppdelning:

- Korta horisonter `15m` och `1h`: laggar + rolling + aktuell vaderhistorik
- Mellanhorisonter `3h` och `6h`: laggar + kalender + vaderprognos
- Lang horisont `24h`: kalender + sasong + solgeometri + vaderprognos blir extra viktigt

## 6. Features

### 6.1 Tidsfeatures

- `hour`
- `minute`
- `day_of_week`
- `day_of_year`
- `month`
- `week_of_year`
- `is_weekend`

For cykliska monster ska vi anvanda sinus/cosinus:

- `sin_hour`, `cos_hour`
- `sin_day_of_year`, `cos_day_of_year`

### 6.2 Solgeometrifeatures

Eftersom problemet galler solproduktion bor solens lage vara en huvudfeature, inte en senare optimering.

- `solar_elevation`
- `solar_azimuth`
- `is_daylight`
- `clear_sky_radiation` om vi kan berakna den

### 6.3 Laggade features

Vi skapar laggar pa flera skalor:

- Korta laggar: `1, 2, 3, 4` steg
- Dygnslaggar: `96` steg
- Veckolaggar: `96 * 7`

Exempel:

- `lag_1`
- `lag_4`
- `lag_96`
- `lag_192`
- `lag_672`

### 6.4 Rullande features

- Rullande medel for `1h`, `3h`, `6h`, `24h`
- Rullande max
- Rullande standardavvikelse

### 6.5 Vaderfeatures

Historiska vaderfeatures:

- `temp_lag_1`, `temp_lag_4`
- `cloud_cover_lag_1`, `cloud_cover_lag_4`
- `ghi_lag_1`, `ghi_lag_4`
- rullande medel och trend pa molnighet och stralning

Framtida vaderfeatures for inferens:

- `temp_forecast_t+1 ... t+96`
- `cloud_cover_forecast_t+1 ... t+96`
- `ghi_forecast_t+1 ... t+96`

For tradmodeller bor vi inte skicka in hela framtidskurvan okontrollerat. I stallet skapar vi horizon-specifika features, till exempel:

- for `1h`-modellen: vaderprognos vid `t+4`
- for `3h`-modellen: vaderprognos vid `t+12`
- for `24h`-modellen: vaderprognos vid `t+96`

### 6.6 Domanspecifika features

- interaktion mellan molnighet och solhojd
- interaktion mellan klarhimmelstralning och observerad effekt
- eventuellt installerad toppeffekt som normaliseringsfaktor

## 7. Modellstack

Vi bor bygga i lager, inte hoppa direkt till deep learning.

### Niva A: Baselines

- `Persistence`: framtida varde approximeras av senaste kanda varde
- `Daily persistence`: samma tidpunkt foregaende dygn
- `Weekly persistence`: samma tidpunkt foregaende vecka

Detta ar viktigt eftersom solarserier ofta ser bra ut pa enkla baselines for korta horisonter.

### Niva B: Trad- och boostingmodeller

Rekommenderat huvudspar:

- `LightGBM` eller `XGBoost`
- Alternativt `RandomForestRegressor` som enkel referens

Fordelar:

- Bra for tabular features
- Hanterar icke-linjaritet
- Fungerar mycket bra pa mindre till medelstora tidsserier
- Bra med blandade historik-, kalender- och vaderfeatures
- Snabbt att kora i RunPod

### Niva C: Neural sekvensmodell

Endast efter att tabular-sparet ar etablerat:

- `LSTM`
- `GRU`
- `TCN`

Rekommendation:

- Prioritera `LightGBM/XGBoost` som produktionskandidat
- Testa LSTM/TCN som forskningsspar

## 8. Valideringsstrategi

Vi far inte slumpa train/test i tidsserier.

Rekommenderad split:

- Train: `2021-01-01` till `2023-06-30`
- Validation: `2023-07-01` till `2023-12-31`
- Test: `2024-01-01` till `2024-04-02`

Detta ska ses som en startpunkt och kan justeras beroende pa faktisk datakvalitet.

Extra viktigt med vaderdata:

- Traning maste simulera verklig inferens
- Om vi ska anvanda vaderprognos i produktion bor vi, sa langt mojligt, trana och utvardera med historiska vaderprognoser eller en tydlig approximation
- Vi far inte laka in faktisk framtida vaderobservation i feature-setet for senare horisonter

Utoka sedan med walk-forward-backtesting:

- Fold 1: trana pa tidig period, validera pa nasta manad eller kvartal
- Fold 2: expandera train-fonstret
- Fold 3: repetera

Mal:

- Fanga sasongsskillnader
- Minska risken att modellen bara lar sig en begransad del av aret

## 9. Metriker

Vi ska rapportera metrik per horisont.

Primara metriker:

- `MAE`
- `RMSE`
- `nMAE` eller `nRMSE` normaliserad mot installerad toppeffekt eller medelproduktion dagtid

Sekundara metriker:

- `MAPE` endast med forsiktighet eftersom nollor ar vanliga
- `R2`

Praktiskt viktigt:

- Separat utvardering for dagtid
- Separat utvardering for toppproduktion
- Jamforelse mot persistence-baseline for varje horisont

## 10. RunPod-anpassad projektstruktur

Eftersom vi kor i RunPod bor projektet vara script-forst, med notebooks som komplement for analys.

Rekommenderad struktur:

```text
solar_google_collab/
  ARCHITECTURE.md
  requirements.txt
  configs/
    base.yaml
    horizons.yaml
    data_sources.yaml
  data/
    raw/
      production/
      weather/
    processed/
  notebooks/
    01_data_audit.ipynb
    02_feature_engineering.ipynb
    03_train_baselines.ipynb
    04_train_boosting_models.ipynb
    05_evaluate_models.ipynb
    06_inference_demo.ipynb
  scripts/
    prepare_data.py
    train_baselines.py
    train_boosting.py
    evaluate.py
    run_inference.py
  src/
    config.py
    data_loader.py
    weather_loader.py
    preprocessing.py
    feature_engineering.py
    targets.py
    train.py
    evaluate.py
    inference.py
    utils.py
  artifacts/
    models/
    metrics/
    figures/
    logs/
```

Kommentar:

- `scripts/` anvands for reproducerbara treningjobb i RunPod
- Notebooks anvands for analys, visualisering och felsokning
- `src/` innehaller ateranvandbar kod
- `artifacts/` sparar modeller, metrics, logs och plots

## 11. RunPod-drift och experimentkorning

RunPod paverkar framfor allt hur vi kor jobb och hanterar artefakter.

Rekommenderad driftmodell:

- En pod med persistent volume for dataset och artefakter
- Traning kor via script, inte manuella notebook-steg
- Konfiguration per experiment i `yaml`
- Modellartefakter sparas lokalt pa volym och kopieras vid behov till extern lagring

Praktiskt upplagg:

- `prepare_data.py` bygger en standardiserad feature-store
- `train_boosting.py --horizon 1h` trener en enskild modell
- en loop eller jobbmatris kor alla horisonter
- `evaluate.py` samlar metrik till en gemensam rapport

## 12. Rekommenderat kodansvar per modul

### `src/config.py`

- Definierar horisonter
- Satter lagglistor, rolling-fonster, split-datum och filvagar

### `src/data_loader.py`

- Laser produktionsdata
- Normaliserar tidsindex
- Resamplar till 15 minuter
- Markerar missing rows

### `src/weather_loader.py`

- Laser vaderdata fran vald kalla
- Mapper kolumnnamn till intern standard
- Synkar vaderdata mot produktionsdata
- Hanterar saknade vaderobservationer

### `src/preprocessing.py`

- Imputering for features dar det behovs
- Filtrering av ogiltiga rader
- Train/validation/test split

### `src/feature_engineering.py`

- Kalenderfeatures
- Cykliska features
- Solgeometrifeatures
- Laggade features
- Rullande features
- Vaderfeatures
- Horizon-specifika vaderprognosfeatures

### `src/targets.py`

- Skapar targetkolumner per horisont
- Bygger multi-horizon-dataset

### `src/train.py`

- Tranar en modell per horisont
- Sparar modeller og feature importance

### `src/evaluate.py`

- Raknar metrik
- Bygger tabeller och figurer
- Jamfor mot baselines

### `src/inference.py`

- Laddar sparade modeller
- Tar senaste observationsfonstret och kommande vaderprognos
- Returnerar prognosobjekt per horisont

## 13. Artefakter och versionshantering

Vi ska spara:

- Processad featurematris eller mellanlager i `parquet`
- Tranade modeller per horisont
- Metrik i `csv` eller `json`
- Figurer i `png`
- Loggar per treningjobb
- Konfiguration som beskriver exakt vilket experiment som korts

Namngivningskonvention:

- `model_h15m.pkl`
- `model_h1h.pkl`
- `metrics_h1h.json`
- `feature_importance_h6h.csv`

I RunPod:

- Montera persistent volume
- Lasa data fran volym eller objektlagring
- Spara artefakter till volym och eventuellt extern backup

## 14. Rekommenderad arbetsordning

### Fas 1: Datagrund

- Data audit
- Resampling
- Saknade tidssteg
- Synk mellan produktion och vader
- Grundplots per dag, vecka och sasong

### Fas 2: Baseline-system

- Persistence-modeller
- Vaderfri baseline och enkel vadermedveten referens
- Enkla metrics per horisont

### Fas 3: Feature-baserade modeller

- Kalender + solgeometri + laggar + rolling + vaderfeatures
- LightGBM/XGBoost per horisont

### Fas 4: Robust evaluering

- Backtesting
- Jamforelse med och utan vaderdata
- Felanalys per sasong och tid pa dygnet

### Fas 5: Avancerade modeller

- Multi-output
- LSTM eller TCN
- Forbattrad hantering av vaderprognoser
- Eventuellt probabilistiska prognoser

## 15. Beslut vi bor lasa nu

Foljande beslut rekommenderas som projektets initiala arkitektur:

1. Vi bygger en gemensam preprocessing- och feature-pipeline.
2. Vi trener separata modeller per prognoshorisont i forsta versionen.
3. Vi anvander persistence som obligatorisk baseline.
4. Vi prioriterar `LightGBM` eller `XGBoost` som huvudmodell.
5. Vi inkluderar vaderdata och solgeometri redan i V1.
6. Vi evaluerar med tidsbaserad split och walk-forward-backtesting.
7. Vi organiserar projektet som `configs + scripts + src + notebooks + artifacts` for RunPod.

## 16. Min rekommendation

Om vi vill komma fram snabbt med hog chans till bra resultat ska vi borja sa har:

- Horisonter: `15m`, `1h`, `3h`, `6h`, `24h`
- Modelltyp: separat `LightGBM` per horisont
- Features: kalender + sinus/cosinus + solgeometri + laggar + rolling + vaderfeatures
- Baselines: senaste varde och samma tid foregaende dygn
- Evaluering: train/validation/test + en enkel walk-forward-backtest
- Infrastruktur: scriptbaserad korning i RunPod med sparade artefakter pa persistent volume

Det ger en robust och realistisk V1 som passar datamangden, ar snabb att trana i RunPod och ar enkel att bygga vidare pa nar vi senare vill testa mer avancerade modeller.
