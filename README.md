<h1>Разработка решений для Артрейд. Аналитика выкупа (Data Analytics)</h1>

<p>Данный репозиторий представляет собой хакатоновский проект по кейсу от компании Artraid.</p>
<p>Цель проекта - выявить факторы, влияющие на процент выкупа товаров медицинского назначения, построить предсказательную модель и оценить финансовые потери компании от отказов.</p>
<p>Над проектом работает команда <strong>CAW (Ctrl + Alt + Win)</strong>.</p>

<h2>Отчёт</h2>
<p><a href="https://docs.google.com/document/d/1ftziYs5l8C9a2R0BMVc3bz_ZoBZlmsitoqJdzfZtR00/edit?tab=t.twchr62luhq5">Открыть отчёт в Google Docs</a></p>
<p><a href="https://drive.google.com/file/d/1VSdjkJVU3iWcX7OdBprYzcU476A5bolq/view?usp=sharing">Открыть презентацию</a></p>

<h2>Пайплайн</h2>
<ol>
  <li><strong>data_preparation/data_preparation.py</strong> - загрузка сырых данных, приведение типов, удаление дублей и незавершённых сделок → <code>data/clean/clean_data.xlsx</code></li>
  <li><strong>data_preparation/feature_engineering.py</strong> - инженерия признаков: флаги, парсинг тегов и состава заказа, временны́е фичи, интервалы, группы доставки и цены → обновляет <code>data/clean/clean_data.xlsx</code></li>
  <li><strong>data_preparation/feature_region.py</strong> - определение региона РФ по городу (OSM + Natasha) → добавляет <code>lead_region</code> в <code>data/clean/clean_data.xlsx</code></li>
  <li><strong>Cohort/cohort_buyout_analysis.py</strong> - когортный анализ выкупа, графики → <code>Cohort/outputs/charts/</code></li>
  <li><strong>fin/buyout_loss_analysis.py</strong> - анализ финансовых потерь, графики → <code>fin/loss_analysis_charts/</code></li>
  <li><strong>ML/</strong> - обучение моделей (LogReg, RF, XGBoost, CatBoost) → <code>ML/Models/</code>, <code>ML/Models_mk2/</code></li>
</ol>

<h2>Как запустить</h2>

<h3>Обновление данных и графиков (одна команда)</h3>

```bash
python update_data.py
```

Скрипт последовательно запускает подготовку данных, feature engineering, регионы, когортный анализ и финансовые потери. После завершения обновите страницу Streamlit (клавиша <code>R</code>).

<h3>Запуск дашборда</h3>

```bash
streamlit run app.py
```

<h2>Состав команды CAW</h2>
<table>
  <tr><th>Роль</th><th>Участник</th></tr>
  <tr><td>Тимлид / Бизнес-аналитик</td><td>Евгения Адамова</td></tr>
  <tr><td>Data Engineer</td><td>Александра Якшина</td></tr>
  <tr><td>Дата-сайнтист</td><td>Дмитрий Аристархов</td></tr>
  <tr><td>BI-визуализатор</td><td>Аделя Бакирова</td></tr>
  <tr><td>Бывший ML-инженер</td><td>Андраник Факирян</td></tr>
</table>
