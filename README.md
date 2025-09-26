## Исследование моделей на сгенерерированных данных

Репозиторий, содержащий код 3-х файлов для проведения исследования моделей на сгенерированных синтетических датасетах

+ все 3 файла запускаются командами в консоли `python name_file.py` 
+ выходные данные сохраняются в папку, в которой происходит запуск
+ предварительно установите библиотеки указанные в файле [requirements.txt](https://github.com/YOya-Shep/test-models/blob/main/requirements.txt)
+ использовалcя `python 3.9.7` (рандомно версию выбрала, тк не хотела самый новый использовать)
+ примеры выходных результатов можно просмотреть в папке [results](https://github.com/YOya-Shep/test-models/tree/main/results)

### task_1 Классификация

В блоке `if __name__ == "__main__":` происходят последовательные вызовы функций и генерация синтетического датасета (параметры указаны там же). Обычно его используют когда пишут модуль, предназначенный для непосредственного исполнения. Результаты сохраняются в папке task_of_classification, и выглядят как файл results_model.txt сохраненными наилучшими параметрами и коэффициентом Джини и файл scatter_plot.png с диаграммой рассеяния показывающей взаимосвязь двух признаков для обоих классов. 
![scatter_plot_2_classes](https://github.com/YOya-Shep/test-models/blob/main/results/task_of_classification_20250926_014520/scatter_plot.png) 

Здесь в функции `scattering_graph` создан список `custom_colors = ["#00528d", "#d36900"]`. Поэтому, если количество классов будет изменено, то нужно
+ либо добавить ещё цветов,
+ либо можно изменить в строке `plt.scatter(X[idx, 0], X[idx, 1], c=custom_colors[i], label=f'Класс {i}', alpha=0.7)` параметр цвета на `c=y, cmap='viridis'` или `c=y, cmap='Set1'`(или использовать другой аналогичный набор цветов).


### task_2 Регрессия

Здесь вынесены в начало файла константы `N_SAMPLES = 1000 N_FEATURES = 10`, весь код разделен на функции и есть отдельная функция def main() - это облегчает настройку и переиспользование кода.

Результаты сохраняются в папку [task2_regression](https://github.com/YOya-Shep/test-models/tree/main/results/task2_regression_20250926_202030). в том числе сохраняется и модель [model.jonlib](https://github.com/YOya-Shep/test-models/blob/main/results/task2_regression_20250926_202030/model.joblib).

Исследуя полученные данные можно сделать следующие выводы

* Нет мультиколлинеарности тк
    + в матрице корреляции все значения близки к 0, => нет линейной связи
    + сохраненные в analisis_results.json значения VIF близки к 1

на полученном графике остатков residuals_plot значения находятся около 0, это подтверждает отсутствие линейной взаимосвязи
![residuals_plot.png](https://github.com/YOya-Shep/test-models/blob/main/results/task2_regression_20250926_202030/residuals_plot.png)

в файл [building_results.json](https://github.com/YOya-Shep/test-models/blob/main/results/task2_regression_20250926_202030/building_results.json) сохранены 
+ прогнозы(predictions)
+ mse около 0,25. Значение близко к дисперсии шума (0.5² = 0.25) => модель почти идеально уловила линейную зависимость
+ r2 близко к 1, это значит что модель хорошо объясняет целевой переменной

Важность признаков сохранена в [analisis_results.json](https://github.com/YOya-Shep/test-models/blob/main/results/task2_regression_20250926_202030/analisis_results.json) как feature_importance и принимает значения близкие к 1, к -1 или к 0. 0 означает, что признак практически не влияет на вывод, 1 - сильно влияет с положительной корреляцией, -1 - сильно влияет с отрицательной корреляцией. Там же рядом сохранены значения остатков (residuals).


### task_3 Временной ряд

Здесь также есть константа `N_SAMPLES = 1000` в начале кода. 

как и во втором файле, в task3_time_series.py есть отдельная функция def main(), поэтому можно использовать 
```
import task3_time_series

task3_time_series.main()
```
для запуска всего функционала, импортируя модуль для повторного использования кода в других проектах. Аналогично можно вызывать отдельные функции


в папке [task3_time_series_time](https://github.com/YOya-Shep/test-models/tree/main/results/task3_time_series_20250926_202111) содержатся результаты работы.
График time_series.png показывает сгенерированный временной ряд, 
![time_series.png](https://github.com/YOya-Shep/test-models/blob/main/results/task3_time_series_20250926_202111/time_series.png)

График AR_vs_ETS.png показывает ряд и предсказания моделей,
![AR_vs_ETS.png](https://github.com/YOya-Shep/test-models/blob/main/results/task3_time_series_20250926_202111/AR_vs_ETS.png)

Анализируя результаты измеренения MAE и RMSE сохраненные в results.json, можно сделать вывод, что автокорреляционная модель справилась немного лучше, чем модель экспоненциального сглаживания
```
{
    "AR": {
        "MAE": 0.9859119679889187,
        "RMSE": 1.2276188549752496
    },
    "ETS": {
        "MAE": 1.1497969955484568,
        "RMSE": 1.4220016537523885
    }
}
```

в папке [task3_time_series_error](https://github.com/YOya-Shep/test-models/tree/main/results/task3_time_series_error) содержатся результаты моделей до исправления. Тут на рисунке AR_vs_ETS видно, что предсказания ETS слишком резко уходят вверх, а предсказания AR постепенно сглаживаются. Чтобы это исправить, были внесены изменения в построение моделей. ETS была добавлена сезонность `model = ExponentialSmoothing(y_train, trend="add", damped_trend=True, seasonal="add", seasonal_periods=50)`, а для AR увеличен параметр lags, отвечающий за количество используемых предыдущих значений, `model = AutoReg(y_train, lags=40)`.
![AR_vs_ETS.png](https://github.com/YOya-Shep/test-models/blob/main/results/task3_time_series_error/AR_vs_ETS.png)
А в файле [results.json](https://github.com/YOya-Shep/test-models/blob/main/results/task3_time_series_error/results.json) видно что первые значения MAE и RMSE были заметно больше итоговых.
