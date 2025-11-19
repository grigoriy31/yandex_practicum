#!/usr/bin/env python
# coding: utf-8

# # Анализ лояльности пользователей Яндекс Афиши

# ## Этапы выполнения проекта
# 
# ### 1. Загрузка данных и их предобработка
# 
# ---
# 
# **Задача 1.1:** Напишите SQL-запрос, выгружающий в датафрейм pandas необходимые данные. Используйте следующие параметры для подключения к базе данных `data-analyst-afisha`:
# 
# - **Хост** — `rc1b-wcoijxj3yxfsf3fs.mdb.yandexcloud.net`
# - **База данных** — `data-analyst-afisha`
# - **Порт** — `6432`
# - **Аутентификация** — `Database Native`
# - **Пользователь** — `praktikum_student`
# - **Пароль** — `Sdf4$2;d-d30pp`
# 
# Для выгрузки используйте запрос из предыдущего урока и библиотеку SQLAlchemy.
# 
# Выгрузка из базы данных SQL должна позволить собрать следующие данные:
# 
# - `user_id` — уникальный идентификатор пользователя, совершившего заказ;
# - `device_type_canonical` — тип устройства, с которого был оформлен заказ (`mobile` — мобильные устройства, `desktop` — стационарные);
# - `order_id` — уникальный идентификатор заказа;
# - `order_dt` — дата создания заказа (используйте данные `created_dt_msk`);
# - `order_ts` — дата и время создания заказа (используйте данные `created_ts_msk`);
# - `currency_code` — валюта оплаты;
# - `revenue` — выручка от заказа;
# - `tickets_count` — количество купленных билетов;
# - `days_since_prev` — количество дней от предыдущей покупки пользователя, для пользователей с одной покупкой — значение пропущено;
# - `event_id` — уникальный идентификатор мероприятия;
# - `service_name` — название билетного оператора;
# - `event_type_main` — основной тип мероприятия (театральная постановка, концерт и так далее);
# - `region_name` — название региона, в котором прошло мероприятие;
# - `city_name` — название города, в котором прошло мероприятие.
# 
# ---
# 

# In[1]:


# Используйте ячейки типа Code для вашего кода,
# а ячейки типа Markdown для комментариев и выводов


# In[2]:


# При необходимости добавляйте новые ячейки для кода или текста


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Ипортируем необходимы бибилиотеки, задаем параметры подключения к БД.
# Переносим ранее созданный SQL запрос.
#     Проверяем корректность выгруженных данных.
#     
# </div>

# In[3]:


from sqlalchemy import create_engine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Параметры подключения к базе данных
db_config = {
    'host': 'rc1b-wcoijxj3yxfsf3fs.mdb.yandexcloud.net',
    'port': 6432,
    'database': 'data-analyst-afisha',
    'user': 'praktikum_student',
    'password': 'Sdf4$2;d-d30pp'
}

# Создаем строку подключения
connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

# Создаем engine SQLAlchemy
engine = create_engine(connection_string)


# In[4]:


query = """
SELECT 
    p.user_id,
    p.device_type_canonical,
    p.order_id,
    p.created_dt_msk AS order_dt,
    p.created_ts_msk AS order_ts,
    p.currency_code,
    p.revenue,
    p.tickets_count,
    p.created_dt_msk::DATE - LAG(p.created_dt_msk::DATE) OVER (
        PARTITION BY p.user_id 
        ORDER BY p.created_dt_msk
    ) AS days_since_prev,
    p.event_id,
    e.event_name_code AS event_name,
    e.event_type_main,
    p.service_name,  -- берём из purchases, как указано в схеме
    c.city_name,
    r.region_name
FROM afisha.purchases AS p
JOIN afisha.events AS e
    ON p.event_id = e.event_id
JOIN afisha.city AS c
    ON e.city_id = c.city_id
JOIN afisha.regions AS r
    ON c.region_id = r.region_id
WHERE 
    p.device_type_canonical IN ('mobile', 'desktop')
    AND e.event_type_main != 'фильм'
ORDER BY p.user_id
"""


# In[5]:


try:
    df = pd.read_sql(query, engine)
    print("Данные успешно загружены!")
    print(df.head())  # Вывод первых нескольких строк DataFrame
except Exception as e:
    print(f"Ошибка при выполнении запроса: {e}")
finally:
    engine.dispose()  # Закрываем соединение с базой данных


# ---
# 
# **Задача 1.2:** Изучите общую информацию о выгруженных данных. Оцените корректность выгрузки и объём полученных данных.
# 
# Предположите, какие шаги необходимо сделать на стадии предобработки данных — например, скорректировать типы данных.
# 
# Зафиксируйте основную информацию о данных в кратком промежуточном выводе.
# 
# ---

# In[6]:


print(df.duplicated().sum())
df.shape
df.info()


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>данные выгружены корректно.
#     единственный столбец с пропусками days_since_prev созданный нами.
#     Необходимы действия для обработки данных:
#     1. Проверка на явные и неявные дубликаты в категорийных столбцах
#     2. Выгрузка статистических данных для числовых слобцов.
# </div>

# In[7]:


# Статистическое описание числовых столбцов
print(df.describe())




# In[8]:


# Проверка на наличие пропусков
print(df.isnull().sum())


# In[9]:


# Проверка уникальных значений в категориальных столбцах
categorical_columns = ['device_type_canonical', 'currency_code', 'event_name', 'event_type_main', 'service_name', 'city_name', 'region_name']
for col in categorical_columns:
    print(f"Уникальные значения в {col}:")
    print(df[col].unique())


# ---
# 
# ###  2. Предобработка данных
# 
# Выполните все стандартные действия по предобработке данных:
# 
# ---
# 
# **Задача 2.1:** Данные о выручке сервиса представлены в российских рублях и казахстанских тенге. Приведите выручку к единой валюте — российскому рублю.
# 
# Для этого используйте датасет с информацией о курсе казахстанского тенге по отношению к российскому рублю за 2024 год — `final_tickets_tenge_df.csv`. Его можно загрузить по пути `https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv')`
# 
# Значения в рублях представлено для 100 тенге.
# 
# Результаты преобразования сохраните в новый столбец `revenue_rub`.
# 
# ---
# 

# In[10]:


# Загрузка данных о курсе тенге
url = "https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv"
tenge_exchange_rate = pd.read_csv(url)
tenge_exchange_rate.info()
print(tenge_exchange_rate.head())
print(tenge_exchange_rate.columns)


# In[11]:



# Загрузка данных о курсе тенге
url = "https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv"
tenge_exchange_rate = pd.read_csv(url)

# Проверяем названия столбцов
print("Столбцы в tenge_exchange_rate:", tenge_exchange_rate.columns)

# Переименуем столбцы для удобства работы
tenge_exchange_rate.rename(columns={'data': 'order_dt', 'curs': 'rate'}, inplace=True)

# Преобразуем столбец order_dt в формат datetime
tenge_exchange_rate['order_dt'] = pd.to_datetime(tenge_exchange_rate['order_dt'])

# Убедимся, что nominal равен 100 (курс приведен для 100 тенге)
if not all(tenge_exchange_rate['nominal'] == 100):
    raise ValueError("Номинал курса должен быть равен 100!")

# Удалим ненужные столбцы
tenge_exchange_rate.drop(columns=['nominal', 'cdx'], inplace=True)

# Объединяем таблицы по дате заказа
df['order_dt'] = pd.to_datetime(df['order_dt'])  # Убедимся, что order_dt в df имеет тип datetime
df = df.merge(tenge_exchange_rate, on='order_dt', how='left')

# Создаем новый столбец revenue_rub
df['revenue_rub'] = df.apply(
    lambda row: row['revenue'] if row['currency_code'] == 'rub' 
               else (row['revenue'] / 100) * row['rate'] if row['currency_code'] == 'kzt' 
               else None, axis=1
)

# Проверка результатов
print(df[['currency_code', 'revenue', 'rate', 'revenue_rub']].head())
print(df['revenue_rub'].describe())


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
#     Выручка успешно приведена к единой валюте (российскому рублю) с использованием курса тенге.
#     Новый столбец revenue_rub содержит выручку в рублях для всех заказов независимо от исходной валюты.
# <br>
# </div>

# ---
# 
# **Задача 2.2:**
# 
# - Проверьте данные на пропущенные значения. Если выгрузка из SQL была успешной, то пропуски должны быть только в столбце `days_since_prev`.
# - Преобразуйте типы данных в некоторых столбцах, если это необходимо. Обратите внимание на данные с датой и временем, а также на числовые данные, размерность которых можно сократить.
# - Изучите значения в ключевых столбцах. Обработайте ошибки, если обнаружите их.
#     - Проверьте, какие категории указаны в столбцах с номинальными данными. Есть ли среди категорий такие, что обозначают пропуски в данных или отсутствие информации? Проведите нормализацию данных, если это необходимо.
#     - Проверьте распределение численных данных и наличие в них выбросов. Для этого используйте статистические показатели, гистограммы распределения значений или диаграммы размаха.
#         
#         Важные показатели в рамках поставленной задачи — это выручка с заказа (`revenue_rub`) и количество билетов в заказе (`tickets_count`), поэтому в первую очередь проверьте данные в этих столбцах.
#         
#         Если обнаружите выбросы в поле `revenue_rub`, то отфильтруйте значения по 99 перцентилю.
# 
# После предобработки проверьте, были ли отфильтрованы данные. Если были, то оцените, в каком объёме. Сформулируйте промежуточный вывод, зафиксировав основные действия и описания новых столбцов.
# 
# ---

# In[12]:


# Проверка пропущенных значений
print(df.isnull().sum())


# In[13]:


# Преобразование revenue_rub в более компактный числовой формат
df['revenue_rub'] = pd.to_numeric(df['revenue_rub'], downcast='float')

# Преобразование tickets_count в целочисленный формат
df['tickets_count'] = pd.to_numeric(df['tickets_count'], downcast='integer')


# In[ ]:





# In[14]:


print(df[['revenue_rub', 'tickets_count']].describe())


# In[15]:





# Создаем фигуру с двумя графиками (для revenue_rub и tickets_count)
plt.figure(figsize=(14, 6))

# Гистограмма для revenue_rub
plt.subplot(1, 2, 1)
plt.hist(df['revenue_rub'], bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.title('Распределение выручки (revenue_rub)', fontsize=14)
plt.xlabel('Выручка, руб.', fontsize=12)
plt.ylabel('Количество заказов', fontsize=12)
plt.grid(alpha=0.3)

# Гистограмма для tickets_count
plt.subplot(1, 2, 2)
plt.hist(df['tickets_count'], bins=20, color='green', alpha=0.7, edgecolor='black')
plt.title('Распределение количества билетов (tickets_count)', fontsize=14)
plt.xlabel('Количество билетов', fontsize=12)
plt.ylabel('Количество заказов', fontsize=12)
plt.grid(alpha=0.3)

# Показываем графики
plt.tight_layout()
plt.show()


# In[16]:


# Диаграмма размаха для revenue_rub
plt.figure(figsize=(10, 6))
plt.boxplot(df['revenue_rub'])
plt.title('Диаграмма размаха для revenue_rub')
plt.ylabel('Выручка, руб.')
plt.show()

# Диаграмма размаха для tickets_count
plt.figure(figsize=(10, 6))
plt.boxplot(df['tickets_count'])
plt.title('Диаграмма размаха для tickets_count')
plt.ylabel('Количество билетов')
plt.show()


# In[17]:


# Вычисление 99 перцентиля
revenue_99_percentile = df['revenue_rub'].quantile(0.99)
tickets_count_99_percentile = df['tickets_count'].quantile(0.99)
# Фильтрация данных
filtered_df = df[df['revenue_rub'] <= revenue_99_percentile]
filtered_df = filtered_df[filtered_df['tickets_count'] <= tickets_count_99_percentile]
# Оценка объема отфильтрованных данных
print(f"Процент удаленных данных: {((len(df) - len(filtered_df)) / len(df)) * 100:.2f}%")


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br> Все данные нормализованы и выгружены корректно.
#     данные отфилтрованы по 99му процентиля дабы исключить выбросы.
#     процент отфильтрофаных данных составляет 1,03
# </div>

# ---
# 
# ### 3. Создание профиля пользователя
# 
# В будущем отдел маркетинга планирует создать модель для прогнозирования возврата пользователей. Поэтому сейчас они просят вас построить агрегированные признаки, описывающие поведение и профиль каждого пользователя.
# 
# ---
# 
# **Задача 3.1.** Постройте профиль пользователя — для каждого пользователя найдите:
# 
# - дату первого и последнего заказа;
# - устройство, с которого был сделан первый заказ;
# - регион, в котором был сделан первый заказ;
# - билетного партнёра, к которому обращались при первом заказе;
# - жанр первого посещённого мероприятия (используйте поле `event_type_main`);
# - общее количество заказов;
# - средняя выручка с одного заказа в рублях;
# - среднее количество билетов в заказе;
# - среднее время между заказами.
# 
# После этого добавьте два бинарных признака:
# 
# - `is_two` — совершил ли пользователь 2 и более заказа;
# - `is_five` — совершил ли пользователь 5 и более заказов.
# 
# **Рекомендация:** перед тем как строить профиль, отсортируйте данные по времени совершения заказа.
# 
# ---
# 

# In[18]:


# Сортируем данные по user_id и order_dt
df = df.sort_values(by=['user_id', 'order_dt']).reset_index(drop=True)


# In[19]:


# Надем дату для первого и последнего заказа пользователся
first_order_date = df.groupby('user_id')['order_dt'].min().rename('first_order_date')
last_order_date = df.groupby('user_id')['order_dt'].max().rename('last_order_date')
# Найдем с какого устройства был первый заказ
first_device = df.groupby('user_id')['device_type_canonical'].first().rename('first_device')
# Найдем первый регион заказа для клиента
first_region = df.groupby('user_id')['region_name'].first().rename('first_region')
# Найдем первго билетного оператора
first_service_name = df.groupby('user_id')['service_name'].first().rename('first_service_name')
# Найдем первый данр мероприятия
first_event_genre = df.groupby('user_id')['event_type_main'].first().rename('first_event_genre')
# Найдем общее кол-во заказов клиента
total_orders = df.groupby('user_id').size().rename('total_orders')
# Найдем среднее кол-во билетов в 1 заказе
avg_tickets_per_order = df.groupby('user_id')['tickets_count'].mean().rename('avg_tickets_per_order')
# Найдем среднее время между заказами
avg_days_between_orders = df.groupby('user_id')['days_since_prev'].mean().rename('avg_days_between_orders')
# Найдем среднее выручка с одного заказа
avg_revenue_per_order = df.groupby('user_id')['revenue_rub'].mean().rename('avg_revenue_per_order')


# In[20]:


# Добавим бинарные признаки
is_two = (total_orders >= 2).astype(int).rename('is_two')
is_five = (total_orders >= 5).astype(int).rename('is_five')


# In[21]:


# Соберем найденные данный в один датафрейм


# In[22]:


user_profile = pd.concat([
    first_order_date,
    last_order_date,
    first_device,
    first_region,
    first_service_name,
    first_event_genre,
    total_orders,
    avg_revenue_per_order,
    avg_tickets_per_order,
    avg_days_between_orders,
    is_two,
    is_five
], axis=1).reset_index()

print(user_profile.head())


# ---
# 
# **Задача 3.2.** Прежде чем проводить исследовательский анализ данных и делать выводы, важно понять, с какими данными вы работаете: насколько они репрезентативны и нет ли в них аномалий.
# 
# Используя данные о профилях пользователей, рассчитайте:
# 
# - общее число пользователей в выборке;
# - среднюю выручку с одного заказа;
# - долю пользователей, совершивших 2 и более заказа;
# - долю пользователей, совершивших 5 и более заказов.
# 
# Также изучите статистические показатели:
# 
# - по общему числу заказов;
# - по среднему числу билетов в заказе;
# - по среднему количеству дней между покупками.
# 
# По результатам оцените данные: достаточно ли их по объёму, есть ли аномальные значения в данных о количестве заказов и среднем количестве билетов?
# 
# Если вы найдёте аномальные значения, опишите их и примите обоснованное решение о том, как с ними поступить:
# 
# - Оставить и учитывать их при анализе?
# - Отфильтровать данные по какому-то значению, например, по 95-му или 99-му перцентилю?
# 
# Если вы проведёте фильтрацию, то вычислите объём отфильтрованных данных и выведите статистические показатели по обновлённому датасету.

# In[23]:


total_users = len(user_profile)
print(f"Общее число пользователей: {total_users}")

mean_revenue_per_order = user_profile['avg_revenue_per_order'].mean()
print(f"Средняя выручка с одного заказа: {mean_revenue_per_order:.2f} руб.")

users_with_two_or_more_orders = user_profile[user_profile['is_two'] == 1]
share_two_or_more_orders = len(users_with_two_or_more_orders) / total_users
print(f"Доля пользователей, совершивших 2 и более заказа: {share_two_or_more_orders:.2%}")

users_with_five_or_more_orders = user_profile[user_profile['is_five'] == 1]
share_five_or_more_orders = len(users_with_five_or_more_orders) / total_users
print(f"Доля пользователей, совершивших 5 и более заказов: {share_five_or_more_orders:.2%}")


# In[24]:


print("Статистика по общему числу заказов:")
print(user_profile['total_orders'].describe())

print("Статистика по среднему числу билетов в заказе:")
print(user_profile['avg_tickets_per_order'].describe())

print("Статистика по среднему количеству дней между покупками:")
print(user_profile['avg_days_between_orders'].describe())


# In[25]:


plt.figure(figsize=(10, 6))
plt.hist(user_profile['total_orders'], bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.title('Распределение общего числа заказов', fontsize=14)
plt.xlabel('Количество заказов', fontsize=12)
plt.ylabel('Количество пользователей', fontsize=12)
plt.grid(alpha=0.3)
plt.show()


# In[26]:


plt.figure(figsize=(10, 6))
plt.hist(user_profile['avg_tickets_per_order'], bins=20, color='green', alpha=0.7, edgecolor='black')
plt.title('Распределение среднего числа билетов в заказе', fontsize=14)
plt.xlabel('Среднее количество билетов', fontsize=12)
plt.ylabel('Количество пользователей', fontsize=12)
plt.grid(alpha=0.3)
plt.show()


# In[27]:


# Фильтрация по total_orders
total_orders_95_percentile = user_profile['total_orders'].quantile(0.95)
filtered_profile = user_profile[user_profile['total_orders'] <= total_orders_95_percentile]

# Фильтрация по avg_tickets_per_order
tickets_95_percentile = user_profile['avg_tickets_per_order'].quantile(0.95)
filtered_profile = filtered_profile[filtered_profile['avg_tickets_per_order'] <= tickets_95_percentile]

# Оценка объема отфильтрованных данных
filtered_users = len(filtered_profile)
print(f"Процент отфильтрованных пользователей: {(1 - filtered_users / total_users) * 100:.2f}%")


# In[28]:


print("Статистика по общему числу заказов (после фильтрации):")
print(filtered_profile['total_orders'].describe())

print("Статистика по среднему числу билетов в заказе (после фильтрации):")
print(filtered_profile['avg_tickets_per_order'].describe())

print("Статистика по среднему количеству дней между покупками (после фильтрации):")
print(filtered_profile['avg_days_between_orders'].describe())


# ---
# 
# ### 4. Исследовательский анализ данных
# 
# Следующий этап — исследование признаков, влияющих на возврат пользователей, то есть на совершение повторного заказа. Для этого используйте профили пользователей.

# 
# 
# #### 4.1. Исследование признаков первого заказа и их связи с возвращением на платформу
# 
# Исследуйте признаки, описывающие первый заказ пользователя, и выясните, влияют ли они на вероятность возвращения пользователя.
# 
# ---
# 
# **Задача 4.1.1.** Изучите распределение пользователей по признакам.
# 
# - Сгруппируйте пользователей:
#     - по типу их первого мероприятия;
#     - по типу устройства, с которого совершена первая покупка;
#     - по региону проведения мероприятия из первого заказа;
#     - по билетному оператору, продавшему билеты на первый заказ.
# - Подсчитайте общее количество пользователей в каждом сегменте и их долю в разрезе каждого признака. Сегмент — это группа пользователей, объединённых определённым признаком, то есть объединённые принадлежностью к категории. Например, все клиенты, сделавшие первый заказ с мобильного телефона, — это сегмент.
# - Ответьте на вопрос: равномерно ли распределены пользователи по сегментам или есть выраженные «точки входа» — сегменты с наибольшим числом пользователей?
# 
# ---
# 

# In[29]:


# Считаем количество пользователей в каждом сегменте
event_genre_distribution = user_profile['first_event_genre'].value_counts(normalize=False).reset_index()
event_genre_distribution.columns = ['first_event_genre', 'user_count']

# Доля пользователей в каждом сегменте
event_genre_distribution['share'] = event_genre_distribution['user_count'] / event_genre_distribution['user_count'].sum()

print("Распределение пользователей по типу первого мероприятия:")
print(event_genre_distribution)


plt.figure(figsize=(8, 8))
plt.pie(event_genre_distribution['user_count'], labels=event_genre_distribution['first_event_genre'], autopct='%1.1f%%')
plt.title('Распределение пользователей по типу первого мероприятия')
plt.show()


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
#     Большинство пользователей начинает свое взаимодействие с Концертов и Другое, 44 и 24% соответсвено. Они и являются точкой входа.
# <br>

# In[30]:


device_distribution = user_profile['first_device'].value_counts(normalize=False).reset_index()
device_distribution.columns = ['first_device', 'user_count']
device_distribution['share'] = device_distribution['user_count'] / device_distribution['user_count'].sum()

print("Распределение пользователей по устройству первого заказа:")
print(device_distribution)

plt.figure(figsize=(8, 6))
sns.barplot(data=device_distribution, x='first_device', y='user_count', palette='viridis')
plt.title('Распределение пользователей по устройству первого заказа')
plt.xlabel('Тип устройства')
plt.ylabel('Количество пользователей')
plt.show()


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Большинство пользователей начинает свое взаимодействие с сервисом - через мобильное утсройство.
# </div>

# In[31]:


region_distribution = user_profile['first_region'].value_counts(normalize=False).reset_index()
region_distribution.columns = ['first_region', 'user_count']
region_distribution['share'] = region_distribution['user_count'] / region_distribution['user_count'].sum()

print("Распределение пользователей по региону первого заказа:")
print(region_distribution)

plt.figure(figsize=(10, 6))
sns.barplot(data=region_distribution.head(10), x='first_region', y='user_count', palette='magma')
plt.title('Распределение пользователей по региону первого заказа (топ-10)')
plt.xlabel('Регион')
plt.ylabel('Количество пользователей')
plt.xticks(rotation=45)
plt.show()


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Самыективные регионы по первому заказу:
#     Каменевский регион 32%
#     Североярская обасть 17%
# </div>

# In[32]:


service_distribution = user_profile['first_service_name'].value_counts(normalize=False).reset_index()
service_distribution.columns = ['first_service_name', 'user_count']
service_distribution['share'] = service_distribution['user_count'] / service_distribution['user_count'].sum()

print("Распределение пользователей по билетному оператору первого заказа:")
print(service_distribution)

plt.figure(figsize=(10, 6))
sns.barplot(data=service_distribution.head(10), x='first_service_name', y='user_count', palette='plasma')
plt.title('Распределение пользователей по билетному оператору первого заказа (топ-10)')
plt.xlabel('Билетный оператор')
plt.ylabel('Количество пользователей')
plt.xticks(rotation=45)
plt.show()


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Хедлайнер по продаже первых билетов: Билеты без пролем - 23,8%
# </div>

# ---
# 
# **Задача 4.1.2.** Проанализируйте возвраты пользователей:
# 
# - Для каждого сегмента вычислите долю пользователей, совершивших два и более заказа.
# - Визуализируйте результат подходящим графиком. Если сегментов слишком много, то поместите на график только 10 сегментов с наибольшим количеством пользователей. Такое возможно с сегментами по региону и по билетному оператору.
# - Ответьте на вопросы:
#     - Какие сегменты пользователей чаще возвращаются на Яндекс Афишу?
#     - Наблюдаются ли успешные «точки входа» — такие сегменты, в которых пользователи чаще совершают повторный заказ, чем в среднем по выборке?
# 
# При интерпретации результатов учитывайте размер сегментов: если в сегменте мало пользователей (например, десятки), то доли могут быть нестабильными и недостоверными, то есть показывать широкую вариацию значений.
# 
# ---
# 

# In[33]:


event_genre_return_rate = (
    user_profile.groupby('first_event_genre')['is_two']
    .agg(['sum', 'count'])
    .reset_index()
)
event_genre_return_rate['return_rate'] = event_genre_return_rate['sum'] / event_genre_return_rate['count']

# Сортируем по убыванию количества пользователей
event_genre_return_rate = event_genre_return_rate.sort_values(by='return_rate', ascending=False)

print("Доля возвращающихся пользователей по типу первого мероприятия:")
print(event_genre_return_rate[['first_event_genre', 'count', 'return_rate']])

plt.figure(figsize=(10, 6))
sns.barplot(data=event_genre_return_rate, x='first_event_genre', y='return_rate', palette='viridis')
plt.title('Доля возвращающихся пользователей по типу первого мероприятия')
plt.xlabel('Тип мероприятия')
plt.ylabel('Доля возвращающихся пользователей')
plt.xticks(rotation=45)
plt.show()


# In[34]:


device_return_rate = (
    user_profile.groupby('first_device')['is_two']
    .agg(['sum', 'count'])
    .reset_index()
)
device_return_rate['return_rate'] = device_return_rate['sum'] / device_return_rate['count']

print("Доля возвращающихся пользователей по устройству первого заказа:")
print(device_return_rate[['first_device', 'count', 'return_rate']])

plt.figure(figsize=(8, 6))
sns.barplot(data=device_return_rate, x='first_device', y='return_rate', palette='plasma')
plt.title('Доля возвращающихся пользователей по устройству первого заказа')
plt.xlabel('Тип устройства')
plt.ylabel('Доля возвращающихся пользователей')
plt.show()


# In[35]:


region_return_rate = (
    user_profile.groupby('first_region')['is_two']
    .agg(['sum', 'count'])
    .reset_index()
)
region_return_rate['return_rate'] = region_return_rate['sum'] / region_return_rate['count']

# Берем топ-10 регионов по количеству пользователей
region_return_rate = region_return_rate.sort_values(by='return_rate', ascending=False).head(10)

print("Доля возвращающихся пользователей по региону первого заказа (топ-10):")
print(region_return_rate[['first_region', 'count', 'return_rate']])

plt.figure(figsize=(12, 6))
sns.barplot(data=region_return_rate, x='first_region', y='return_rate', palette='magma')
plt.title('Доля возвращающихся пользователей по региону первого заказа (топ-10)')
plt.xlabel('Регион')
plt.ylabel('Доля возвращающихся пользователей')
plt.xticks(rotation=45)
plt.show()


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Не смотря на ярко выраженные точки входа по представленным выше категориям,
# выделить какую либо категорию с большим кол-вом возвратных клиентов возможности.
# Наоборот наблюдаютмя точки выхода:
#     При среднем % возврата пользователей 60%.
#     Клиенты которые первым приобретиали билеты на Спорт и Елки составляет 56 и 54%
#     
# </div>

# ---
# 
# **Задача 4.1.3.** Опираясь на выводы из задач выше, проверьте продуктовые гипотезы:
# 
# - **Гипотеза 1.** Тип мероприятия влияет на вероятность возврата на Яндекс Афишу: пользователи, которые совершили первый заказ на спортивные мероприятия, совершают повторный заказ чаще, чем пользователи, оформившие свой первый заказ на концерты.
# - **Гипотеза 2.** В регионах, где больше всего пользователей посещают мероприятия, выше доля повторных заказов, чем в менее активных регионах.
# 
# ---

# In[36]:


# Фильтрация данных для спортивных мероприятий и концертов
sports_data = event_genre_return_rate[event_genre_return_rate['first_event_genre'] == 'спорт']
concerts_data = event_genre_return_rate[event_genre_return_rate['first_event_genre'] == 'концерты']

# Доля возвращающихся пользователей
sports_return_rate = sports_data['return_rate'].values[0] if not sports_data.empty else None
concerts_return_rate = concerts_data['return_rate'].values[0] if not concerts_data.empty else None

print(f"Доля возвращающихся пользователей (спортивные мероприятия): {sports_return_rate:.2%}")
print(f"Доля возвращающихся пользователей (концерты): {concerts_return_rate:.2%}")


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Гипотеза 1 не подтверждена
# </div>

# In[37]:


# Разделение на активные и менее активные регионы
active_regions = region_return_rate.head(2)  # Берем топ-2 региона по количеству пользователей
less_active_regions = region_return_rate.tail(8)  # Остальные регионы

# Средняя доля возвращающихся пользователей
active_regions_avg_return_rate = active_regions['return_rate'].mean()
less_active_regions_avg_return_rate = less_active_regions['return_rate'].mean()

print(f"Средняя доля возвращающихся пользователей (активные регионы): {active_regions_avg_return_rate:.2%}")
print(f"Средняя доля возвращающихся пользователей (менее активные регионы): {less_active_regions_avg_return_rate:.2%}")


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Гипотеза 2 - не подтвеждена.
# </div>

# ---
# 
# #### 4.2. Исследование поведения пользователей через показатели выручки и состава заказа
# 
# Изучите количественные характеристики заказов пользователей, чтобы узнать среднюю выручку сервиса с заказа и количество билетов, которое пользователи обычно покупают.
# 
# Эти метрики важны не только для оценки выручки, но и для оценки вовлечённости пользователей. Возможно, пользователи с более крупными и дорогими заказами более заинтересованы в сервисе и поэтому чаще возвращаются.
# 
# ---
# 
# **Задача 4.2.1.** Проследите связь между средней выручкой сервиса с заказа и повторными заказами.
# 
# - Постройте сравнительные гистограммы распределения средней выручки с билета (`avg_revenue_rub`):
#     - для пользователей, совершивших один заказ;
#     - для вернувшихся пользователей, совершивших 2 и более заказа.
# - Ответьте на вопросы:
#     - В каких диапазонах средней выручки концентрируются пользователи из каждой группы?
#     - Есть ли различия между группами?
# 
# Текст на сером фоне:
#     
# **Рекомендация:**
# 
# 1. Используйте одинаковые интервалы (`bins`) и прозрачность (`alpha`), чтобы визуально сопоставить распределения.
# 2. Задайте параметру `density` значение `True`, чтобы сравнивать форму распределений, даже если число пользователей в группах отличается.
# 
# ---
# 

# In[38]:


# Группа 1: Пользователи, совершившие один заказ
one_order_users = user_profile[user_profile['total_orders'] == 1]

# Группа 2: Вернувшиеся пользователи (2 и более заказов)
returning_users = user_profile[user_profile['is_two'] == 1]


# In[39]:


# Построение гистограмм
plt.figure(figsize=(12, 6))

# Группа 1: Пользователи с одним заказом
plt.hist(one_order_users['avg_revenue_per_order'], bins=50, alpha=0.6, label='Один заказ', density=True, color='red')

# Группа 2: Вернувшиеся пользователи
plt.hist(returning_users['avg_revenue_per_order'], bins=50, alpha=0.6, label='2+ заказа', density=True, color='blue')

# Настройка графика
plt.title('Распределение средней выручки с заказа для разных групп пользователей', fontsize=14)
plt.xlabel('Средняя выручка с заказа, руб.', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Показываем график
plt.show()


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Пользователи совершившие 2 и более покупок менее активно покупают самые дешевые билеты,
#     Пользователи совершившие 1 покупку очень часто делают самые шевые покупки. 
# </div>

# In[40]:


# Статистика для пользователей с одним заказом
print("Статистика для пользователей с одним заказом:")
print(one_order_users['avg_revenue_per_order'].describe())

# Статистика для вернувшихся пользователей
print("Статистика для вернувшихся пользователей:")
print(returning_users['avg_revenue_per_order'].describe())


# ---
# 
# **Задача 4.2.2.** Сравните распределение по средней выручке с заказа в двух группах пользователей:
# 
# - совершившие 2–4 заказа;
# - совершившие 5 и более заказов.
# 
# Ответьте на вопрос: есть ли различия по значению средней выручки с заказа между пользователями этих двух групп?
# 
# ---
# 

# In[41]:


# Группа 1: Пользователи, совершившие 2–4 заказа
group_2_to_4_orders = user_profile[(user_profile['total_orders'] >= 2) & (user_profile['total_orders'] <= 4)]

# Группа 2: Пользователи, совершившие 5 и более заказов
group_5_plus_orders = user_profile[user_profile['total_orders'] >= 5]


# In[42]:


# Построение гистограмм
plt.figure(figsize=(12, 6))

# Группа 1: Пользователи с 2–4 заказами
plt.hist(group_2_to_4_orders['avg_revenue_per_order'], bins=50, alpha=0.6, label='2–4 заказа', density=True, color='orange')

# Группа 2: Пользователи с 5+ заказами
plt.hist(group_5_plus_orders['avg_revenue_per_order'], bins=50, alpha=0.6, label='5+ заказов', density=True, color='green')

# Настройка графика
plt.title('Распределение средней выручки с заказа для разных групп пользователей', fontsize=14)
plt.xlabel('Средняя выручка с заказа, руб.', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Показываем график
plt.show()


# In[43]:


# Статистика для пользователей с 2–4 заказами
print("Статистика для пользователей с 2–4 заказами:")
print(group_2_to_4_orders['avg_revenue_per_order'].describe())

# Статистика для пользователей с 5+ заказами
print("Статистика для пользователей с 5+ заказами:")
print(group_5_plus_orders['avg_revenue_per_order'].describe())


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Даныне группы пользователей совершают +/- одинаковые заказы.
# </div>

# ---
# 
# **Задача 4.2.3.** Проанализируйте влияние среднего количества билетов в заказе на вероятность повторной покупки.
# 
# - Изучите распределение пользователей по среднему количеству билетов в заказе (`avg_tickets_count`) и опишите основные наблюдения.
# - Разделите пользователей на несколько сегментов по среднему количеству билетов в заказе:
#     - от 1 до 2 билетов;
#     - от 2 до 3 билетов;
#     - от 3 до 5 билетов;
#     - от 5 и более билетов.
# - Для каждого сегмента подсчитайте общее число пользователей и долю пользователей, совершивших повторные заказы.
# - Ответьте на вопросы:
#     - Как распределены пользователи по сегментам — равномерно или сконцентрировано?
#     - Есть ли сегменты с аномально высокой или низкой долей повторных покупок?
# 
# ---

# In[44]:


# Создание сегментов
user_profile['ticket_segment'] = pd.cut(
    user_profile['avg_tickets_per_order'],
    bins=[0, 2, 3, 5, float('inf')],
    labels=['1-2', '2-3', '3-5', '5+']
)


# In[45]:


# Агрегация данных
segment_analysis = (
    user_profile.groupby('ticket_segment')
    .agg(total_users=('user_id', 'count'), returning_users=('is_two', 'sum'))
    .reset_index()
)

# Расчет доли повторных покупок
segment_analysis['returning_share'] = segment_analysis['returning_users'] / segment_analysis['total_users']

# Вывод результатов
print(segment_analysis[['ticket_segment', 'total_users', 'returning_users', 'returning_share']])


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Чаще всего пользователи покупают 2-3 билета, реже всего 5+ билетов
# </div>

# ---
# 
# #### 4.3. Исследование временных характеристик первого заказа и их влияния на повторные покупки
# 
# Изучите временные параметры, связанные с первым заказом пользователей:
# 
# - день недели первой покупки;
# - время с момента первой покупки — лайфтайм;
# - средний интервал между покупками пользователей с повторными заказами.
# 
# ---
# 
# **Задача 4.3.1.** Проанализируйте, как день недели, в которой была совершена первая покупка, влияет на поведение пользователей.
# 
# - По данным даты первого заказа выделите день недели.
# - Для каждого дня недели подсчитайте общее число пользователей и долю пользователей, совершивших повторные заказы. Результаты визуализируйте.
# - Ответьте на вопрос: влияет ли день недели, в которую совершена первая покупка, на вероятность возврата клиента?
# 
# ---
# 

# In[46]:


# Выделение дня недели из first_order_date
user_profile['first_order_weekday'] = user_profile['first_order_date'].dt.dayofweek
# Агрегация данных
weekday_analysis = (
    user_profile.groupby('first_order_weekday')
    .agg(total_users=('user_id', 'count'), returning_users=('is_two', 'sum'))
    .reset_index()
)

# Расчет доли повторных покупок
weekday_analysis['returning_share'] = weekday_analysis['returning_users'] / weekday_analysis['total_users']

# Приведение номеров дней недели к названиям
weekday_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
weekday_analysis['weekday_name'] = weekday_analysis['first_order_weekday'].apply(lambda x: weekday_names[x])
weekday_analysis = weekday_analysis.sort_values(by='returning_share', ascending=False)
# Вывод результатов
print(weekday_analysis[['weekday_name', 'total_users', 'returning_users', 'returning_share']])


# In[47]:



# Построение графика
plt.figure(figsize=(10, 6))
sns.barplot(data=weekday_analysis, x='weekday_name', y='returning_share', palette='viridis')

# Настройка графика
plt.title('Доля возвращающихся пользователей по дням недели первой покупки', fontsize=14)
plt.xlabel('День недели', fontsize=12)
plt.ylabel('Доля возвращающихся пользователей', fontsize=12)
plt.xticks(fontsize=12)
plt.grid(alpha=0.3)

# Показываем график
plt.show()


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Чаще всего возвращаются люди совершившие покувки в сб и пн.
# </div>

# ---
# 
# **Задача 4.3.2.** Изучите, как средний интервал между заказами влияет на удержание клиентов.
# 
# - Рассчитайте среднее время между заказами для двух групп пользователей:
#     - совершившие 2–4 заказа;
#     - совершившие 5 и более заказов.
# - Исследуйте, как средний интервал между заказами влияет на вероятность повторного заказа, и сделайте выводы.
# 
# ---
# 

# In[48]:


# Группа 1: Пользователи, совершившие 2–4 заказа
group_2_to_4_orders = user_profile[(user_profile['total_orders'] >= 2) & (user_profile['total_orders'] <= 4)]

# Группа 2: Пользователи, совершившие 5 и более заказов
group_5_plus_orders = user_profile[user_profile['total_orders'] >= 5]


# In[49]:


# Средний интервал между заказами для группы 2–4 заказа
mean_days_between_orders_2_to_4 = group_2_to_4_orders['avg_days_between_orders'].mean()

# Средний интервал между заказами для группы 5+ заказов
mean_days_between_orders_5_plus = group_5_plus_orders['avg_days_between_orders'].mean()

print(f"Средний интервал между заказами (2–4 заказа): {mean_days_between_orders_2_to_4:.2f} дней")
print(f"Средний интервал между заказами (5+ заказов): {mean_days_between_orders_5_plus:.2f} дней")


# In[50]:



# Построение гистограмм
plt.figure(figsize=(12, 6))

# Группа 1: Пользователи с 2–4 заказами
plt.hist(group_2_to_4_orders['avg_days_between_orders'], bins=30, alpha=0.6, label='2–4 заказа', density=True, color='orange')

# Группа 2: Пользователи с 5+ заказами
plt.hist(group_5_plus_orders['avg_days_between_orders'], bins=30, alpha=0.6, label='5+ заказов', density=True, color='green')

# Настройка графика
plt.title('Распределение среднего интервала между заказами для разных групп пользователей', fontsize=14)
plt.xlabel('Средний интервал между заказами, дней', fontsize=12)
plt.ylabel('Плотность распределения', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Показываем график
plt.show()


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Люди совершающие 2 и более заказов - чащего всего совершаю в тот же день или близлежайшие - возможно докупают билеты на уже приобретенные мероприятия для друзей, родных
# </div>

# ---
# 
# #### 4.4. Корреляционный анализ количества покупок и признаков пользователя
# 
# Изучите, какие характеристики первого заказа и профиля пользователя могут быть связаны с числом покупок. Для этого используйте универсальный коэффициент корреляции `phi_k`, который позволяет анализировать как числовые, так и категориальные признаки.
# 
# ---
# 
# **Задача 4.4.1:** Проведите корреляционный анализ:
# - Рассчитайте коэффициент корреляции `phi_k` между признаками профиля пользователя и числом заказов (`total_orders`). При необходимости используйте параметр `interval_cols` для определения интервальных данных.
# - Проанализируйте полученные результаты. Если полученные значения будут близки к нулю, проверьте разброс данных в `total_orders`. Такое возможно, когда в данных преобладает одно значение: в таком случае корреляционный анализ может показать отсутствие связей. Чтобы этого избежать, выделите сегменты пользователей по полю `total_orders`, а затем повторите корреляционный анализ. Выделите такие сегменты:
#     - 1 заказ;
#     - от 2 до 4 заказов;
#     - от 5 и выше.
# - Визуализируйте результат корреляции с помощью тепловой карты.
# - Ответьте на вопрос: какие признаки наиболее связаны с количеством заказов?
# 
# ---

# In[51]:


# Рассчет матрицы корреляции Пирсона
correlation_matrix = user_profile[['total_orders', 'avg_revenue_per_order', 'avg_tickets_per_order', 'avg_days_between_orders']].corr()


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Матрица корреляции Пирсона', fontsize=14)
plt.show()


# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Исходя из матрицы корреляции мы видим что наблюдается сильная связь между средней стоимостью покупки и средним кол-м билетов.
# </div>

# In[ ]:





# ### 5. Общий вывод и рекомендации
# 
# В конце проекта напишите общий вывод и рекомендации: расскажите заказчику, на что нужно обратить внимание. В выводах кратко укажите:
# 
# - **Информацию о данных**, с которыми вы работали, и то, как они были подготовлены: например, расскажите о фильтрации данных, переводе тенге в рубли, фильтрации выбросов.
# - **Основные результаты анализа.** Например, укажите:
#     - Сколько пользователей в выборке? Как распределены пользователи по числу заказов? Какие ещё статистические показатели вы подсчитали важным во время изучения данных?
#     - Какие признаки первого заказа связаны с возвратом пользователей?
#     - Как связаны средняя выручка и количество билетов в заказе с вероятностью повторных покупок?
#     - Какие временные характеристики влияют на удержание (день недели, интервалы между покупками)?
#     - Какие характеристики первого заказа и профиля пользователя могут быть связаны с числом покупок согласно результатам корреляционного анализа?
# - Дополните выводы информацией, которая покажется вам важной и интересной. Следите за общим объёмом выводов — они должны быть компактными и ёмкими.
# 
# В конце предложите заказчику рекомендации о том, как именно действовать в его ситуации. Например, укажите, на какие сегменты пользователей стоит обратить внимание в первую очередь, а какие нуждаются в дополнительных маркетинговых усилиях.

# <div class="alert alert-info">
# <b>Комментарий студента:</b>
#     В  результате проведенного анализа всего в выборке:
#     
#     Общее число пользователей: 21933
#     
#     Средняя выручка с одного заказа: 574.02 руб.
#     
#     Доля пользователей, совершивших 2 и более заказа: 61.82%
#     
#     Доля пользователей, совершивших 5 и более заказов: 29.18%
#     
#     Большинство пользователей начинает свое взаимодействие с Концертов и Другое, 44 и 24% соответсвено. Они и являются точкой входа.
#     
#     Первый заказ чаще делают с мобильных устройств.
#     
#     Хедлайнер по продаже первых билетов: Билеты без пролем - 23,8%
#     
#     Средний интервал между заказами (2–4 заказа): 21.33 дней
#     
#     Средний интервал между заказами (5+ заказов): 9.64 дней - скорее всего связанов с вязи с докупание доп билетов для родственников и/или друзей.
#     
#     наблюдается сильная корреляия между кол-вом билетов и стоимостью заказ, что и логично.
# <br>
# </div>

# <div class="alert alert-info">
# <b>Комментарий студента:</b>
# <br>Заказчику стоит обратить внимание на пользователей покупающих в субботу и пн, с доп. маркетинговой активностью по предложению билетов друзьям, родственникам.
# Особенно из категорий: Театр, выставки концерты.
# </div>

# In[ ]:





# ### 6. Финализация проекта и публикация в Git
# 
# Когда вы закончите анализировать данные, оформите проект, а затем опубликуйте его.
# 
# Выполните следующие действия:
# 
# 1. Создайте файл `.gitignore`. Добавьте в него все временные и чувствительные файлы, которые не должны попасть в репозиторий.
# 2. Сформируйте файл `requirements.txt`. Зафиксируйте все библиотеки, которые вы использовали в проекте.
# 3. Вынести все чувствительные данные (параметры подключения к базе) в `.env`файл.
# 4. Проверьте, что проект запускается и воспроизводим.
# 5. Загрузите проект в публичный репозиторий — например, на GitHub. Убедитесь, что все нужные файлы находятся в репозитории, исключая те, что в `.gitignore`. Ссылка на репозиторий понадобится для отправки проекта на проверку. Вставьте её в шаблон проекта в тетрадке Jupyter Notebook перед отправкой проекта на ревью.

# **Вставьте ссылку на проект в этой ячейке тетрадки перед отправкой проекта на ревью.**
