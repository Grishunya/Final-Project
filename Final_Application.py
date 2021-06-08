import pandas as pd
import numpy as np
import pygame
from pygame import mixer
import streamlit as st
import requests
from bs4 import BeautifulSoup
import folium
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from streamlit_folium import folium_static
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import time
Part = st.sidebar.selectbox('Select the parts one by one:', ('Part 0', 'Parts 1-2', 'Parts 3-4', 'Part 5', 'Part 6'))
if Part == 'Part 0':
    pygame.mixer.init()
    st.write("### While working with this project, you'll be able to find some information about your "
             "body advice on how you should treat it depending on some information provided by you. "
             "These advice are completely optional, I don't insist on you changing your life the way shown below. "
             "You should love yourself the way you look and feel :)" )
    Quest = st.selectbox("Want to get the right mood?", ('No', 'Yes'))
    if Quest == 'Yes':
        mixer.music.load("Music.mp3")
        mixer.music.play(-1)
        st.write("##### If you get tired of the music, change the option back to 'No'")
    else:
        mixer.music.stop()
if Part == 'Parts 1-2':
    weight = int(st.number_input('Enter your weight (in kg): ', ))
    height = int(st.number_input('Enter your height (in cm): ', ))
    st.title('Part 1: Predictions')
    with st.echo(code_location='below'):
        Data_ML = pd.read_csv('weight-height.csv')
        st.write('##### Take a quick look on the data:')
        st.write(Data_ML.head())
    st.write("#### Now, let's change the data for genders into a numerical type:")
    with st.echo(code_location='below'):
        Data_ML['Gender'].replace('Female',0, inplace=True)
        Data_ML['Gender'].replace('Male',1, inplace=True)
        Params = Data_ML[['Height', 'Weight']].values
        Genders = Data_ML['Gender'].values
        Params_train, Params_test, Genders_train, Genders_test = train_test_split(Params, Genders)
        lin_reg = LinearRegression().fit(Params_train, Genders_train)
        lin_pred = lin_reg.predict(Params_test)
        my_gender = lin_reg.predict([[height * 0.393701, weight * 2.20462]])
    st.write("### Let's test it out:")
    with st.echo(code_location='below'):
        def gender_reveal(prediction):
            if abs(prediction - 0) > abs(prediction - 1):
                return 'male'
            else:
                return 'female'
        st.write('According to my prediction, you are a', gender_reveal(my_gender))
    check_p = st.selectbox('Please, tell me, if the prediction is correct', ('Yes', 'No'))
    with st.echo(code_location='below'):
        if check_p == 'Yes':
            st.write('Great! It has worked for you correctly')
        else:
            st.write('Probably, there was not enough information to predict your gender precisely enough.')
            st.write('You seem to be different from thousands of people, which is amazing :)')
            if gender_reveal(my_gender) == 'female':
                my_gender = 1
            else:
                my_gender = 0
    st.write('### This is the end of the 1st part')
    Ticker = st.checkbox('Open part 2', )
    if Ticker:
        st.write('## Part 2: Websites')
        st.write("#### This part contains working with selenium, which requires using a browser you work with. "
             "I have been working with Safari, but in case you have another one, change it in the code, "
             "or specify the path. You may also be asked to allow remote automation for the code to run. "
             "I have written similar messages when the package is used, but it's better to be aware of these "
             "problems in advance.")
        Gender = gender_reveal(my_gender)
        age = 0
        with st.echo(code_location='below'):
            age = int(st.number_input('Enter your age:', ))
            Info = [height, weight, age, Gender]
            Outcomes = ['Underweight', 'Healthy weight', 'Overweight', 'Obese']
            BMI = round(weight / ((height / 100) ** 2), 1)
            if Info[3] == 'female':
                url = 'https://www.calculator.net/calorie-calculator.html?csex=f'
            else:
                url = 'https://www.calculator.net/calorie-calculator.html?csex=m'
            if BMI < 18.5:
                State = 'Underweight'
            elif BMI >= 18.5 and BMI <= 24.9:
                State = 'Healthy weight'
            elif BMI >= 25 and BMI <= 29.9:
                State = 'Overweight'
            else:
                State = 'Obese'
            No = Outcomes.index(State)
        st.write('##### If you are using a browser other than Safari, change it in the next cell')
        checker = 0
        if age != 0:
            with st.echo(code_location='below'):
                driver = webdriver.Safari()
                driver.get('http://bmijs.is.tuebingen.mpg.de/body_masses/generate_bmi?utf8=✓&locale=en&body_mass%5Bunit_measurment%5D=M&body_mass%5Bgender%5D=' + Gender.capitalize() + '&body_mass%5Bage%5D=' + str(age) + '&body_mass%5Bheight%5D=' + str(height/100) + '&body_mass%5Bfeet%5D=&body_mass%5Binches%5D=&body_mass%5Bweight%5D=' + str(weight) + '&commit=Calculate+BMI')
                time.sleep(5) # The page has some animation to load
                driver.save_screenshot('body.png')
                driver.quit()
                st.write('Your BMI is', BMI)
                st.write('This is how your body is predicted to look like:')
                st.image(Image.open('body.png'))
                if No == 0:
                    st.write('You are', str.lower(State) + ". Let's work on gaining weight plans.")
                elif No == 1:
                    st.write('You are a', str.lower(State) + ". Let's work on maintaining weight plans.")
                else:
                    st.write('You are', str.lower(State) + ". Let's work on losing weight plans.")

            st.write("### Let's get deeper and find what's your daily calorie intake value should be:")
            checker = 1
            # If you are using a browser other than Safari, change it in the next cell
            if checker == 1:
                with st.echo(code_location='below'):
                    driver = webdriver.Safari()
                    driver.get(url)
                    element = driver.find_element_by_name('cheightmeter')
                    element.send_keys(str(Info[0]))
                    element = driver.find_element_by_name('cage')
                    element.send_keys(str(Info[2]))
                    element = driver.find_element_by_name('ckg')
                    element.send_keys(str(Info[1]) + Keys.RETURN)
                    try:
                        elem = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "result_box")))
                    finally:
                        content = driver.page_source
                        driver.quit()
                    # I hope that you will be able to tune the browser so that the code runs correctly,
                    # but in case it does not, uncomment run the following cell, and ignore the one after:
                    # Calories = 2000
                    all_plans = BeautifulSoup(content).find_all('b')[0:7]
                    if No == 0:
                        Calories = all_plans[5].text
                    elif No == 1:
                        Calories = all_plans[0].text
                    elif No == 2:
                        Calories = all_plans[1].text
                    else:
                        Calories = all_plans[2].text
                    st.write('To achieve the set goal, you should consume', Calories, 'calories daily')
                    st.write('Please, remember this value. it will be asked in the following part')
                st.write('### This is the end of the 2nd part')
if Part == 'Parts 3-4':
    Calories = st.number_input('Please, specify the amount of calories you were recommended:', )
    st.write("###### I couldn't get the value from the previous page because multi-page"
             " working with streamlit doesn't allow it" )
    st.write('## Part 3: Tables')
    with st.echo(code_location='below'):
        Meals = pd.read_csv('Recipes.csv')
        # Data retrieved from https://www.kaggle.com/hugodarwood/epirecipes
        st.write(Meals.sample(2))
    st.write("#### Let's get rid of the most columns that we won't need:")
    with st.echo(code_location='below'):
        Meals = Meals[['title', 'rating', 'calories', 'breakfast', 'lunch', 'dinner', 'snack', 'fat', 'protein']]
        st.write(Meals.sample(3))
    with st.echo(code_location='below'):
        Recipes = pd.read_json('Recipes.json')
        # Data retrieved from https://www.kaggle.com/hugodarwood/epirecipes
        st.write(Recipes.sample(2))
    st.write("#### Let's get rid of the most columns that we won't need:")
    with st.echo(code_location='below'):
        Recipes = Recipes[['title', 'directions', 'ingredients']]
    st.write(Recipes.sample(2))
    with st.echo(code_location='below'):
        Meals = Meals.set_index('title').join(Recipes.set_index('title'))
        Breakfasts = Meals[Meals['breakfast'] == 1.0][['rating', 'calories', 'fat', 'protein', 'directions', 'ingredients']].dropna().query('calories < 1500')
        Lunches = Meals[Meals['lunch'] == 1.0][['rating', 'calories', 'fat', 'protein', 'directions', 'ingredients']].dropna().query('calories < 1500')
        Dinners = Meals[Meals['dinner'] == 1.0][['rating', 'calories', 'fat', 'protein', 'directions', 'ingredients']].dropna().query('calories < 1500')
        Snacks = Meals[Meals['snack'] == 1.0][['rating', 'calories', 'fat', 'protein', 'directions', 'ingredients']].dropna().query('calories < 1500')
    st.write("#### Now, let's select only the best, verified recipes:")
    with st.echo(code_location='below'):
        Lunches = Lunches.reset_index()
        Breakfasts = Breakfasts.reset_index()
        Dinners = Dinners.reset_index()
        Snacks = Snacks.reset_index()
        Lunches = Lunches.where(Lunches['rating'] >= 3.0).dropna()
        Breakfasts = Breakfasts.where(Breakfasts['rating'] >= 3.0).dropna()
        Dinners = Dinners.where(Dinners['rating'] >= 3.0).dropna()
        Snacks = Snacks.where(Snacks['rating'] >= 3.0).dropna()
        st.write(Dinners.sample(2))
    st.write("#### Now, let's make a random meal plan for you:")
    with st.echo(code_location='below'):
        meals = [Breakfasts.sample(1), Lunches.sample(1), Dinners.sample(1), Snacks.sample(1)]
        Plan = pd.concat(meals)
        Plan = Plan.reset_index()
        factor = (int(Calories)/Plan.sum()['calories']).round(1)
        st.write(Plan)
    st.write("### You have seen a meal plan that may work for you. "
                         "If you don't like the meals, you may rerun the cells above and get another list of foods "
             "(When working with code itself), or reload the page.")
    with st.echo(code_location='below'):
        st.write("### Creating a file with your meals:")
        text = "To satisfy your recommended calorie intake amount, multiply the portions by " + str(factor)
        f = open("Meal plan.txt", "a")
        f.truncate(0)
        f.write(text)
        for index, row in Plan.iterrows():
            f.write('\n')
            f.write('\n')
            f.write('\n')
            text = 'Meal ' + str(index + 1) + ':\n'
            f.writelines(text)
            f.write('\nIngredients:\n')
            text = '\n'.join(Plan.iloc[index]['ingredients'])
            f.writelines(text)
            f.write('\n')
            f.write('\n')
            text = '\n'.join(Plan.iloc[index]['directions'])
            f.writelines(text)
            f.write('\n')
            for i in range(3, 6):
                text = '\n' + Plan.columns.values.tolist()[i].capitalize() + ' per portion: ' + str(Plan.iloc[index][i])
                f.writelines(text)
        f.close()
    with st.echo(code_location='below'):
        st.write("### Since I have decided to create this project using streamlit,"
                         "the file may be unreachable to you. "
                         "That's why I have decided to write it here line by line for you:"
                         ""
                         "")
        R = open('Meal plan.txt')
        for line in R:
            st.write(line)
    st.write('### This is the end of the 3rd part')
    Ticker = st.checkbox('Open part 4', )
    if Ticker:
        st.write('## Part 4: R')
        with st.echo(code_location='below'):
            Breakfasts = Breakfasts[['title', 'calories', 'rating', 'protein', 'fat']].groupby('rating').mean().reset_index()
            Lunches = Lunches[['title', 'calories', 'rating', 'protein', 'fat']].groupby('rating').mean().reset_index()
            Dinners = Dinners[['title', 'calories', 'rating', 'protein', 'fat']].groupby('rating').mean().reset_index()
        st.write("### I have decided to plot some graphs to get an idea of preferences of the people "
                 "who have rated the recipes provided.")
        st.write("### Let's create a file we'll need:")
        with st.echo(code_location='below'):
            Df = pd.concat([Breakfasts, Lunches, Dinners])
            st.write(Df.sample(2))
            with open (Path(f"R") / "Data.csv", "w") as f:
                f.write(Df.to_csv())
        st.write("### Working with R in this notebook was challenging. "
                 "Eventually, I have decided to create a separate file, which includes working "
                 "with ggplot2, as well as a ggplot2 extension named patchwork (to plot on the same graph easily) "
                 "and tibble for making tables, which is a part of tidyverse. You can find it in the 'R' folder")

        st.image('R/myplot.png')
        st.write("Here's the code I've ran in R:")
        st.write("\n"
                 "```{r}"
                 "\n"
                 "library(patchwork)"
                 "\n"
                 "library(ggplot2)"
                 "\n"
                 "library(tibble)"
                 "\n"
                 "Data = read.csv('Data.csv')"
                 "\n"
                 "Tibble_data = as_tibble(Data)"
                 "\n"
                 "values = c('Breakfast', 'Breakfast', 'Breakfast', 'Breakfast', 'Lunch', 'Lunch', 'Lunch', 'Lunch', 'Dinner', 'Dinner', 'Dinner', 'Dinner')"
                 "\n"
                 "```"
                 "\n"
                 "```{r}"
                 "\n"
                 "Tibble_data <- Tibble_data %>% add_column(Meal = values)"
                 "\n"
                 "```"
                 "\n"
                 "```{r}"
                 "\n"
                 "p1 <- ggplot(Tibble_data, aes(x = rating, y = fat)) + geom_smooth(method = 'loess' , fill = 'lightblue', formula = 'y ~ x')"
                 "\n"
                 "p2 <- ggplot(Tibble_data, aes(x = rating, y = calories, fill = Meal)) + geom_boxplot(size = 0.25) + theme(legend.position='none')" 
                 "\n"
                 "p3 <- ggplot(Tibble_data, aes(x = rating, y = protein, fill = Meal)) + geom_col(width = 0.5)  + coord_polar(theta = 'y')" 
                 "\n"
                 "myplot <- p1 / (p2 | p3)"
                 "\n"
                 "```"
                 "\n"
                 "```{r}"
                 "\n"
                 "png('myplot.png')"
                 "\n"
                 "print(myplot)"
                 "\n"
                 "dev.off()"
                 "\n"
                 "```")
        st.write('### This is the end of the 4th part')
if Part == 'Part 5':
    st.write('## Part 5: API & GeoData')
    st.write("Let's create a map of fitness centers in Moscow")
    st.write("#### Wait a little for the data to load")
    with st.echo(code_location='below'):
        addresses = []
        names = []
        url = "https://sportgyms.ru/moscow/page/"
        for x in range(1,51):
            web = url + str(x)
            r = requests.get(web)
            s = BeautifulSoup(r.text)
            adrs = s.find_all('div', {'class' : 'ciAdress'})
            nams = s.find_all('div', {'class' : 'clubWrap clubWrapCat'})
            for i in range(len(adrs)):
                addresses.append(adrs[i].text[7:])
                names.append(nams[i].find('h2').text)
        Gyms = pd.DataFrame({'Name' : names, 'Address' : addresses})
    st.write("Here's an example on how the data looks like:")
    with st.echo(code_location='below'):
        st.write(Gyms.sample(2))
    st.write("Working with API takes a lot of time. That's why I have decided to prepare"
             " the required data in advance. I have also attached the code that I have been using "
             "for creating the dataset (As comments). You can go check that it works perfectly fine if tou wish,"
             " but it takes approx. 10 min to run it. You will also need to get an API key on the "
             "https://positionstack.com website and use it as API_Key variable. It is completely free.")
    API_Key = 'd2734b66ff7e0bbf2f1e7f4c81298946'
    with st.echo(code_location='below'):
        # Lon = []
        # Lat = []
        # for i in range(len(Gyms)):
            # text = Gyms.iloc[i]['Address']
            # resp = requests.get('http://api.positionstack.com/v1/forward?access_key=' + API_Key + '&query=Москва,' + text)
            # resp_json_payload = resp.json()
            # try:
                # lng = resp_json_payload['data'][0]['longitude']
                # Lon.append(lng)
                # lat = resp_json_payload['data'][0]['latitude']
                # Lat.append(lat)
            # except:
                # Lon.append(None)
                # Lat.append(None)
        # Geos = pd.DataFrame({'lon' : Lon, 'lat' : Lat})
        # Geos.to_csv('Geos.csv')
        Geos = pd.read_csv('Geos.csv')
        Gyms = Gyms.join(Geos)
        Gyms = Gyms.dropna()
        Gyms = Gyms[['lon', 'lat']].drop_duplicates()
        Gyms = Gyms[Gyms['lon'] <= 40].merge(Gyms[Gyms['lat'] >= 52.5]).merge(Gyms[Gyms['lat'] <= 57.5]).merge(Gyms[Gyms['lon'] >= 35])
        Map = folium.Map([55.75215, 37.61819], zoom_start=10)
        for ind, row in Gyms.iterrows():
            folium.Circle([row.lat, row.lon], radius=1, color = 'blue').add_to(Map)
        folium_static(Map)
    st.write('### This is the end of the 5th part')
if Part == 'Part 6':
    st.write('## PART 6: SQL, Numpy & Visualisation')
    st.write("### The next thing that may be useful is the data about different activities. "
             "So, let's get it structured:")
    with st.echo(code_location='below'):
        Exercise = pd.read_csv('exercise.csv')
        st.write(Exercise.sample(2))
    with st.echo(code_location='below'):
        ### FROM (https://gist.github.com/ischurov/a40be845fa91da6b0bb4a26209636180)
        conn = sqlite3.connect("database.sqlite")
        c = conn.cursor()
        conn.commit()
        def sql(request):
            return pd.read_sql_query(request, conn)
        ### END FROM
        Exercise.to_sql("Acts", conn)
        Visual_df = sql('''
        SELECT "Activity, Exercise or Sport (1 hour)" AS "Activity (1 hr)", "Calories per kg" FROM Acts
        WHERE "Calories per kg" IS NOT NULL
        ORDER BY "Calories per kg" DESC
        ''')
        st.write(Visual_df)
    st.write("### As you can see, there are too many entries, so let's divide the data into a few dataframes")
    with st.echo(code_location='below'):
        Visual_df.to_sql("Vd", conn)
        first_df = sql('''
        SELECT * FROM Vd
        WHERE "Calories per kg" >= 2.5
        ''')
        second_df = sql('''
        SELECT * FROM Vd
        WHERE "Calories per kg" <= 2.5 AND "Calories per kg" >= 1.5
        ''')
        third_df = sql('''
        SELECT * FROM Vd
        WHERE "Calories per kg" <= 1.5 AND "Calories per kg" >= 1.0
        ''')
        fourth_df = sql('''
        SELECT * FROM Vd
        WHERE "Calories per kg" <= 1.0
        ''')
    st.write("#### Let's visualise the dataframes one be one:")
    with st.echo(code_location='below'):
        plt.bar(first_df['Calories per kg'], first_df['Activity (1 hr)'], width = 0.75, color = 'lightblue')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plt.show())
    with st.echo(code_location='below'):
        sns.kdeplot(second_df['Calories per kg'], shade=True)
        st.pyplot()
    with st.echo(code_location='below'):
        ### Partly taken from https://www.data-to-viz.com/graph/circularbarplot.html
        plt.figure(figsize = (10, 7.5))
        ax = plt.subplot(111, polar = True)
        plt.axis('off')
        upperLimit = 50
        lowerLimit = 40
        max_value = third_df['Calories per kg'].max()
        slope = (max_value - lowerLimit) / max_value
        heights = slope * third_df['Calories per kg'] + lowerLimit
        width = 2 * np.pi / len(third_df.index)
        indexes = list(range(1, len(third_df.index)+1))
        angles = [element * width for element in indexes]
        bars = ax.bar(x = angles, height = heights, width = width, bottom = lowerLimit, linewidth = 2,
                      edgecolor = "white", color = "mediumpurple")
        for bar, angle, height, label in zip(bars, angles, heights, third_df["Activity (1 hr)"]):
            rotation = np.rad2deg(angle)
            alignment = ""
            if angle >= np.pi / 2 and angle < 3 * np.pi/2:
                alignment = "right"
                rotation = rotation + 180
            else:
                alignment = "left"
            ax.text(x = angle, y = lowerLimit + bar.get_height() + 3, s = label, ha = alignment,
                va = 'center', rotation = rotation, rotation_mode = "anchor")
        st.pyplot(plt.show())
    with st.echo(code_location='below'):
        fig = go.Figure(go.Bar(x = fourth_df['Calories per kg'], y = fourth_df['Activity (1 hr)'], orientation='h'))
        st.write(fig)