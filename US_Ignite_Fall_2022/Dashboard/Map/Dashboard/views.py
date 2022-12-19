from django.shortcuts import render
from django.http import HttpResponse
import pickle
import numpy as np
import html
import requests
import datetime

AMBER = '#FFBF00'
RED = '#FF0000'
GREEN = '#00FF00'
BLACK = '#000000'

color_list24 = [RED, AMBER, BLACK, RED, RED, GREEN, GREEN, GREEN, GREEN, AMBER, RED, GREEN, GREEN, AMBER, GREEN, GREEN,
                GREEN, GREEN, RED, GREEN]
color_list48 = [AMBER, BLACK, GREEN, AMBER, RED, GREEN, AMBER, GREEN, GREEN, BLACK, GREEN, AMBER, GREEN, RED, GREEN,
                GREEN, RED, GREEN, GREEN, GREEN]
color_list72 = [AMBER, BLACK, GREEN, GREEN, RED, GREEN, RED, GREEN, GREEN, RED, GREEN, GREEN, GREEN, RED, GREEN, GREEN,
                GREEN, GREEN, GREEN, RED]

street_mapping = {'Fontaine Blvd': 6,'Mesa Ridge Pkwy': 10,'E Ohio Ave': 5,'S Nevada Ave': 14,'S Academy Blvd': 13,'SH-115': 15,
'Lake Ave': 8,'Norad Rd': 12,'E Las Vegas St': 4, 'Barkeley Ave': 0,'Magrath Ave': 9,'Fountain Mesa Rd': 7,'Chiles Ave': 3,
'Nelson Blvd': 11,'Broadmoor Bluffs Dr': 1,'Venetucci Blvd': 16,'Charter Oak Ranch Rd': 2,'Westmeadow Dr': 17}

weather_mapping = {'Clear': 0,'Clouds': 1,'Drizzle': 2,'Rain': 7,'Mist': 6,'Snow': 9,'Fog': 4,'Thunderstorm': 11,'Squall': 10,'Haze': 5,'Smoke': 8,'Dust': 3}

key = '966b218f226c901278d784736dd591bf'
url = 'https://api.openweathermap.org/data/2.5/forecast?lat={}&lon={}&appid={}'.format('38.8339', '-104.8214', key)


def dashboard24(request):
    context = {}

    context['color'] = [html.escape(s) for s in color_list24]

    return render(request, 'dashboard.html', context)


def dashboard48(request):
    context = {}

    context['color'] = [html.escape(s) for s in color_list48]

    return render(request, 'dashboard.html', context)


def dashboard72(request):
    context = {}

    context['color'] = [html.escape(s) for s in color_list72]

    return render(request, 'dashboard.html', context)


def download_file(request):
    # Open the file using the open() function
    with open('Prediction.csv', 'rb') as f:
        # Read the contents of the file
        file_data = f.read()

    # Create a HttpResponse object
    response = HttpResponse(file_data, content_type='text/plain')

    # Set the content-disposition header to indicate that the file should be downloaded
    response['Content-Disposition'] = 'attachment; filename="Prediction.csv"'

    # Return the response object
    return response


def update(request):
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Implement Prediction Here
    # Use Threshold to Generate BRAG Value
    record_db = pd.DataFrame(columns=['street','00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00'])
    res = requests.get(url)
    data = res.json()
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)

    weather_every3 = []
    values = data['list']
    for item in values:
      if(tomorrow.strftime("%Y-%m-%d") in item['dt_txt']):
        each_record = []
        #each_record.append(item['dt_txt'])
        each_record.append(item['main']['temp'])
        each_record.append(item['main']['pressure'])
        each_record.append(item['main']['humidity'])
        each_record.append(item['wind']['speed'])
        each_record.append(item['wind']['deg'])
        each_record.append(item['weather'][0]['main'])
        weather_every3.append(each_record)

    for each_slot in weather_every3:
      each_slot[5] = weather_mapping[each_slot[5]]

    predict_records = []
    '''
      Following code snippet to generate mean values of the jam level being taken into consideration
      ###
        new_df = df[['street', 'level']].copy()
        new_df.columns = ['street', 'level']
        new_df.groupby(['street']).mean()
      ###
    '''
    average_jam_level = [2.860915,2.860915,2.738095,2.983193,2.699341,3.000000,3.385214,3.069444,3.156332,3.009276,3.433259,2.967448,3.091623,2.000000,2.951084,2.864466,1.898548,2.913121,3.260163]

    '''
      Input parameters for prediction of BRAG score for each street:
      ###
        temp
        pressure
        humidity
        wind_speed
        wind_deg
        weather_main
        street
        level
        month
        week of the year
        day of the week
        day of the month
        hour of the day
      ###
    '''

    for i in range(0,18):
      k=0
      temp=[]
      for j in range(0,8):
        #print(each_slot)
        temp = weather_every3[j].copy()
        temp+=([i,average_jam_level[i],tomorrow.month,tomorrow.isocalendar()[1],tomorrow.weekday(),tomorrow.day,k])
        #print(temp)
        predict_records.append(temp)
        k=k+3

    all_predictions = []
    for i in range(0,len(predict_records)):
      BRAG_prediction = model.predict_proba(np.array(predict_records[i]).reshape(1,-1))[0][1]
      all_predictions.append(BRAG_prediction)

    new_record=[]
    new_record.append(list(street_mapping.keys())[list(street_mapping.values()).index(0)])
    new_record.append(all_predictions[0])
    for i in range(1,len(all_predictions)):
      if(i%8!=0):
        new_record.append(all_predictions[i])
      if(i%8==0):
        #print(new_record)
        record_db.loc[len(record_db)] = new_record
        new_record=[]
        new_record.append(list(street_mapping.keys())[list(street_mapping.values()).index(predict_records[i][6])])
        new_record.append(all_predictions[i])

    record_db['00:00','03:00','06:00,'09:00','12:00','15:00','18:00','21:00'] = record_db['00:00','03:00','06:00,'09:00','12:00','15:00','18:00','21:00'].apply(threshold)
    #record_db contains the updates BRAG scores for all the streets for the next 24 hours.

    record_db.to_csv('Predictions.csv', encoding='utf-8', index=False)


def threshold(score):
    if score >= 0.9:
        return "BLACK"
    elif 0.75 <= score < 0.9:
        return "RED"
    elif 0.5 <= score < 0.75:
        return "AMBER"
    else:
        return "GREEN"
