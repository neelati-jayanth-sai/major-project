
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

# Create your views here.
from Remote_User.models import ClientRegister_Model,energy_economy_prediction,detection_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            detection_accuracy.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def View_Prediction_Of_Energy_Economy_Type_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'Low'
    print(kword)
    obj = energy_economy_prediction.objects.all().filter(Q(Prediction=kword))
    obj1 = energy_economy_prediction.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio12 = ""
    kword12 = 'High'
    print(kword12)
    obj12 = energy_economy_prediction.objects.all().filter(Q(Prediction=kword12))
    obj112 = energy_economy_prediction.objects.all()
    count12 = obj12.count();
    count112 = obj112.count();
    ratio12 = (count12 / count112) * 100
    if ratio12 != 0:
        detection_ratio.objects.create(names=kword12, ratio=ratio12)


    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_Prediction_Of_Energy_Economy_Type_Ratio.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def View_Prediction_Of_Energy_Economy_Type(request):
    obj =energy_economy_prediction.objects.all()
    return render(request, 'SProvider/View_Prediction_Of_Energy_Economy_Type.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Predicted_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Predicted_Datasets.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = energy_economy_prediction.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:

        row_num = row_num + 1

        ws.write(row_num, 0, my_row.Fid, font_style)
        ws.write(row_num, 1, my_row.Brand, font_style)
        ws.write(row_num, 2, my_row.Model, font_style)
        ws.write(row_num, 3, my_row.AccelSec, font_style)
        ws.write(row_num, 4, my_row.TopSpeed_KmH, font_style)
        ws.write(row_num, 5, my_row.Range_Km, font_style)
        ws.write(row_num, 6, my_row.Efficiency_WhKm, font_style)
        ws.write(row_num, 7, my_row.FastCharge_KmH, font_style)
        ws.write(row_num, 8, my_row.RapidCharge, font_style)
        ws.write(row_num, 9, my_row.PowerTrain, font_style)
        ws.write(row_num, 10, my_row.PlugType, font_style)
        ws.write(row_num, 11, my_row.BodyStyle, font_style)
        ws.write(row_num, 12, my_row.Segment, font_style)
        ws.write(row_num, 13, my_row.Charging_Price, font_style)
        ws.write(row_num, 14, my_row.Prediction, font_style)

    wb.save(response)
    return response

def train_model(request):
    detection_accuracy.objects.all().delete()
    df = pd.read_csv('dataset.csv',encoding='utf-8')
    print(df)
    
    X = df[['Battery Capacity (kWh)', 'Base Energy Consumption (kWh/km)', 'Speed (km/h)', 'Passengers', 'Temperature (°C)', 'Terrain']]
    y = df['Predicted Range (km)']

    print(X)
    print(y)
    models = []
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    from sklearn.ensemble import RandomForestRegressor
    print("Random Forest Regressor")
    RFR = RandomForestRegressor(random_state=42)
    RFR.fit(X_train, y_train)
    predictions = RFR.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    accuracy = r2_score(y_test, predictions)
    print("Random Forest Regressor:")
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    #print(confusion_matrix(y_test, predictions))
    models.append(('Random_Forest_Regressor',RFR ))
    detection_accuracy.objects.create(names="Random Forest Regressor", ratio=accuracy*100)
    
    #Support Vector Regressor
    from sklearn.svm import SVR
    svr = SVR()  
    svr.fit(X_train, y_train)
    svrPrediction = svr.predict(X_test)
    mse = mean_squared_error(y_test, svrPrediction)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, svrPrediction)
    print("Support Vector Regressor:")
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    #print(confusion_matrix(y_test, svrPrediction))
    accuracy = r2_score(y_test, svrPrediction)
    models.append(('Support_Vector_Regressor', svr))
    detection_accuracy.objects.create(names="Support Vector Regressor", ratio=accuracy*1000)
    
    #DecisionTreeRegressor
    from sklearn.tree import DecisionTreeRegressor
    DTR = DecisionTreeRegressor(random_state=42)
    DTR.fit(X_train, y_train)
    decisionTreeRegressor = DTR.predict(X_test)
    mse = mean_squared_error(y_test, decisionTreeRegressor)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, decisionTreeRegressor)
    print("Decision Tree Regressor:")
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    #print(confusion_matrix(y_test, decisionTreeRegressor))
    accuracy = r2_score(y_test, decisionTreeRegressor)
    models.append(('Decision_Tree_Regressor', DTR))
    detection_accuracy.objects.create(names="Decision Tree Regressor", ratio=accuracy*100)
    
    #GradientBoostingRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print("Gradient Boosting Regressor:")
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    accuracy = r2_score(y_test, predictions)
    models.append(('Gradient_Boosting_Regressor', model))
    detection_accuracy.objects.create(names="Gradient Boosting Regressor", ratio=accuracy*100)
    
    
    csv_format = 'Results.csv'
    df.to_csv(csv_format, index=False)
    df.to_markdown
    obj = detection_accuracy.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})