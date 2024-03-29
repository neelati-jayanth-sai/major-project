from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,energy_economy_prediction,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Energy_Economy_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            Bus_Name = request.POST.get('Bus_Name')
            Company = request.POST.get('Company')
            Battery_Capacity_kWh = float(request.POST.get('Battery_Capacity_kWh'))
            Base_Energy_Consumption_kWh_km = float(request.POST.get('Base_Energy_Consumption_kWh_km'))
            Speed_km_h = float(request.POST.get('Speed_km_h'))
            Passengers = float(request.POST.get('Passengers'))
            Temperature_C = float(request.POST.get('Temperature_C'))
            Terrain = float(request.POST.get('Terrain'))
            Standard_Range_km = float(request.POST.get('Standard_Range_km'))

        x_input=[[Battery_Capacity_kWh, Base_Energy_Consumption_kWh_km, Speed_km_h, Passengers, Temperature_C ,  Terrain ]]
        df = pd.read_csv('dataset.csv',encoding='utf-8')
        X = df[['Battery Capacity (kWh)', 'Base Energy Consumption (kWh/km)', 'Speed (km/h)', 'Passengers', 'Temperature (°C)', 'Terrain']]
        y = df['Predicted Range (km)']

        print(X)
        print(y)
        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        from sklearn.tree import DecisionTreeRegressor
        DTR = DecisionTreeRegressor(random_state=42)
        DTR.fit(X_train, y_train)
        decisionTreeRegressor = DTR.predict(x_input)
        print("decisionTreeRegressor",decisionTreeRegressor)

        energy_economy_prediction.objects.create(
        Fid="",
        Brand=Company,
        Model=Bus_Name,
        AccelSec="0.0",
        TopSpeed_KmH=Speed_km_h,
        Range_Km=Standard_Range_km,
        Efficiency_WhKm=Base_Energy_Consumption_kWh_km,
        FastCharge_KmH="",
        RapidCharge="",
        PowerTrain=Terrain,
        PlugType="",
        BodyStyle="",
        Segment="",
        Charging_Price=Temperature_C,
        Prediction=decisionTreeRegressor)

        return render(request, 'RUser/Predict_Energy_Economy_Type.html',{'objs': decisionTreeRegressor})
    return render(request, 'RUser/Predict_Energy_Economy_Type.html')



