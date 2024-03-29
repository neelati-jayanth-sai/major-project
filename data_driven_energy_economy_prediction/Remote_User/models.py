from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class energy_economy_prediction(models.Model):

    Fid= models.CharField(max_length=300)
    Brand= models.CharField(max_length=300)
    Model= models.CharField(max_length=300)
    AccelSec= models.CharField(max_length=300)
    TopSpeed_KmH= models.CharField(max_length=300)
    Range_Km= models.CharField(max_length=300)
    Efficiency_WhKm= models.CharField(max_length=300)
    FastCharge_KmH= models.CharField(max_length=300)
    RapidCharge= models.CharField(max_length=300)
    PowerTrain= models.CharField(max_length=300)
    PlugType= models.CharField(max_length=300)
    BodyStyle= models.CharField(max_length=300)
    Segment= models.CharField(max_length=300)
    Charging_Price= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



