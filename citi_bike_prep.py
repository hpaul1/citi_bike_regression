import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.neighbors import DistanceMetric
from math import radians
import math

#parameters
weather_in='weather.csv'
bike_in='citi_bike_data_in/202109-citibike-tripdata.csv'
file_out='cleaned_data.csv'

def importBikeData(bike_in):
    '''imports citi bike file into a pandas dataframe. This data is available at https://ride.citibikenyc.com/system-data.
    '''
    try:
        df=pd.read_csv(bike_in)
        print(df.head())
    except:
        print("Unable to read the Citi Bike File")

    print(df.head())
    print(len(df))

    #add date column
    df['DATE']=pd.to_datetime(df['ended_at']).dt.date

    print(df.dtypes)

    return df

def getDates(df):
    '''Gets min and max dates for the citi bike data to subset the weather data. This data was requested from the National Centers for Environmental Information.
    More information can be found at https://www.ncdc.noaa.gov/cdo-web/datasets#GHCND'''
    min_date=min(df['DATE'])
    max_date=max(df['DATE'])

    return [min_date, max_date]


def importWeatherData(weather_in, dates):
    '''Imports weather  data for New York City and limits the dates to match the date range of the citi bike data'''
    
    min_bike_date=dates[0]
    print(min_bike_date)
    max_bike_date=dates[1]
    print(max_bike_date)

    #import data
    df_weather=pd.read_csv(weather_in)
    print(df_weather.head())

    #only incude data from days that are in the citi bike data set
    df_weather['DATE']=pd.to_datetime(df_weather['DATE']).dt.date
    df_weather=df_weather[((df_weather['DATE']>=min_bike_date) & (df_weather['DATE']<=max_bike_date))]

    print('Min Date',min(df_weather['DATE']))
    print('Max Date',max(df_weather['DATE']))

    #inspect data
    print('total rows', len(df_weather))
    print(df_weather.head())
    print(df_weather.dtypes)

    #check dates in analysis
    print('Number of Dates',len(pd.unique(df_weather['DATE'])))

    #check locations in analysis 
    print('unique locations',pd.unique(df_weather['NAME']))
    print('number of unique locations',len(pd.unique(df_weather['NAME'])))

    #subset data to use JFK International Airport for the weather
    df_weather=df_weather[df_weather['NAME']=='JFK INTERNATIONAL AIRPORT, NY US']

    #Keep only needed columns
    df_weather=df_weather[['DATE','STATION', 'NAME','LATITUDE','LONGITUDE','PRCP','SNOW','TMAX','TMIN']]

    return df_weather

def calculateDistance(bike_df):
    '''Calculate the distance between start and end stations using the haversine equation'''
    bike_df['start_lat'] = np.radians(bike_df['start_lat'])
    bike_df['start_lng'] = np.radians(bike_df['start_lng'])
    bike_df['end_lat'] = np.radians(bike_df['end_lat'])
    bike_df['end_lng'] = np.radians(bike_df['end_lng'])

    #calculate distance using haversine formula using the radius of the earth at 40.68 degrees north. Distance in km
    bike_df['distance'] = 6369.092 * 2 * (np.arcsin(np.sqrt(np.sin((bike_df['end_lat']-bike_df['start_lat'])/2)**2 \
        + np.cos(bike_df['start_lat']) * np.cos(bike_df['end_lat']) * np.sin((bike_df['end_lng']-bike_df['start_lng'])/2)**2)))

    print('Max Distance:',max(bike_df['distance']))
    print('Min Distance:',min(bike_df['distance']))

    return bike_df

def cleanBike(df):
    '''Add rush hour categorization, trip time, and distance'''
    
    #calculate the trip time
    df['started_at']=pd.to_datetime(df['started_at'])
    df['ended_at']=pd.to_datetime(df['ended_at'])
    df['trip_time']=df['ended_at']-df['started_at']

    #Mon:0, Tues:1, Wed:2, Thurs:3, Fri:4, Sat:6, Sun:7
    df['day'] = df['started_at'].apply(lambda time: time.dayofweek)

    #categorize if trip is AM Rush, PM Rush, or Not Rush
    AM_RUSH_START='07:30:00'
    AM_RUSH_END='09:00:00'
    PM_RUSH_START='15:00:00'
    PM_RUSH_END='19:00:00'
    
    choice = ['am_rush', 'pm_rush']
    cond = [
    ((df['day']<6) & ((df['started_at'].dt.time.astype(str) > AM_RUSH_START) &(df['started_at'].dt.time.astype(str) < AM_RUSH_END))), 
    ((df['day']<6) & ((df['started_at'].dt.time.astype(str) > PM_RUSH_START) & (df['started_at'].dt.time.astype(str) < PM_RUSH_END)))
    ]

    df['rush']=np.select(cond, choice, 'not_rush') 

    #calculate distance 
    df=calculateDistance(df)

    return df

def mergeWeather(df_bike,df_weather):
    '''Merge Weather Data by Day to get daily weather information for citi bike trips'''
    pre_merge_bike=(len(df_bike))
    df_weather=df_weather.drop_duplicates(subset=['DATE'])
    df_merge=df_bike.merge(df_weather, how='left',)
    
    #check that merge kept the same number of columns as the dataframe before
    if (len(df_merge)-pre_merge_bike)==0:
        print('merge sucessful')
    
    return df_merge

def regressionAnalysis(df):
    '''Prepare the data for a regression model to be run in r'''

    #set binary variable
    df['membership_ind'] = df['member_casual'].apply(lambda x: 1 if x == 'member' else 0)
    print(df['membership_ind'].value_counts())

    df['total_precip']=df['PRCP']+df['SNOW']
    df=df.drop(columns=['rideable_type','PRCP','STATION', 'NAME', 'LATITUDE','LONGITUDE','SNOW'])

    #categorize by precipitation that day. Considered a rainy/snowy (precip=1) day if the total precipitation was greater than 0.1 
    df['precip_ind'] = df['total_precip'].apply(lambda x: 1 if x > 0.1 else 0)

    #indicate if the trip is in rush hour or not
    df['rush_ind'] = df['rush'].apply(lambda x: 0 if x =='not_rush' else 1)

    return df

def addStatistics(df):
    '''Calculate grouped statistics'''
    df['trip_time'] = df['trip_time'].dt.total_seconds()
    df['trip_time'] = df['trip_time']/60

    #identify which starting stations have the highest percent of members
    station_means=df.groupby('start_station_name')['membership_ind'].agg(['mean'])
    print('Stations with the highest percentage of members')
    print (station_means.sort_values(by='mean',ascending=False).head(10))

    print('Rush Hour: Mean Distance(km), Trip Duration (minutes), and Membership')
    print(df.groupby('rush')['distance'].mean())
    print(df.groupby('rush')['trip_time'].mean())
    print(df.groupby('rush')['membership_ind'].mean())

    print('Day: Mean Distance (km), Trip Duration(minutes), and Membership')
    print(df.groupby('day')['distance'].mean())
    print(df.groupby('day')['trip_time'].mean())
    print(df.groupby('day')['membership_ind'].mean())

    print('Membership Type: Mean Distance(km) and Trip Time (minutes)')
    print(df.groupby('member_casual')['distance'].mean())
    print(df.groupby('member_casual')['trip_time'].mean())
    

    return

def main():
    df_bike=importBikeData(bike_in=bike_in)
    dates=getDates(df_bike)
    df_weather=importWeatherData(weather_in=weather_in,dates=dates)
    df_bike=cleanBike(df_bike)
    df=mergeWeather(df_bike,df_weather)
    df=regressionAnalysis(df)
    addStatistics(df)
    df.to_csv(file_out,index=False)

    return

if __name__ == "__main__":
    main()