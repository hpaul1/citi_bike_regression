# Citi Bike Regression Analysis

## Project Overview 
The goal of this project is to explore citi bike data, specifically focused on citi bike membership, compared to casual (non-member) riders. This project examines which stations had the highest membership rates and what factors (trip duration, distance, time of day, Precipitation, Temperature) were significant in predicting if a rider has a citi bike membership.  

## Data 
- Citibike ride data is from September 2021 and was downloaded from  https://ride.citibikenyc.com/system-data. 

- Weather Data was used from the JFK Airport and downloaded from National Centers for Environmental Information https://www.ncdc.noaa.gov/cdo-web/datasets#GHCND 

## Set Up 
Install citi_bike_regression packages with pip

```bash
cd citi_bike_regression
pip install requirements.txt
```

## Process
Two scripts (`citi_bike_prep.py` and `citi_bike_analysis.rmd`) work together to explore Citi Bike membership and the relation between Citi Bike trip metrics and weather data.

### Data Exploration and Processing  

`citi_bike_prep.py` takes in weather data and citi bike trip data and prepares that data to be used for a logistic regression. Additionally, `citi_bike_prep.py` provides group statistics including the stations with the highest proportion of members and the mean distance (in km), trip time (in minutes), and membership percentage by time of day (AM Rush, PM Rush, Not Rush) and day of the week. 

#### Input
Citi Bike Data : `citi_bike_data_in/202109-citibike-tripdata.csv`

Weather Data: `weather.csv`
#### Output
Cleaned Data:  `cleaned_data.csv`
#### How to Run 
``` bash
python citi_bike_prep.py
```

### Regression Analysis 
 `citi_bike_analysis.rmd` takes the cleaned output from the python script (`cleaned_data.csv`) and performs a logistic regression analysis looking at variables that predict if the rider has a membership. 

#### Input
Cleaned Data:  `cleaned_data.csv`

#### How to Run 
This is an R Notebook and can be run in the notebook (`citi_bike_analysis.rmd`) itself 
