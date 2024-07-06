import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import can


speed_data = [80, 25, 50, 42, 78, 38, 95, 87, 32, 89, 61, 93, 55, 22, 70, 60, 45, 30, 85, 40, 65, 73, 58]
acceleration_data = [1.5, 2.7, 1.3, 2.8, 1.6, 1.4, 2.9, 3.7, 2.6, 3.8, 2.9, 3.4, 2.5, 2.2, 1.8, 2.0, 3.0, 2.3, 3.2, 2.1, 2.8, 1.7, 2.5]
braking_data = [1.1, 2.0, 2.9, 0.7, 2.6, 2.9, 1.0, 2.7, 2.0, 2.8, 2.6, 0.9, 1.1, 2.2, 1.3, 2.3, 1.9, 1.5, 2.5, 1.2, 1.8, 1.4, 2.4]
tailgating_distance = [0.5, 1.0, 12.5, 7.5, 11.5, 12.5, 12.5, 11.0, 16.5, 5.5, 19.0, 2.0, 14.0, 12.5, 10.0, 15.0, 6.0, 8.0, 18.0, 9.0, 17.0, 4.0, 13.0]
lane_discipline = [13, 1, 3, 9, 8, 1, 9, 19, 15, 8, 0, 1, 8, 6, 12, 4, 5, 14, 7, 10, 2, 16, 11]
cornering_speed = [45.5, 6.3, 17.7, 46.1, 12.6, 69.0, 90.1, 57.7, 15.1, 80.7, 57.1, 70.2, 17.8, 13.9, 33.5, 53.2, 70.8, 22.4, 88.5, 38.7, 61.4, 49.6, 28.3]


def calculate_score(value, threshold, close_range=0.5):
    if value > threshold + close_range:
        return 1
    elif threshold < value <= threshold + close_range:
        return 2
    elif value == threshold:
        return 3
    elif threshold - close_range < value < threshold:
        return 4
    else:  
        return 5

def extract_data_from_json(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    
    speed_data, acceleration_data, braking_data, tailgating_distance, lane_discipline, cornering_speed = [], [], [], [], [], []

    for entry in json_data['Rows']:
        data = entry['Data']
        metric = data[3]['ScalarValue']
        
        if 'ScalarValue' in data[5]:
            value = float(data[5]['ScalarValue'])

            if metric == 'Vehicle.OBD.speed_data':
                speed_data.append(value)
            elif metric == 'Vehicle.OBD.acceleration_data':
                acceleration_data.append(value)
            elif metric == 'Vehicle.OBD.braking_data':
                braking_data.append(value)
            elif metric == 'Vehicle.OBD.tailgating_distance':
                tailgating_distance.append(value)
            elif metric == 'Vehicle.OBD.Lane_discipline':
                lane_discipline.append(value)
            elif metric == 'Vehicle.OBD.cornering_speed':
                cornering_speed.append(value)

    return speed_data, acceleration_data, braking_data, tailgating_distance, lane_discipline, cornering_speed

def predict_scores(file_path):
    extracted_data = extract_data_from_json(file_path)

    thresholds = [60, 3.8, 2.8, 4, 5, 30]

    speed_data, acceleration_data, braking_data, tailgating_distance, lane_discipline, cornering_speed = extracted_data
    speed_scores = [calculate_score(speed, thresholds[0]) for speed in speed_data]
    acceleration_scores = [calculate_score(accel, thresholds[1]) for accel in acceleration_data]
    braking_scores = [calculate_score(brake, thresholds[2]) for brake in braking_data]
    tailgating_scores = [calculate_score(tailgating, thresholds[3]) for tailgating in tailgating_distance]
    lane_discipline_scores = [calculate_score(lane, thresholds[4]) for lane in lane_discipline]
    cornering_speed_scores = [calculate_score(cornering, thresholds[5]) for cornering in cornering_speed]

    features = list(zip(speed_scores, acceleration_scores, braking_scores, tailgating_scores, lane_discipline_scores, cornering_speed_scores))
    
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    target = [sum(feature_set) / 6 for feature_set in features]
    
    X_train, X_test, y_train, y_test = train_test_split(features_normalized, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    
    prediction_1=predictions.mean()


    min_pred = 1
    max_pred = 5

    scaled_predictions = (prediction_1 - min_pred) / (max_pred - min_pred) * 100
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return scaled_predictions


def send_scores_to_can(predicted_scores):
    bus = can.interface.Bus(channel='vcan', bustype='socketcan')

    message = can.Message(
        arbitration_id=0x100,  
        data=json.dumps({'predicted_scores': predicted_scores}).encode('utf-8'),
        extended_id=False
    )
    bus.send(message)

file_path = '/Users/sehwagvijay/Desktop/Predictive-Driver-Scoring-for-Insurance-Risk-Assessment-Using-Machine-Learning/invehicle_data.json'
predicted_scores = predict_scores(file_path)
print(predicted_scores)
#send_scores_to_can(predicted_scores)
