import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# ------------------------------
# 1. Load and preprocess data
# ------------------------------
data = pd.read_csv('Churn_Modelling.csv')
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode Gender
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

# One-hot encode Geography
onehot_encoder_geo = OneHotEncoder(handle_unknown='ignore')
geo_encoded = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)

# Features and target
X = data.drop('Exited', axis=1)
y = data['Exited']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save encoders and scaler
with open('label_encoder_gender.pkl', 'wb') as file:
    pickle.dump(label_encoder_gender, file)
with open('onehot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(onehot_encoder_geo, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# ------------------------------
# 2. Define Keras model
# ------------------------------
def create_model(neurons=32, layers=1):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))  # Use Input layer to avoid warnings
    for _ in range(layers):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ------------------------------
# 3. Set up KerasClassifier
# ------------------------------
model = KerasClassifier(model=create_model, verbose=0)

# ------------------------------
# 4. Define GridSearch parameters
# ------------------------------
param_grid = {
    'model__neurons': [16, 32, 64],
    'model__layers': [1, 2],
    'batch_size': [32, 64],
    'epochs': [30]
}

# Early stopping
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# ------------------------------
# 5. Perform GridSearchCV
# ------------------------------
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=1, verbose=1)
grid_result = grid.fit(X_train, y_train, callbacks=[early_stop])

# ------------------------------
# 6. Show best results
# ------------------------------
print("Best accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# ------------------------------
# 7. Save the best model (new format)
# ------------------------------
best_model = grid_result.best_estimator_.model_
best_model.save('best_churn_model.keras')  # modern Keras format