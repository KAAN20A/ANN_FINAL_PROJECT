import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
filename = "incoming_signal_Yusuf_Mirac_GOCEN_220448.xlsx"
window_size = 5 
try:
    df = pd.read_excel(filename)
    print(f"Successfully loaded {len(df)} samples.")
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
    exit()
df['rolling_mean'] = df['rssi'].rolling(window=window_size).mean()
df['rolling_std']  = df['rssi'].rolling(window=window_size).std()
df['min_val']      = df['rssi'].rolling(window=window_size).min()
df['max_val']      = df['rssi'].rolling(window=window_size).max()
df['energy']       = df['rssi'].rolling(window=window_size).apply(lambda x: np.sum(x**2))
df['diff']         = df['rssi'].diff()
for i in range(1, 10):
    df[f'lag_{i}'] = df['rssi'].shift(i)

df = df.dropna().reset_index(drop=True)

feature_cols = ['rssi', 'rolling_mean', 'rolling_std', 'min_val', 'max_val', 'energy', 'diff'] + [f'lag_{i}' for i in range(1, 10)]
df['block_id'] = df.index // window_size  

X = df[feature_cols].values
y_text = df['event'].values
groups = df['block_id'].values

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)
num_classes = len(encoder.classes_)
splitter = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = keras.Sequential([
    keras.layers.Dense(128, input_dim=X_train.shape[1], activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

print("\nStarting Training (Strictly Unweighted)...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)
loss, acc = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Confusion Matrix (Pure Unweighted)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_pure.png')

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve (Pure Unweighted)')
plt.legend()
plt.savefig('loss_curve_pure.png')

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))