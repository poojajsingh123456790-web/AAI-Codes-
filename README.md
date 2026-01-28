# AAI-Codes-

Practical No. 1Aim: Implementing advanced deep learning algorithms such as convolutional neural networks (CNNs) or recurrent neural networks (RNNs) using Python libraries like TensorFlow or PyTorch. 

Code:

import tensorflow as tffrom tensorflow.keras import Sequentialfrom tensorflow.keras.layers import Embedding, SimpleRNN, Densefrom tensorflow.keras.datasets import imdbfrom tensorflow.keras.preprocessing.sequence import pad_sequences# Step 1: Load and Prepare the IMDb Datasetmax_features = 10000 # Use the top 10,000 most frequent wordsmaxlen = 100 # Limit each review to 100 words# Load the dataset(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)# Pad sequences to ensure all inputs are the same lengthx_train = pad_sequences(x_train, maxlen=maxlen)x_test = pad_sequences(x_test, maxlen=maxlen)# Step 2: Define the RNN Modelmodel = Sequential([Embedding(max_features, 32, input_length=maxlen), # Embedding layerSimpleRNN(32, activation='relu'), # RNN layerDense(1, activation='sigmoid') # Output layer])# Step 3: Compile the Modelmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])# Step 4: Train the Modelmodel.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)# Step 5: Evaluate the Modeltest_loss, test_acc = model.evaluate(x_test, y_test)print(f"Test Accuracy: {test_acc:.2f}")








Practical No. 2Aim: Building a natural language processing (NLP) model for sentiment analysis or text classification. 

Code:

from transformers import pipeline# Load the pre-trained sentiment-analysis pipelinesentiment_analyzer = pipeline('sentiment-analysis')# Example texts to classifytexts = ["I love this product, it's amazing!","This is the worst service I've ever had.","I'm so happy with my purchase, highly recommend!","I'm not satisfied at all with this experience."]# Function to analyze sentimentdef analyze_sentiment(texts):for text in texts:result = sentiment_analyzer(text)label = result[0]['label']score = result[0]['score']print(f"Text: {text}\nSentiment: {label} (Confidence: {score:.2f})\n")# Call the function to classify sentimentsanalyze_sentiment(texts)





Practical No. 3Aim: Creating a chatbot using advanced techniques like transformer models

Code:

from transformers import pipeline# Step 1: Load a Pre-trained Transformer Modelchatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")# Step 2: Start a Chat Sessionprint("Chatbot: Hello! I'm here to chat with you. Type 'exit' to end the conversation.")# Step 3: Loop for Chattingwhile True:user_input = input("You: ")if user_input.lower() == "exit":print("Chatbot: Goodbye!")break# Generate a Responseresponse = chatbot(user_input, max_length=50, num_return_sequences=1)print("Chatbot:", response[0]['generated_text'])





Practical No. 4Aim: Developing a recommendation system using collaborative filtering or deep learning approaches. 

Code:

import pandas as pdfrom sklearn.metrics.pairwise import cosine_similarityfrom sklearn.preprocessing import StandardScaler# Step 1: Load datasetdf = pd.read_csv('C:/Users//Desktop/ml-latest-small/ratings.csv') # Assuming columns: userId, movieId, rating# Step 2: Create user-item interaction matrixinteraction_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)# Step 3: Normalize the data (optional but helps with similarity calculation)scaler = StandardScaler(with_mean=False)interaction_matrix_scaled = scaler.fit_transform(interaction_matrix)# Step 4: Compute user-user similarityuser_similarity = cosine_similarity(interaction_matrix_scaled)user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)# Step 5: Generate recommendationsdef recommend(user_id, k=5):# Find similar userssimilar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:k+1]# Collect weighted ratings from similar userssimilar_users_ratings = interaction_matrix.loc[similar_users.index]weighted_ratings = similar_users_ratings.T.dot(similar_users)# Exclude movies already rated by the useruser_rated = interaction_matrix.loc[user_id]recommendations = weighted_ratings[user_rated == 0].sort_values(ascending=False).head(k)return recommendations.index.tolist()# Example: Recommend movies for user ID user_id = int(input("Enter your input"))recommendations = recommend(user_id)print(f"Recommendations for User {user_id}: {recommendations}")





Practical No. 5Aim: Implementing a computer vision project, such as object detection or image segmentation. Requirements: Python: 3.8.10

Code:

from ultralytics import YOLOimport cv2# Load the YOLO modelmodel = YOLO("C:/Users/Desktop/Yolo-Weights/yolov8n.pt")# Process the image without automatically showing itresults = model("C:/Users//Desktop//p5/imgs/1.jpg")# If results is a list, access the first element (which should contain the image)image = results[0].plot() # Plot the results (draw bounding boxes, etc.)# Resize the image to a suitable size before displayingresized_image = cv2.resize(image, (800, 800)) # Adjust 800x800 to your preferred size# Create the OpenCV window with a normal resizing optioncv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)# Resize the window to match the size of the imagecv2.resizeWindow("Processed Image", resized_image.shape[1], resized_image.shape[0])# Display the resized image in the windowcv2.imshow("Processed Image", resized_image)# Wait for a key press and close the windowcv2.waitKey(0)cv2.destroyAllWindows()




Practical No. 6Aim: Applying reinforcement learning algorithms to solve complex decision-making problems. 

Code:

import numpy as npimport random# Define the environmentgrid_size = 3 # Smaller gridgoal_state = (2, 2)obstacle_state = (1, 1) # Single obstacleactions = ['up', 'down', 'left', 'right']action_to_delta = {'up': (-1, 0),'down': (1, 0),'left': (0, -1),'right': (0, 1)}# Initialize Q-table (simple 3D array for states and actions)q_table = np.zeros((grid_size, grid_size, len(actions)))# Parametersalpha = 0.1 # Learning rategamma = 0.9 # Discount factorepsilon = 1.0 # Exploration rateepsilon_decay = 0.99min_epsilon = 0.1episodes = 200 # Fewer episodes# Reward functiondef get_reward(state):if state == goal_state:return 10 # Reward for reaching the goalelif state == obstacle_state:return -10 # Penalty for hitting the obstaclereturn -1 # Step penalty# Check if the new state is validdef is_valid_state(state):return 0 <= state[0] < grid_size and 0 <= state[1] < grid_size and state != obstacle_state# Main Q-learning loopfor episode in range(episodes):state = (0, 0) # Start at the top-left cornertotal_reward = 0while state != goal_state:# Choose an action (epsilon-greedy strategy)if random.uniform(0, 1) < epsilon:action = random.choice(actions) # Exploreelse:action = actions[np.argmax(q_table[state[0], state[1]])] # Exploit best action# Perform the actiondelta = action_to_delta[action]next_state = (state[0] + delta[0], state[1] + delta[1])# Stay in the same state if the move is invalidif not is_valid_state(next_state):next_state = state# Get reward and update Q-tablereward = get_reward(next_state)total_reward += rewardbest_next_action = np.max(q_table[next_state[0], next_state[1]])q_table[state[0], state[1], actions.index(action)] += alpha * (reward + gamma * best_next_action - q_table[state[0], state[1], actions.index(action)])# Update statestate = next_state# Decay epsilonepsilon = max(min_epsilon, epsilon * epsilon_decay)print(f"Episode {episode + 1}: Total Reward = {total_reward}")# Display the learned policypolicy = np.full((grid_size, grid_size), ' ')for i in range(grid_size):for j in range(grid_size):if (i, j) == goal_state:policy[i, j] = 'G' # Goalelif (i, j) == obstacle_state:policy[i, j] = 'X' # Obstacleelse:best_action = np.argmax(q_table[i, j])policy[i, j] = actions[best_action][0].upper() # First letter of the best actionprint("Learned Policy:")print(policy)





Practical No. 7Aim: Utilizing transfer learning to improve model performance on limited datasets

Code:

import tensorflow as tffrom tensorflow.keras.applications import MobileNetfrom tensorflow.keras.models import Modelfrom tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2Dfrom tensorflow.keras.optimizers import Adam# ParametersIMG_SIZE = 128 # Smaller image size for faster computationBATCH_SIZE = 16 # Reduced batch size to save memoryEPOCHS = 2 # Fewer epochs for quicker trainingLEARNING_RATE = 0.001# Load and Preprocess MNIST Dataset(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()# Use only a subset of the data (e.g., 10,000 samples for training)x_train, y_train = x_train[:10000], y_train[:10000]x_test, y_test = x_test[:2000], y_test[:2000]# Preprocessing functiondef preprocess(image, label):image = tf.image.resize(tf.expand_dims(image, axis=-1), (IMG_SIZE, IMG_SIZE)) / 255.0image = tf.image.grayscale_to_rgb(image) # Convert grayscale to RGBlabel = tf.one_hot(label, depth=10) # One-hot encode labelsreturn image, label# Create TensorFlow datasetstrain_dataset = (tf.data.Dataset.from_tensor_slices((x_train,y_train)).map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))test_dataset = (tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))# Load the smaller pre-trained MobileNet modelbase_model = MobileNet(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))# Freeze the base modelbase_model.trainable = False# Add custom layers on topx = base_model.outputx = GlobalAveragePooling2D()(x) # Reduce dimensionsx = Dropout(0.3)(x) # Dropout for regularizationpredictions = Dense(10, activation="softmax")(x) # Output layer for 10 classes# Create the full modelmodel = Model(inputs=base_model.input, outputs=predictions)# Compile the modelmodel.compile(optimizer=Adam(learning_rate=LEARNING_RATE),loss="categorical_crossentropy",metrics=["accuracy"])# Train the modelhistory = model.fit(train_dataset,validation_data=test_dataset,epochs=EPOCHS)# Evaluate the model on the test datasetevaluation = model.evaluate(test_dataset, verbose=1)# Print the evaluation metricsprint(f"Test Loss: {evaluation[0]:.4f}")print(f"Test Accuracy: {evaluation[1]:.4f}")




Practical No. 8Aim: Building a deep learning model for time series forecasting or anomaly detection. 

Code:

# time seriesimport numpy as npimport pandas as pdimport matplotlib.pyplot as pltfrom sklearn.preprocessing import MinMaxScalerfrom tensorflow.keras.models import Sequentialfrom tensorflow.keras.layers import LSTM, Dense# 1. Prepare and normalize datadata = pd.DataFrame(np.sin(np.linspace(0, 100, 1000)), columns=['value'])scaler = MinMaxScaler(feature_range=(0, 1))scaled_data = scaler.fit_transform(data)# 2. Create dataset for LSTMX, y = [], []for i in range(len(scaled_data) - 10):X.append(scaled_data[i:i+10, 0])y.append(scaled_data[i+10, 0])X, y = np.array(X), np.array(y)X = X.reshape(X.shape[0], X.shape[1], 1)# 3. Split data into train and testtrain_size = int(len(X) * 0.8)X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]# 4. Build and train modelmodel = Sequential([LSTM(50, input_shape=(X_train.shape[1], 1)), Dense(1)])model.compile(optimizer='adam', loss='mse')model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)# 5. Predict and plotpredictions = scaler.inverse_transform(model.predict(X_test))y_test = scaler.inverse_transform(y_test.reshape(-1, 1))plt.plot(y_test, label='True')plt.plot(predictions, label='Predicted')plt.legend()plt.show()





Practical No. 9Aim: Implementing a machine learning pipeline for automated feature engineering and model selection. 

Code: 

import pickleimport numpy as npimport pandas as pdfrom sklearn.model_selection import train_test_splitfrom sklearn.compose import ColumnTransformerfrom sklearn.impute import SimpleImputerfrom sklearn.preprocessing import OneHotEncoderfrom sklearn.preprocessing import MinMaxScalerfrom sklearn.pipeline import Pipelinefrom sklearn.feature_selection import SelectKBest,chi2from sklearn.tree import DecisionTreeClassifierdf = pd.read_csv('C:/Users//Desktop/p10/train.csv')df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)# Step 1 -> train/test/splitX_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']), df['Survived'], test_size=0.2, random_state=42)X_train.head()y_train.sample(5)# imputation transformertrf1 = ColumnTransformer([('impute_age',SimpleImputer(),[2]),('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])],remainder='passthrough')# one hot encodingtrf2 = ColumnTransformer([('ohe_sex_embarked',OneHotEncoder(sparse=False,handle_unknown='ignore'),[1,6])],remainder='passthrough')# Scalingtrf3 = ColumnTransformer([('scale',MinMaxScaler(),slice(0,10))])# Feature selectiontrf4 = SelectKBest(score_func=chi2,k=8)# train the modeltrf5 = DecisionTreeClassifier()pipe = Pipeline([('trf1',trf1),('trf2',trf2),('trf3',trf3),('trf4',trf4),('trf5',trf5)])# trainpipe.fit(X_train,y_train)pipe.named_steps# Display Pipelinefrom sklearn import set_configset_config(display='diagram')# Predicty_pred = pipe.predict(X_test)from sklearn.metrics import accuracy_scoreaccuracy_score(y_test,y_pred)# cross validation using cross_val_scorefrom sklearn.model_selection import cross_val_scorecross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()from sklearn.model_selection import GridSearchCV# Corrected parameter gridparams = {'trf5__max_depth': [1, 2, 3, 4, 5, None]}grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')grid.fit(X_train, y_train)grid.best_score_grid.best_params_# export pickle.dump(pipe,open('C:/Users/Desktop/p10/pipe.pkl','wb'))predict.pyimport pickleimport numpy as npimport pandas as pd pipe = pickle.load(open('C:/Users/Desktops/pipe.pkl','rb'))# Assume user inputtest_input2 = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'],dtype=object).reshape(1,7)# Adding a new row to the dataframe# test_input2 = np.vstack([# test_input2, # np.array([12, 'female', 47.0, 0, 0, 54.3, 'C'], dtype=object).reshape(1, 7),# np.array([3, 'male', 23.0, 0, 0, 12.3, 'S'], dtype=object).reshape(1, 7)# ])columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']test_input2_df = pd.DataFrame(test_input2, columns=columns)# Assume user inputprint(pipe.predict(test_input2_df))







Practical No. 10Aim: Using advanced optimization techniques like evolutionary algorithms or Bayesian optimization for hyperparameter tuning. 

Code:

from sklearn.datasets import load_irisfrom sklearn.ensemble import RandomForestClassifierfrom sklearn.model_selection import cross_val_scorefrom skopt import BayesSearchCVfrom skopt.space import Real, Integer# Load datasetdata = load_iris()X, y = data.data, data.target# Define the modelmodel = RandomForestClassifier(random_state=42)# Define the search space for hyperparametersparam_space = {'n_estimators': Integer(10, 200), # Number of trees'max_depth': Integer(1, 20), # Maximum depth of a tree'min_samples_split': Real(0.01, 0.3), # Minimum fraction of samples required to split'min_samples_leaf': Integer(1, 10), # Minimum samples at a leaf node'max_features': Real(0.1, 1.0), # Fraction of features to consider for split}# Bayesian Optimization with Cross-Validationopt = BayesSearchCV(estimator=model,search_spaces=param_space,n_iter=50, # Number of parameter settings to trycv=5, # Number of cross-validation foldsn_jobs=-1, # Use all processorsrandom_state=42)# Perform the optimizationopt.fit(X, y)# Resultsprint("Best Parameters:", opt.best_params_)print("Best Cross-Validation Score:", opt.best_score_)





