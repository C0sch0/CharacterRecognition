import image_preprocessing
import characteristics
import measure
import pandas as pd
import os


# el programa iniciara intentando predecir una serie de datos usando KNN.
# Vera primero si encuentra las bb_dd escritas en csv.
# Si no, las creara a partir de imagenes segmentadas.
# opcional: generar la segmentacion de img_grande a multiples_imgs usando metodo de contornos


# Se realiza normalizacion con media y std ya que mejora considerablemente el resultado. Detalles de esto en informe.
# Para cada vector de imagen, calculamos sus features geometricas [7 Momentos de Hu, Roundness, Area, Perimetro]
# Luego, utilizamos tecnica de KNN con N=1, usando los 5 primeros momentos de Hu.
# Por ultimo, analizamos los resultados con graficos de dispersion de datos que nos ayuden a comprender los datos.

# Let's begin

CHARS = ["A", "D", "F", "G", "S"]
PREDICTORS = ['Hu_1', 'Hu_2', 'Hu_3', 'Hu_4']
INPUT_PATH_01 = os.path.join("imgs", "training", "Training_01.png")
INPUT_PATH_02 = os.path.join("imgs", "training", "Training_02.png")
TESTING_PATH = os.path.join("imgs", "testing", "Testing.png")
SEGMENTED_TRAINING_PATH = os.path.join("imgs", "segmented_training")
SEGMENTED_TESTING_PATH = os.path.join("imgs", "segmented_testing")
TRAINING_CSV_PATH = 'training_data.csv'
TESTING_CSV_PATH = 'testing_data.csv'


def test():
    df_training = None
    df_testing = None
    try:
        df_training = pd.read_csv(TRAINING_CSV_PATH)
    except FileNotFoundError:
        print("--- training dataset not found, creating one ---")
        df_training = create_labeled_data(SEGMENTED_TRAINING_PATH, TRAINING_CSV_PATH)

    try:
        df_testing = pd.read_csv(TESTING_CSV_PATH)
    except FileNotFoundError:
        print("--- testing dataset not found, creating one ---")
        df_testing = create_labeled_data(SEGMENTED_TESTING_PATH, TESTING_CSV_PATH)

    if df_testing is not None:
        # normalize db
        normalize = df_training.loc[:, PREDICTORS]
        mean = normalize.mean(0)
        std = normalize.std(0)
        normalized = (normalize - mean) / std
        normalized['tag'] = df_training['tag']

        all_predictions = []
        for index, entry in df_testing.iterrows():
            normalize_entry = pd.DataFrame([entry], columns=PREDICTORS)
            normalize_entry = (normalize_entry - mean) / std

            # Key function: Predict based on KNN what this entry looks like
            prediction = characteristics.predict(normalize_entry, normalized, PREDICTORS)

            all_predictions.append([entry['tag'], prediction])
            print("It was a: ", entry['tag'],  " and I said: ", prediction)
        # print(all_predictions)
        # Now we see how we did
        accuracy = measure.confusion(CHARS, all_predictions)
        print("accuracy: {}% ".format(accuracy))


# For every image, we create binary, calculate geometric features
def create_labeled_data(input_path, csv_path):
    geometric_chars = ['tag', 'Area', 'Roundness', 'Hu_1', 'Hu_2', 'Hu_3', 'Hu_4', 'Hu_5', 'Hu_6', 'Hu_7']
    df_training = pd.DataFrame(columns=geometric_chars)

    for image_path in os.listdir(input_path):
        if ".png" in image_path:
            tag = [image_path[0].capitalize()]
            binary_image = image_preprocessing.binarize(os.path.join(input_path, image_path))
            features = tag + characteristics.extract_geometric_features(binary_image)
            # todo: return a dict so assigment is key based, depending on position is too risky
            new_row = {'tag': features[0],
                       'Area': features[1],
                       'Roundness': features[2],
                       'Hu_1': features[3],
                       'Hu_2': features[4],
                       'Hu_3': features[5],
                       'Hu_4': features[6],
                       'Hu_5': features[7],
                       'Hu_6': features[8],
                       'Hu_7': features[9]}
            df_training = df_training.append(new_row, ignore_index=True)

    df_training.to_csv(csv_path, index=False)
    print("Saving training data into csv")
    return df_training


# Uncomment if you want to trigger the segmentation. We will read the saved ones.
# You will have to tag them manually by name change or array with tags. I did it manually for simplicity.
#image_preprocessing.segment_into_multiple(INPUT_PATH_01, SEGMENTED_TRAINING_PATH)
#image_preprocessing.segment_into_multiple(INPUT_PATH_02, SEGMENTED_TRAINING_PATH)
#image_preprocessing.segment_into_multiple(TESTING_PATH, SEGMENTED_TESTING_PATH)

test()
