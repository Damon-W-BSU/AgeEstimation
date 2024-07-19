from typing import Iterator, List, Union, Tuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1
from tensorflow.keras.callbacks import History

def visualize_augmentations(data_generator: ImageDataGenerator, df: pd.DataFrame):
    series = df.iloc[2]

    df_augmentation_visualization = pd.concat([series, series], axis=1).transpose()

    iterator_visualizations = data_generator.flow_from_dataframe(
        dataframe=df_augmentation_visualization,
        x_col="path",
        y_col="age",
        class_mode="raw",
        target_size=(224, 224),
        batch_size=1,
    )

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1) 
        batch = next(iterator_visualizations)
        img = batch[0]
        img = img[0, :, :, :]
        plt.imshow(img)
    plt.show()
    plt.close()

def get_callbacks(model_name: str) -> List[Union[EarlyStopping]]:
    logdir = (
        "logs/scalars/" + model_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_mean_absolute_error",
        min_delta=0.01,
        patience=10,
        verbose=2,
        mode="min",
        restore_best_weights=True,
    )

    # model_checkpoint_callback = ModelCheckpoint(
    #     "models/" + model_name + ".h5",
    #     monitor="val_mean_absolute_error",
    #     verbose=0,
    #     save_best_only=True,
    #     mode="min",
    #     save_freq="epoch",
    # ) 

    return [early_stopping_callback]

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, val = train_test_split(df, test_size=0.2, random_state=69) # switch random_state to 'None' for random split
    val, test = train_test_split(val, test_size=0.5, random_state=69)

    print("shape train: ", train.shape)  
    print("shape val: ", val.shape)  
    print("shape test: ", test.shape) 

    print("Descriptive statistics of train:")
    print(train.describe())
    return train, val, test  
 
def get_mean_baseline(train: pd.DataFrame, val: pd.DataFrame) -> float:
    y_hat = train["age"].mean()
    val["y_hat"] = y_hat

    mae = MeanAbsoluteError()
    mae = mae(val["age"], val["y_hat"]).numpy()

    print("mean baseline MAE: ", mae)

    return mae

def create_generators(
    df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, plot_augmentations: any
) -> Tuple[Iterator, Iterator, Iterator]:
    
    train_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.75, 1),
        shear_range=0.1,
        zoom_range=[0.75, 1],
        horizontal_flip=True,
        validation_split=0.2,
    )
    validation_generator =  ImageDataGenerator(
        rescale =1.0/255
    )
    test_generator = ImageDataGenerator(
        rescale=1.0 / 255
    )
    if plot_augmentations == True:
        visualize_augmentations(train_generator, df)
    train_generator = train_generator.flow_from_dataframe(
        dataframe=train,
        x_col="path",
        y_col="age",
        class_mode="raw",
        target_size=(224, 224),
        batch_size=64,
    )
    validation_generator = validation_generator.flow_from_dataframe(
        dataframe=val,
        x_col="path",
        y_col="age",
        class_mode="raw",
        target_size=(224, 224),
        batch_size=64,
    )
    test_generator = test_generator.flow_from_dataframe(
        dataframe=test,
        x_col="path",
        y_col="age",
        class_mode="raw",
        target_size=(224, 224),
        batch_size=64,
    )
    return train_generator, validation_generator, test_generator

def adapt_efficient_net() -> Model:
    inputs = layers.Input(
        shape=(224, 224, 3)
    )  
    model = EfficientNetB1(include_top=False, input_tensor=inputs, weights="imagenet",)

    model.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation='relu')(x) # additional dense layer
    top_dropout_rate = 0.4
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(1, name="pred")(x)

    model = keras.Model(inputs, outputs, name="EfficientNet")

    return model

def run_model(
    model_name: str,
    model_function: Model,
    lr: float,
    train_generator: Iterator,
    validation_generator: Iterator,
    test_generator: Iterator,
) -> Tuple[History, List[float]]:
    
    callbacks = get_callbacks(model_name)
    model = model_function

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer, loss="mean_absolute_error", metrics=[MeanAbsoluteError(), RootMeanSquaredError()]
    )
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    eval = model.evaluate(
        test_generator,
        callbacks=callbacks,
    )
    return history, eval

def plot_results(model_history_eff_net: History, mean_baseline: float):
    # Prepare data for plotting
    epochs = range(len(model_history_eff_net.history['mean_absolute_error']))
    mae = model_history_eff_net.history['mean_absolute_error']
    val_mae = model_history_eff_net.history['val_mean_absolute_error']

    # Plot MAE and validation MAE
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mae, label='MAE')
    plt.plot(epochs, val_mae, label='Validation MAE')
    
    # Plot mean baseline and ticks
    plt.axhline(y=mean_baseline, color='r', linestyle='--', label='Mean Baseline')
    plt.grid(axis='y', linestyle='-', color='lightgrey')

    # Add labels and legend
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.ylim(0, 20)
    plt.legend()
    plt.savefig("results.png")
    plt.show()

def run (small_sample=False):
    images = os.listdir('facial_data/processed_images/')
    paths = []
    ages = []

    count = 0
    for i in images:
        if count % 1 == 0:
            paths.append(i)
            ages.append(int(i[0:i.index('_')]))
        count += 1

    d ={'path': paths, 'age': ages}
    df = pd.DataFrame(data=d)
    df["path"] = ('facial_data/processed_images/' + df['path'])
    df["age"] = df["age"].astype(float)
    
    if small_sample == True:
        df = df.iloc[0:1000]

    print(df)
    train, val, test = split_data(df)
    mean_baseline = get_mean_baseline(train, val)

    train_generator, validation_generator, test_generator = create_generators(
        df=df, train=train, val=val, test=test, plot_augmentations=True
    )

    eff_net_history, test_eval = run_model(
        model_name="eff_net",
        model_function=adapt_efficient_net(),
        lr=0.001,
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
    )

    plot_results(eff_net_history, mean_baseline)

    print(eff_net_history.history.keys())

    # Print Best MAE
    nums = eff_net_history.history['val_mean_absolute_error']
    smallest = min(nums)
    smallest_index = nums.index(smallest)
    print("Best MAE:", smallest, "@ epoch", smallest_index + 1,)

    # Print associated RMSE
    nums = eff_net_history.history['val_root_mean_squared_error']
    smallest = nums[smallest_index]
    print("(RMSE:", smallest, ")")

    # Print test results
    print("Test MAE:", test_eval[1])
    print("Test RMSE:", test_eval[2])

# Run code
if __name__ == "__main__":
    run(small_sample=False)