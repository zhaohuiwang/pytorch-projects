
"""
Financial predictions, inventory forecasting, or resource allocation. 

Elevating Forecasting with PyTorch Lightning and PyTorch Forecasting
PyTorch has the capability to manage the entire pipeline for deep learning and artificial intelligence projects. However, certain applications, such as forecasting, can become quite complex. To address this, the third-party framework PyTorch Forecasting has been introduced.

PyTorch Lightning is a lightweight PyTorch wrapper that provides a high-level interface for training PyTorch models. It is designed to simplify and standardize the training loop, making it easier to write cleaner, more modular code for deep learning projects. PyTorch Lightning introduces a set of abstractions and conventions that remove boilerplate code and allow researchers and practitioners to focus more on the model architecture and experiment configurations.
PyTorch Forecasting is a framework built on top of PyTorch Lightning designed to facilitate time series forecasting using neural networks for practical applications. 


lightning.pytorch is equivalent to pytorch_lightning

"""
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
#from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

import torch

"""
#  code snippet establishes an environment for time series forecasting utilizing the PyTorch Forecasting library. Baseline class, which provides a simple benchmark model, serves as a reference point for evaluating the performance of more intricate models. 
# TemporalFusionTransformer employs attention mechanisms to effectively capture temporal patterns
# TimeSeriesDataSet class adeptly manages features such as time indexing and grouping, both of which are crucial for ensuring accurate forecasts.
# GroupNormalizer ensures that input features are properly scaled. This step is significant as it may enhance model convergence during training, particularly when dealing with time series data characterized by varying scales.
# SMAPE (Symmetric Mean Absolute Percentage Error), Poisson Loss, and Quantile Loss to measure the performance of the forecasting models. These metrics are valuable in evaluating the accuracy of the models predictions when compared to actual outcomes.
# optimize_hyperparameters function allows for automated searching of optimal model parameters. 
"""

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# for using TensorBoard, a robust visualization tool for TensorFlow projects.
# import tensorflow as tf
# import tensorboard as tb
"""
# the code ensures a smooth integration of TensorBoard's file handling capabilities with TensorFlow's input/output operations.
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# This dataset is already included in pytorch forecasting library so import it using below commands: as of 10/07/2024 this get_stallion_data() does not work 
# from pytorch_forecasting.data.examples import get_stallion_data
# data = get_stallion_data()
# Get Device for Training
"""

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



BASE_URL = "https://raw.githubusercontent.com/sktime/pytorch-forecasting/refs/heads/main/examples/data/stallion.parquet"

data = pd.read_parquet(BASE_URL)
data.head()
data.columns

# add time index
# ensure the time index commences from zero 
# which enhances their manageability and interpretability.
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month

data["time_idx"] -= data["time_idx"].min()

"""
add additional features. month feature for possible season patterns and trends. categories have be strings
computes a logarithmic transformation of the volume column, resulting in the creation of an additional column that reflects this transformation. To maintain numerical stability and circumvent issues associated with zero values, a small constant is added during the transformation process. The logarithmic transformation plays a vital role in normalizing the data, reducing skewness, and rendering the data more suitable for various statistical models and machine learning algorithms. Such adjustments can enhance prediction accuracy and overall model performance.
"""

data["month"] = data.date.dt.month.astype(str).astype("category")
data["log_volume"] = np.log(data.volume + 1e-8)

# average metrics present valuable insights into performance trends concerning specific products and organizations over time.
data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")

# list all special_days (were int64 dtype) since we want to encode special days as one variable 
special_days = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
]
# reassign values in the special day columns: - or special day column name
data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")

data.sample(10, random_state=521)

data.describe()

data["time_idx"].max()

# Number of months to predict
max_prediction_length = 6 
# the maximum encoder length represents the extent of historical data employed in forecasting future values.
max_encoder_length = 24
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    # group identifiers, agency and sku, which may have a significant impact on the predictive outcomes
    static_categoricals=["agency", "sku"],
    # both static and time-varying features that could influence the target
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    time_varying_known_categoricals=["special_days", "month"],
    variable_groups={"special_days": special_days},
    # group of categorical variables can be treated as one variable
    time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],
    # normalization to enhance the learning stability across various groups.
    target_normalizer=GroupNormalizer(
        # use softplus and normalize by group
        groups=["agency", "sku"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = 128  
# setting it between 32 to 128 this hyperparameter influences the models performance, memory consumption, and training duration.
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
# with num_workers=0 singnify that no supplementary worker threads will be utilized for data loading. bath size is 10 times of valifation as validation typically requires less frequent updates compared to training, allowing for quicker evaluations through the use of larger batches.


# calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
# torch.cat function concatenates the target values into a single tensor for comprehensive analysis
baseline_predictions = Baseline().predict(val_dataloader)

(actuals.to(device) - baseline_predictions.to(device)).abs().mean().item()

# configure network and trainer
L.pytorch.seed_everything(42)
trainer = L.pytorch.Trainer(
    max_epochs=20,
    # accelerator="auto",
    # accelerator="mps",
    accelerator="cpu",
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,  # most important hyperparameter apart from learning rate
    # number of attention heads. Set to up to 4 for large datasets
    attention_head_size=1,
    dropout=0.1,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=8,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    # reduce learning rate if no improvement in validation loss after x epochs
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


"""
# AttributeError: 'Trainer' object has no attribute 'tuner'
res = trainer.tuner.lr_find(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()
"""

# configure network and trainer, establishes a training configuration
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = L.pytorch.Trainer(
    max_epochs=30,
    # accelerator="auto",
    # accelerator="mps",
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=30,  # coment in for training, running valiation every 30 batches
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

"""
The EarlyStopping callback is incorporated to monitor the validation loss throughout the training. Should the validation loss fail to improve by a designated amount, referred to as min_delta, over a specific number of epochs known as patience, the training process will terminate prematurely. This approach is beneficial in mitigating overfitting, particularly when the models performance declines on previously unseen data after a given point.
The LearningRateMonitor is introduced to track the learning rate employed during the training. This information offers insights into how variations in the learning rate may impact the models performance over time.
The TensorBoardLogger is initialized to record the training and validation results in a specified directory, known as lightning_logs. TensorBoard functions as a powerful visualization tool, assisting in the monitoring of various metrics, such as loss and learning rate. This facilitates a thorough analysis of the model's performance during the training phase.

The Trainer class is configured with several important parameters.
The max_epochs parameter restricts the training to a maximum of 30 epochs.
The accelerator parameter signifies that the computational resources.
The enable_model_summary option allows for the display of the model architecture summary at the commencement of training, aiding in the comprehension of the models structure.
The gradient_clip_val parameter is included to avert exploding gradients, which can destabilize the learning process. 
The limit_train_batches parameter is set to restrict training to 30 batches, a practice that can be advantageous during experimental phases to conserve training time.
"""

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

"""
The Temporal Fusion Transformer model has the capability to identify and capture intricate temporal patterns, rendering it especially useful for multi-horizon forecasting tasks (information on the target variable at multiple future points in time). 
The hidden size parameter designates the number of neurons within the hidden layers, which plays a significant role in the model's ability to learn complex representations.
The attention head size is another important consideration, as it dictates the number of attention heads utilized in the attention mechanism, permitting the model to focus on distinct segments of the input sequence concurrently.
The dropout parameter randomly deactivating a portion of the neurons and fostering the model's capacity to extract more resilient features.
The hidden continuous size parameter specifies the dimension of hidden layers dedicated to continuous features, which allows the model to engage effectively with diverse types of input data. 
The output size parameter outlines the number of quantiles for prediction, enabling the model to deliver not only point forecasts but also a spectrum of potential outcomes.
The QuantileLoss function, which aligns with the models capability to predict different quantiles of the target distribution.
The log interval parameter facilitates the recording of training metrics at specified intervals, aiding in the monitoring of the training process.
The plateau patience parameter adjust the learning rate when improvements in loss plateau.
"""

# fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
"""
The inclusion of validation checks during the training process aids in monitoring potential issues such as overfitting or underfitting, providing an opportunity to adjust the modeling parameters or techniques accordingly.
The use of DataLoaders contributes to a more efficient training process by managing data in batches. This functionality is particularly important for handling memory usage and computation time, especially when working with extensive datasets.

The tuning process typically involves experimenting with different configurations to identify the combination that yields the best results on a given dataset. This may include varying the learning rate, batch size, and the number of layers in a neural network, among other factors. Various techniques, such as grid search, random search, or more advanced methods like Bayesian optimization, can be employed to systematically search through the hyperparameter space.

Ultimately, the goal of hyperparameter tuning is to enhance the model's accuracy and generalization capability, ensuring that it performs well on unseen data. Through careful adjustment of these parameters, practitioners aim to achieve a model that is both effective and efficient in addressing the specific problem at hand.
"""
# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)


# calcualte mean absolute error on validation set
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(
    val_dataloader,
    trainer_kwargs=dict(accelerator="cpu")
    # only necessary when the model was trained on cpu
    )
(actuals - predictions).abs().mean()

# raw predictions are a dictionary from which all kind of information including quantiles can be extracted
raw_predictions_tuple = best_tft.predict(
    val_dataloader,
    mode="raw",
    return_x=True,
    trainer_kwargs=dict(accelerator="cpu")
    )


"""
isinstance(raw_predictions_tuple, tuple) # True

>>> raw_predictions_tuple[0].keys()
('prediction', 'encoder_attention', 'decoder_attention', 'static_variables', 'encoder_variables', 'decoder_variables', 'decoder_lengths', 'encoder_lengths')
>>> raw_predictions_tuple[1].keys()
dict_keys(['encoder_cat', 'encoder_cont', 'encoder_target', 'encoder_lengths', 'decoder_cat', 'decoder_cont', 'decoder_target', 'decoder_lengths', 'decoder_time_idx', 'groups', 'target_scale'])

raw_predictions_tuple[0].keys()
# ('output', 'x', 'index', 'decoder_lengths', 'y')
predictions.dtype

dir(raw_predictions_tuple)
len(raw_predictions_tuple)

"""

raw_predictions, x = raw_predictions_tuple.get('output'), raw_predictions_tuple.get('x')

for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions_tuple, idx=idx, add_loss_to_title=True)
    plt.show()


# calcualte metric by which to display
predictions = best_tft.predict(
    val_dataloader,
    trainer_kwargs=dict(accelerator="cpu")
    )
mean_losses = SMAPE(reduction="none").loss(predictions, actuals).mean(1)

# the Symmetric Mean Absolute Percentage Error (SMAPE)
# tensor.mean(dim=None, keepdim=False, *, dtype=None) returns the mean value of each row of the input tensor in the given dimension dim 
SMAPE(reduction="none").loss(predictions, actuals).shape # torch.Size([350, 6])
SMAPE(reduction="none").loss(predictions, actuals).mean(1).shape # torch.Size([350])

indices = mean_losses.argsort(descending=True)  # sort losses

for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(
        x, raw_predictions, idx=indices[idx], add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles)
    )
    plt.show()

  
predictions_tuple = best_tft.predict(
    val_dataloader,
    return_x=True,
    trainer_kwargs=dict(accelerator="cpu")
    )
predictions, x = predictions_tuple.get('output'), predictions_tuple.get('x')

predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)

best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals); plt.show()


"""
To make predictions on a specific subset of data, we utilize the filter method to extract subsequences from the dataset. In this instance, we focus on the subsequence within the training dataset that corresponds to the group identifiers Agency_01 and SKU_01. Moreover, we are interested in the first predicted value that aligns with the time index 15. 

Our intention is to generate output for all seven quantiles associated with this prediction. Consequently, we anticipate receiving a tensor with the dimensions of 1 x n_timesteps x n_quantiles, which, in this case, translates to 1 x 6 x 7. This means that we are predicting for a single subsequence six time steps into the future, while providing seven quantile estimates for each of those time steps.

The prediction is executed in a quantiles mode, which typically generates multiple quantile levels such as the median and interquartile ranges, rather than providing a singular point forecast. This multi-faceted output is instrumental in understanding the uncertainty and variability associated with the forecasts.

"""
best_tft.predict(
    training.filter(lambda x: (x.agency == "Agency_01") & (x.sku == "SKU_01") & (x.time_idx_first_prediction == 15)),
    mode="quantiles",
    trainer_kwargs=dict(accelerator="cpu")
)

raw_predictions_tuple = best_tft.predict(
    training.filter(lambda x: (x.agency == "Agency_01") & (x.sku == "SKU_01") & (x.time_idx_first_prediction == 15)),
    mode="raw",
    return_x=True,
    trainer_kwargs=dict(accelerator="cpu")
)
raw_prediction, x = raw_predictions_tuple.get('output'), raw_predictions_tuple.get('x')
best_tft.plot_prediction(x, raw_prediction, idx=0); plt.show()

# Since our dataset includes covariates, it is essential to specify the known covariates in advance when making predictions using new data.
# select last 24 months from data (max_encoder_length is 24)
encoder_data = data[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

# select last known data point and create decoder data from it by repeating it and incrementing the month
# in a real world dataset, we should not just forward fill the covariates but specify them to account
# for changes in special days and prices (which you absolutely should do but we are too lazy here)
last_data = data[lambda x: x.time_idx == x.time_idx.max()]
decoder_data = pd.concat(
    [last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i)) for i in range(1, max_prediction_length + 1)],
    ignore_index=True,
)
 
# add time index consistent with "data"
decoder_data["time_idx"] = decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
decoder_data["time_idx"] += encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()

"""
adjustments are made to the time index. It ensures that the new time indices corresponding to decoder_data commence consecutively from the maximum value obtained from a related dataset called encoder_data. This adjustment is essential in cases where decoder_data is utilized in a model that has been previously trained on encoder_data, thereby maintaining continuity in the time series data.
The importance of this coding structure lies in its application to time series analysis or machine learning, where it is critical to preserve a sequential order of time-related data for the effective performance of certain algorithms. By standardizing the time index, the code allows for the comparison or integration of the encoder and decoder datasets without any ambiguities regarding discrepancies in their respective time scales.

"""

# adjust additional time feature(s)
decoder_data["month"] = decoder_data.date.dt.month.astype(str).astype("category")  # categories have be strings

# combine encoder and decoder data
new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)



new_raw_predictions_tuple = best_tft.predict(
    new_prediction_data,
    mode="raw",
    return_x=True,
    trainer_kwargs=dict(accelerator="cpu")
)
new_raw_predictions, new_x = new_raw_predictions_tuple.get('output'), new_raw_predictions_tuple.get('x')

for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False)
    plt.show()


interpretation = best_tft.interpret_output(
    raw_predictions_tuple[0],
    reduction="sum"
    )

best_tft.plot_interpretation(interpretation)
plt.show()




dependency = best_tft.predict_dependency(
    val_dataloader.dataset, "discount_in_percent", np.linspace(0, 30, 30), show_progress_bar=True, mode="dataframe",
    trainer_kwargs=dict(accelerator="cpu")
)

# plotting median and 25% and 75% percentile
agg_dependency = dependency.groupby("discount_in_percent").normalized_prediction.agg(
    median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
)
ax = agg_dependency.plot(y="median")
ax.fill_between(agg_dependency.index, agg_dependency.q25, agg_dependency.q75, alpha=0.3); plt.show()