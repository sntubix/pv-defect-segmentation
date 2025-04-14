import sys
# Add your custom path to sys.path at the beginning to prioritize it
custom_path = "/SCDD-image-segmentation-keras"
if custom_path not in sys.path:
    sys.path.insert(0, custom_path)    
import wandb
import os
import pandas as pd
import numpy as np
from keras_segmentation.models.unet import vgg_unet, mobilenet_unet, resnet50_unet
from keras_segmentation.models.segnet import vgg_segnet, mobilenet_segnet, resnet50_segnet
from keras_segmentation.models.pspnet import vgg_pspnet
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras_segmentation.predict import evaluate_and_plot_confusion_matrix
import keras.backend as K
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Main path
main_path = os.getcwd()
dataset_path = "/home/shanifi/code/arranged_dataset_20221008"


# Train images and annotations path
train_image_path = os.path.join(dataset_path, "dataset_20221008/el_images_train")
print(train_image_path)
train_annotations_path = os.path.join(dataset_path, "dataset_20221008/el_masks_train")

# Validation images and annotations path
val_image_path = os.path.join(dataset_path, "dataset_20221008/el_images_val")
val_annotations_path = os.path.join(dataset_path, "dataset_20221008/el_masks_val")

# Test images and annotations path
test_image_path = os.path.join(dataset_path, "dataset_20221008/el_images_test")
test_annotation_dir = os.path.join(dataset_path, "dataset_20221008/el_masks_test")

# CSV file for classes
csv_path = os.path.join(dataset_path, "dataset_20221008/ListOfClassesAndColorCodes_20221008.csv")
df = pd.read_csv(csv_path)
colors = df[['Red', 'Green','Blue']].apply(lambda x: (x['Red'], x['Green'], x['Blue']), axis=1).tolist()
class_names = df['Desc'].tolist()
class_labels = df['Label'].tolist()
class_dict = {class_labels[i]: class_names[i] for i in range(len(class_labels))}

# tracking with wandb
wandb.init(
    name = "vgg_segnet_v2",
    project="scdd_segmentation_keras", 
    entity="ubix",
    config={
        "architecture": "vgg_segnet",
        "dataset": "arranged_dataset_20221008",
        "n_classes": 29,
        "input_height": 416,
        "input_width": 608,
        "epochs":65,
        "batch_size":8,
        # "steps_per_epoch": None,
        #"steps_per_epoch":len(os.listdir(train_image_path)),
        "log_every_n_batch": 1,
        "validate": True,
        "colors":colors,
        "labels_Desc":class_names,
        "labels": class_labels,
        "class_dict": class_dict,
	"val_batch_size": 16,
    })

# Checkpoint path
checkpoint_path = os.path.join(main_path, wandb.run.name, "checkpoint/")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Paths to save prediction
prediction_output_dir = os.path.join(main_path , wandb.run.name, "predictions/")
if not os.path.exists(prediction_output_dir):
    os.makedirs(prediction_output_dir)
print(prediction_output_dir)  

#focus_classes = [0, 1, 4, 14, 15]
class_weights = [
    0.8,  # "bckgnd"
    3.0,  # "sp multi"
    3.0,  # "sp mono"
    6.0,  # "sp dogbone"
    5.0,  # "ribbons"
    2.0,  # "border"
    4.0,  # "text"
    0.7,  # "padding"
    4.0,  # "clamp"
    6.0,  # "busbars"
    10.0,  # "crack rbn edge"
    5.0,  # "inactive"
    9.0,  # "rings"
    4.0,  # "material"
    7.0,  # "crack"
    5.0,  # "gridline"
    7.0,  # "splice"
    7.0,  # "dead cell"
    6.0,  # "corrosion"
    6.0,  # "belt mark"
    3.0,  # "edge dark"
    2.0,  # "frame edge"
    8.0,  # "jbox"
    9.0,  # "meas artifact"
    8.0,  # "sp mono halfcut"
    6.0,  # "scuff"
    5.0,  # "corrosion cell"
    5.0,  # "brightening"
    7.0,  # "star"
]
wandb.config.class_weights = class_weights

# Define the model 
model = vgg_segnet(n_classes=wandb.config.n_classes, input_height=wandb.config.input_height, input_width=wandb.config.input_width)

# # Define the loss function with the computed class weights
# wcce = model.WeightedCategoricalCrossentropy(weights)
# custom_loss = wcce(targets,predictions)
# # Re-compile the model with the custom loss function
# model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])


# Log total parameters
total_params = model.count_params()

# Calculate trainable and non-trainable parameters
trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
wandb.config.total_params = total_params
wandb.config.trainable_params = trainable_params
wandb.config.non_trainable_params = non_trainable_params

# Custom WandB callback to log loss and accuracy after each batch/epoch
class WandbCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Log loss and accuracy after each epoch."""
        metrics_to_log = {
            "epoch": epoch + 1,
            "loss": logs.get('loss'),
            "accuracy": logs.get('accuracy'),
            "mean_iou": logs.get('mean_io_u')
        }
        
        if wandb.config.validate:
            metrics_to_log["val_loss"] = logs.get('val_loss')
            metrics_to_log["val_accuracy"] = logs.get('val_accuracy')
            metrics_to_log["val_mean_iou"] = logs.get('val_mean_io_u')
        
        wandb.log(metrics_to_log)


    def on_batch_end(self, batch, logs=None):
        """Log loss and accuracy after each batch."""
        if batch % wandb.config.log_every_n_batch == 0:
            wandb.log({
                "batch": batch + 1,
                "batch_loss": logs.get('loss'),
                "batch_accuracy": logs.get('accuracy'),
                "batch_mean_iou": logs.get('mean_io_u')
            })

# Create ModelCheckpoint callback to save only the best model based on validation metric
checkpoint_filepath = os.path.join(checkpoint_path, "best_model.keras")
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="accuracy", 
    save_best_only=True,
    save_weights_only=False,  # Save entire model, not just weights
    save_freq="epoch", # Save model at the end of the epoch if it is the best
    verbose=1,
)

# Early stopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',        
    patience=10,                
    restore_best_weights=True 
)

# Train the model with the custom wandb callback
model.train(
    train_images=train_image_path,
    train_annotations=train_annotations_path,
    val_images=val_image_path,
    val_annotations=val_annotations_path,
    validate=wandb.config.validate,
    checkpoints_path=checkpoint_path,
    epochs=wandb.config.epochs,
    batch_size = wandb.config.batch_size,
    val_batch_size=wandb.config.val_batch_size,
    steps_per_epoch=len(os.listdir(train_image_path))//wandb.config.batch_size,
    #steps_per_epoch=256,
    val_steps_per_epoch=len(os.listdir(val_image_path)),
    callbacks=[WandbCallback(), checkpoint_callback, early_stopping],
    class_weights=class_weights,
)

# After training, save the best model as an .h5 file
best_model_h5_path = os.path.join(checkpoint_path, f"{wandb.run.name}_best_model.h5")
model.save(best_model_h5_path)
artifact = wandb.Artifact(name=wandb.run.name, type="model")
artifact.add_dir(checkpoint_path)
wandb.log_artifact(artifact)

# Predict segmentation
predictions = model.predict_multiple(
    inp_dir=test_image_path,
    out_dir=prediction_output_dir,
    class_names=class_names,
    show_legends=True,
    colors=colors,
    class_dict=class_dict,
    gt_mask_dir=test_annotation_dir,
)

all_images = {}
for out_frame in os.listdir(prediction_output_dir):
    image_name = os.path.splitext(out_frame)[0]
    all_images[image_name] = wandb.Image(os.path.join(prediction_output_dir, out_frame), caption=f"Prediction for {out_frame}")
wandb.log({"predictions": all_images})


# Log loss function and layer information
wandb.config.update({
    "loss_function": model.loss,
    "optimizer": model.optimizer.get_config() if model.optimizer else "Not Defined",
    "layers": [layer.name for layer in model.layers]
})

# evaluating the model 
evaluation_result= model.evaluate_segmentation( inp_images_dir= test_image_path , annotations_dir= test_annotation_dir)
print(evaluation_result)

cm_evaluation = evaluate_and_plot_confusion_matrix(model=model, inp_images_dir= test_image_path , annotations_dir= test_annotation_dir, class_names=class_names,)
print(cm_evaluation)

# Prepare class-wise IoU for logging
class_wise_IU = evaluation_result['class_wise_IU']
class_iou_dict = {f"{i}:{class_names[i]}_IoU": iou for i, iou in enumerate(class_wise_IU)}

# Prepare median IoU per class for logging
median_IU_per_class = evaluation_result['class_wise_median_IU']
median_iou_dict = {f"{i}:{class_names[i]}_Median_IoU": median_iou for i, median_iou in enumerate(median_IU_per_class)}

# Create a wandb.Table for class-wise IoU logging
class_wise_IU_table = wandb.Table(columns=["Class Name", "Class Index", "IoU", "Run Name"])
run_name = wandb.run.name
for i, iou in enumerate(class_wise_IU):
    class_wise_IU_table.add_data(class_names[i], i, iou, run_name)
wandb.log({"class_wise_IU_table": class_wise_IU_table})

# Log evaluation results
wandb.log({"frequency_weighted_IU": evaluation_result['frequency_weighted_IU'], 
            "mean_IU": evaluation_result['mean_IU'], 
            "class_wise_IU": class_iou_dict, 
	    "class_wise_median_IU": median_iou_dict,
            "run_name": run_name,
            })

# Finish the WandB run
wandb.finish()
