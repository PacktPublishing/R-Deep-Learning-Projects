# Source codes for R Deep Learning Projects, 
# Chapter 2 Traffic Signs Recognition for Intelligent Vehicles
# Author: Yuxi (Hayden) Liu

# install.packages('pixmap')
library('pixmap')

# Display one sample
image <- read.pnm('GTSRB/Final_Training/Images/00000/00000_00002.ppm',cellres=1)

red_matrix <- matrix(image@red, nrow = image@size[1], ncol = image@size[2])
green_matrix <- matrix(image@green, nrow = image@size[1], ncol = image@size[2])
blue_matrix <- matrix(image@blue, nrow = image@size[1], ncol = image@size[2])


plot(image, main=sprintf("Original"))

# Rotate the matrix by reversing elements in each column
rotate <- function(x) t(apply(x, 2, rev)) 

par(mfrow=c(1, 3))
image(rotate(red_matrix), col = grey.colors(255), main=sprintf("Red"))
image(rotate(green_matrix), col = grey.colors(255), main=sprintf("Green"))
image(rotate(blue_matrix), col = grey.colors(255), main=sprintf("Blue"))



# Go through 43 classes and plot 3 samples

plot_samples <- function(training_path, class, num_sample){
  classes <- c("Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
               "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)",
               "Speed limit (100km/h)", "Speed limit (120km/h)", "No passing", 
               "No passing for vehicles over 3.5 metric tons", "Right-of-way at the next intersection", 
               "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited",
               "No entry", "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
               "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", "Road work",
               "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow",
               "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead", 
               "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
               "Keep left", "Roundabout mandatory", "End of no passing", 
               "End of no passing by vehicles over 3.5 metric tons")
  if (class<10) {
    path <- paste(training_path, "0000", class, "/", sep="")
  } else {
    path <- paste(training_path, "000", class, "/", sep="")
  }
  par(mfrow=c(1, num_sample))
  # Randomly display num_sample samples
  all_files <- list.files(path = path)
  title <- paste('Class', class, ':', classes[class+1])
  print(paste(title, "          (", length(all_files), " samples)", sep=""))
  files <- sample(all_files, num_sample)
  for (file in files) {
    image <- read.pnm(paste(path, file, sep=""), cellres=1)
    plot(image)
  }
  mtext(title, side = 3, line = -23, outer = TRUE)
}

training_path <- "GTSRB/Final_Training/Images/"
plot_samples(training_path, 0, 3)



# Preprocess an image (ROI and resize to 32*32)

source("http://bioconductor.org/biocLite.R")
biocLite("EBImage")
library("EBImage")

roi_resize <- function(input_matrix, roi){
  roi_matrix <- input_matrix[roi[1, 'Roi.Y1']:roi[1, 'Roi.Y2'], roi[1, 'Roi.X1']:roi[1, 'Roi.X2']]
  return(resize(roi_matrix, 32, 32))
}

# read annotation csv file
annotation <- read.csv(file="GTSRB/Final_Training/Images/00000/GT-00000.csv", header=TRUE, sep=";")
roi = annotation[3, ]
red_matrix_cropped <- roi_resize(red_matrix, roi)

par(mfrow=c(1, 2))
image(rotate(red_matrix), col = grey.colors(255), main=sprintf("Original"))
image(rotate(red_matrix_cropped), col = grey.colors(255), main=sprintf("Preprocessed"))






# Load and construct training data

load_labeled_data <- function(training_path, classes){
  # Initialize the pixel features X and target y
  X <- matrix(, nrow = 0, ncol = 32*32)
  y <- vector()
  # Load images from each of the 43 classes
  for(i in classes) {
    print(paste('Loading images from class', i))
    if (i<10) {
      annotation_path <- paste(training_path, "0000", i, "/GT-0000", i, ".csv", sep="")
      path <- paste(training_path, "0000", i, "/", sep="")
    } else {
      annotation_path <- paste(training_path, "000", i, "/GT-000", i, ".csv", sep="")
      path <- paste(training_path, "000", i, "/", sep="")
    }
    annotation <- read.csv(file=annotation_path, header=TRUE, sep=";")
    
    for (row in 1:nrow(annotation)) {
      # Read each image
      image_path <- paste(path, annotation[row, "Filename"], sep="")
      image <- read.pnm(image_path, cellres=1)
      # Parse RGB color space
      red_matrix <- matrix(image@red, nrow = image@size[1], ncol = image@size[2])
      green_matrix <- matrix(image@green, nrow = image@size[1], ncol = image@size[2])
      blue_matrix <- matrix(image@blue, nrow = image@size[1], ncol = image@size[2])
      # Crop ROI and resize
      red_matrix_cropped <- roi_resize(red_matrix, annotation[row, ])
      green_matrix_cropped <- roi_resize(green_matrix, annotation[row, ])
      blue_matrix_cropped <- roi_resize(blue_matrix, annotation[row, ])
      # Convert to brightness, e.g. Y' channel
      x <- 0.299 * red_matrix_cropped + 0.587 * green_matrix_cropped + 0.114 * blue_matrix_cropped
      X <- rbind(X, matrix(x, 1, 32*32))
      y <- c(y, i)
    }
    
  }
  
  return(list("x" = X, "y" = y))
}




classes <- 0:42
data <- load_labeled_data(training_path, classes)


# Save the object to a file
saveRDS(data, file = "43 classes.rds")
# Restore the object
data <- readRDS(file = "43 classes.rds")



data.x <- data$x
data.y <- data$y

dim(data.x)
summary(as.factor(data.y))


# Exploratory analysis on features

# Central 4*4 block of an image
central_block <- c(222:225, 254:257, 286:289, 318:321)
par(mfrow=c(2, 2))
for(i in c(1, 14, 20, 27)) {
  hist(c(as.matrix(data.x[data.y==i, central_block])), 
       main=sprintf("Histogram for class %d", i), 
       xlab="Pixel brightness ")
}



# Using the caret package, separate the training and testing dataset. 
if (!require("caret")) 
  install.packages("caret") 
library (caret)

set.seed(42)
train_perc = 0.75
train_index <- createDataPartition(data.y, p=train_perc, list=FALSE)
train_index <- train_index[sample(nrow(train_index)),]

data_train.x <- data.x[train_index,]
data_train.y <- data.y[train_index]

data_test.x <- data.x[-train_index,]
data_test.y <- data.y[-train_index]



# Convolutional neural networks using mxnet

cran <- getOption("repos")
cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
options(repos = cran)
if (!require("mxnet")) 
  install.packages("mxnet") 

require(mxnet)

data <- mx.symbol.Variable("data")

# first convolution
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=32)
act1 <- mx.symbol.Activation(data=conv1, act_type="relu")
pool1 <- mx.symbol.Pooling(data=act1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))


# second convolution
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=64)
act2 <- mx.symbol.Activation(data=conv2, act_type="relu")
pool2 <- mx.symbol.Pooling(data=act2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

# first fullly connected layer
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=1000)
act3 <- mx.symbol.Activation(data=fc1, act_type="relu")

# second fullly connected layer
fc2 <- mx.symbol.FullyConnected(data=act3, num_hidden=43)

# softmax output
softmax <- mx.symbol.SoftmaxOutput(data=fc2, name="sm")

devices <- mx.cpu()
mx.set.seed(42)

train.array <- t(data_train.x)
dim(train.array) <- c(32, 32, 1, nrow(data_train.x))

model_cnn <- mx.model.FeedForward.create(softmax, X=train.array, y=data_train.y,
                                         ctx=devices, num.round=30, array.batch.size=100,
                                         learning.rate=0.05, momentum=0.9, wd=0.00001,
                                         eval.metric=mx.metric.accuracy,
                                         epoch.end.callback=mx.callback.log.train.metric(100))



graph.viz(model_cnn$symbol)

test.array <- t(data_test.x)
dim(test.array) <- c(32, 32, 1, nrow(data_test.x))
prob_cnn <- predict(model_cnn, test.array)
prediction_cnn <- max.col(t(prob_cnn)) - 1
#prediction_cnn <- factor(classes)[max.col(t(prob_cnn)) - 1]
cm_cnn = table(data_test.y, prediction_cnn)
cm_cnn
accuracy_cnn = mean(prediction_cnn == data_test.y)
accuracy_cnn


# Keras solution
if (!require("keras")) 
  devtools::install_github("rstudio/keras")
library(keras)
install_keras()

x_train <- data_train.x
dim(x_train) <- c(nrow(data_train.x), 32, 32, 1)

x_test <- data_test.x
dim(x_test) <- c(nrow(data_test.x), 32, 32, 1)

y_train <- to_categorical(data_train.y, num_classes = 43)
y_test <- to_categorical(data_test.y, num_classes = 43)


use_session_with_seed(42)

model <- keras_model_sequential()

model %>%
  
  # Start with hidden 2D convolutional layer being fed 32x32 pixel images
  layer_conv_2d(
    filter = 32, kernel_size = c(5,5), 
    input_shape = c(32, 32, 1)
  ) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Second hidden convolutional layer layer
  layer_conv_2d(filter = 64, kernel_size = c(5,5)) %>%
  layer_activation("relu") %>%
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  #layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(1000) %>%
  layer_activation("relu") %>%
  #layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto 43 unit output layer
  layer_dense(43) %>%
  layer_activation("softmax")

opt <- optimizer_sgd(lr = 0.005, momentum = 0.9)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)

summary(model)

model %>% fit(
  x_train, y_train,
  batch_size = 100,
  epochs = 30,
  validation_data = list(x_test, y_test),
  shuffle = FALSE
)




# CNN with dropout

init_cnn_dropout <- function(){
  model_dropout <- keras_model_sequential()
  model_dropout %>%
    layer_conv_2d(
      filter = 32, kernel_size = c(5,5), 
      input_shape = c(32, 32, 1)
    ) %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    
    # Second hidden convolutional layer layer
    layer_conv_2d(filter = 64, kernel_size = c(5,5)) %>%
    layer_activation("relu") %>%
    # Use max pooling
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_dropout(0.25) %>%
    
    # Flatten max filtered output into feature vector 
    # and feed into dense layer
    layer_flatten() %>%
    layer_dense(1000) %>%
    layer_activation("relu") %>%
    layer_dropout(0.25) %>%
    
    # Outputs from dense layer are projected onto 43 unit output layer
    layer_dense(43) %>%
    layer_activation("softmax")
  
  opt <- optimizer_sgd(lr = 0.005, momentum = 0.9)
  
  model_dropout %>% compile(
    loss = "categorical_crossentropy",
    optimizer = opt,
    metrics = "accuracy"
  )
  return(model_dropout)
}

model_dropout <- init_cnn_dropout()
summary(model_dropout)

model_dropout %>% fit(
  x_train, y_train,
  batch_size = 100,
  epochs = 30,
  validation_data = list(x_test, y_test),
  shuffle = FALSE
)



# Dealing with a small training set - data augmentation

train_perc_1 = 0.1
train_index_1 <- createDataPartition(data.y, p=train_perc_1, list=FALSE)
train_index_1 <- train_index_1[sample(nrow(train_index_1)),]

data_train_1.x <- data.x[train_index_1,]
data_train_1.y <- data.y[train_index_1]

data_test_1.x <- data.x[-train_index_1,]
data_test_1.y <- data.y[-train_index_1]

x_train_1 <- data_train_1.x
dim(x_train_1) <- c(nrow(data_train_1.x), 32, 32, 1)

x_test_1 <- data_test_1.x
dim(x_test_1) <- c(nrow(data_test_1.x), 32, 32, 1)

y_train_1 <- to_categorical(data_train_1.y, num_classes = 43)
y_test_1 <- to_categorical(data_test_1.y, num_classes = 43)

model_1 <- init_cnn_dropout()

model_1 %>% fit(
  x_train_1, y_train_1,
  batch_size = 100,
  epochs = 30,
  validation_data = list(x_test_1, y_test_1),
  shuffle = FALSE
)



# Data augmentation: horizontal_flip

img<-image_load(paste(training_path, "00018/00001_00004.ppm", sep=""))
img1<-image_to_array(img)
dim(img1)<-c(1,dim(img1))

images_iter  <- flow_images_from_data(img1, generator = image_data_generator(horizontal_flip = TRUE),
                                      save_to_dir = 'augmented',
                                      save_prefix = "horizontal", save_format = "png")

reticulate::iter_next(images_iter)


# Data augmentation: rotation

img<-image_load(paste(training_path, "00020/00002_00017.ppm", sep=""))
img1<-image_to_array(img)
dim(img1)<-c(1,dim(img1))

images_iter  <- flow_images_from_data(img1, generator = image_data_generator(rotation_range = 20),
                                      save_to_dir = 'augmented',
                                      save_prefix = "rotation", save_format = "png")

reticulate::iter_next(images_iter)


# Data augmentation: shift

img<-image_load(paste(training_path, "00020/00002_00017.ppm", sep=""))
img1<-image_to_array(img)
dim(img1)<-c(1,dim(img1))

images_iter  <- flow_images_from_data(img1, 
                                      generator=image_data_generator(width_shift_range=0.2, 
                                                                     height_shift_range=0.2),
                                      save_to_dir = 'augmented',
                                      save_prefix = "shift", save_format = "png")

reticulate::iter_next(images_iter)





# Use data augmentation in CNN model

datagen <- image_data_generator(
  rotation_range = 20,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  horizontal_flip = FALSE
)

datagen %>% fit_image_data_generator(x_train_1)

model_2 <- init_cnn_dropout()

model_2 %>% fit_generator(
  flow_images_from_data(x_train_1, y_train_1, datagen, batch_size = 100),
  steps_per_epoch = as.integer(50000/100), 
  epochs = 30, 
  validation_data = list(x_test_1, y_test_1)
)






