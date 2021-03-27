# Instructions for running

### Step 1: Switch to the directory
Open a CMD window and in that change directory to wherever you pull the 
code to such that you are in the "fistaReconstruction" directory.
###### example: "cd /path/to/fistaReconstruction"

### Step 2: Install dependencies
Once in the directory, run:
###### "pip3 install requirements.txt"
this will install all the dependencies needed to run all the code.

### Step 3: To launch tensorboard
Once you have installed the dependencies you can run the following command to start a tensorboard instance:
##### "tensorboard --logdir tbLogs/autoEncoder20210326-113644" (this will launch the tensorboard for the autoEncoder, if you want the classifier change autoEncoder20210326-113644 to Classifier20210326-105835)

### Step 3.5: Looking at the tensorboard
When you run Step 3, it should create a localhost webapp that you can access from any browser on the computer. To access, simply open a web browser and navigate to the url it prompts with (it should be localhost:6006).
