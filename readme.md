# Lymphocyte Detection using YOLOv5
 Amit Suveer, [LinkedIn](https://www.linkedin.com/in/amitsuveer)

Here I tried lymphocyte detection using YOLOv5 on publicly available dataset provided by Prof. Andrew Janowczyk. The dataset can be downloaded from the website [here](http://www.andrewjanowczyk.com/deep-learning/). The YOLOv5 model used here is downloaded from [ultranalytics git repo](https://github.com/ultralytics/yolov5) and finetuned on lymphocyte data. With the data in your hard drive, let's begin. 

### Installation and Setup

To run this solution it is important that it doesn't mess-up with your current setup. Thus we will create a separate _conda environment_ for this solution. You can use any virtual environment but in this example I am using **conda**. 

In case you don't have Anaconda or Miniconda installed, then it is recommended to install Miniconda for a blog [here](https://varhowto.com/install-miniconda-ubuntu-20-04/) in 3 steps. You can use any other source of information as well to install Miniconda, up to you. 

If you have Anaconda or Miniconda installed we can straight away create an environment called <_lymph_yolov5_> and prepare the environment for further use. To do this, open a terminal and locate the folder where you cloned this repo, then follow the set of instructions one-by-one.

```bash
# locate folder assignment_amit
cd /some_path/Lymphocyte_Detection

# create an environment
conda create --yes --name lymph_yolov5

# activate the environment
conda activate lymph_yolov5

# install pip in the environment
conda install --yes pip

# install requirements using pip (may take a few 1-2 mins)
pip install -r yolov5/requirements.txt

# adding new-env to jupyter notebook
python -m ipykernel install --user --name lymph_yolov5

# launch jupyter lab and open it in your browser
jupyter lab
```

In the _jupyter lab_ select the kernel **lymph_yolov5** and run all cells. Full code should run automatically.


To remove the <_lymph_yolov5_> conda environment type following command in your terminal. You will need to deactivate the environment first if it's activated, else call remove command straight away.

```bash
# to deactivate the environment
conda deactivate

# to remove the environment
conda env remove --name lymph_yolov5
```
