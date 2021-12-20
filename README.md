# aptos-kaggle

The dataset can be found in kaggle.
link is https://www.kaggle.com/c/aptos2019-blindness-detection

training is done in kaggle. After getting the models the saved model is tracked using dvc.
I converted the model from keras format to tf format for tensorflow_serving. This tf format model
also tracked using dvc.
My score is 
![Alt text](https://github.com/RavitejaBadugu/aptos-kaggle/blob/main/images/Screenshot%202021-12-20%20203909.png)

To get them first run dvc list

It gives the .dvc files.

later run dvc pull. To get the models download.

After downloading the models run docker-compose up --build.

** home page of app **
![Alt text](/images/home_page.png?raw=true)
